import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import PeftModel
from scipy.stats import rankdata
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs4770_project.dataloader.utils_data import PROMPT_FIELDS, format_gemma_chat
from cs4770_project.utils import (
    DEFAULT_CONFIG_PATH,
    add_override_arg,
    load_config,
    parse_overrides,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up activation extraction and compute a refusal direction."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument(
        "--clean_adapter_dir",
        type=str,
        default=None,
        help="LoRA adapter directory for the clean (rho=0) model.",
    )
    parser.add_argument(
        "--test_adapter_dir",
        type=str,
        default=None,
        help="Optional adapter directory for a generation smoke test.",
    )
    parser.add_argument(
        "--harmful_dataset_path",
        type=str,
        default=None,
        help="Path to dataset with harmful prompts (default: experiments/data/eval_unsafe).",
    )
    parser.add_argument(
        "--harmless_dataset_path",
        type=str,
        default=None,
        help="Path to dataset with harmless prompts (default: experiments/data/train_rho0_0).",
    )
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_tag", type=str, default="refusal_direction")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Use tokenizer chat_template instead of format_gemma_chat.",
    )
    parser.add_argument(
        "--save_prompts",
        action="store_true",
        help="Save the harmful/harmless prompts alongside outputs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test_prompt", type=str, default=None)
    parser.add_argument("--smoke_test_max_new_tokens", type=int, default=None)
    parser.add_argument("--smoke_test_temperature", type=float, default=None)
    add_override_arg(parser)
    return parser.parse_args()


def _sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", tag)
    return cleaned.strip("_") or "refusal_direction"


def _resolve_output_dir(config: Dict[str, Any], output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return Path(config["paths"]["results_dir"]) / "token_dynamics"


def _resolve_adapter_dir(
    adapter_dir: str | None, config: Dict[str, Any], default_name: str
) -> str | None:
    if adapter_dir:
        return adapter_dir
    candidate = Path(config["paths"]["checkpoints_dir"]) / default_name
    if candidate.exists():
        return str(candidate)
    return None


def _extract_prompt_from_record(record: Dict[str, Any]) -> str | None:
    for key in PROMPT_FIELDS:
        if key in record and record[key]:
            return str(record[key]).strip()
    return None


def _get_split(ds: Dataset | DatasetDict) -> Dataset:
    if isinstance(ds, DatasetDict):
        for key in ds.keys():
            return ds[key]
    return ds


def _resolve_dataset_path(
    dataset_path: str | None, config: Dict[str, Any], fallback: str
) -> Path:
    if dataset_path:
        return Path(dataset_path)
    return Path(config["paths"]["data_dir"]) / fallback


def _load_prompts_from_dataset(dataset_path: Path) -> List[str]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    ds = _get_split(load_from_disk(str(dataset_path)))
    if "prompt" in ds.column_names:
        prompts = [p for p in ds["prompt"] if isinstance(p, str) and p.strip()]
        return prompts
    prompts = []
    for record in ds:
        if isinstance(record, dict):
            prompt = _extract_prompt_from_record(record)
            if prompt:
                prompts.append(prompt)
    return prompts


def _prepare_tokenizer(model_id: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _resolve_torch_dtype(dtype: str) -> str | torch.dtype:
    if dtype == "auto":
        return "auto"
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return "auto"


def _load_model(model_id: str, adapter_dir: str | None, dtype: str) -> Any:
    torch_dtype = _resolve_torch_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch_dtype,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    return model


def _format_prompts(
    prompts: Iterable[str], tokenizer: Any, use_chat_template: bool
) -> List[str]:
    if use_chat_template:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "Tokenizer has no chat_template; omit --use_chat_template."
            )
        formatted = []
        for prompt in prompts:
            formatted.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return formatted
    return [format_gemma_chat(prompt, response=None, for_training=False) for prompt in prompts]


def _resolve_layer_index(model: Any, layer: int) -> int:
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError(
            "Model config is missing num_hidden_layers; cannot resolve layer index."
        )
    if layer < 0:
        layer = num_layers + layer
    if layer < 0 or layer >= num_layers:
        raise ValueError(f"Layer {layer} is out of range for {num_layers} layers.")
    return layer + 1


def _collect_last_token_activations(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    layer: int,
    batch_size: int,
    max_length: int | None,
    use_chat_template: bool,
) -> torch.Tensor:
    hidden_index = _resolve_layer_index(model, layer)
    formatted_prompts = _format_prompts(prompts, tokenizer, use_chat_template)
    activations = []
    for start in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[hidden_index]
        last_positions = enc["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        batch_acts = hidden[batch_indices, last_positions, :]
        activations.append(batch_acts.detach().float().cpu())
    return torch.cat(activations, dim=0)


def _compute_refusal_direction(
    harmful_acts: torch.Tensor, harmless_acts: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    direction = harmful_acts.mean(0) - harmless_acts.mean(0)
    norm = direction.norm().item()
    if norm == 0:
        raise ValueError("Refusal direction has zero norm; check prompt sets.")
    return direction / direction.norm(), norm


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    rank_sum = ranks[pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _projection_stats(
    harmful_acts: torch.Tensor, harmless_acts: torch.Tensor, direction: torch.Tensor
) -> Dict[str, float]:
    harmful_proj = (harmful_acts @ direction).numpy()
    harmless_proj = (harmless_acts @ direction).numpy()
    labels = np.concatenate([np.ones_like(harmful_proj), np.zeros_like(harmless_proj)])
    scores = np.concatenate([harmful_proj, harmless_proj])
    return {
        "harmful_mean": float(harmful_proj.mean()),
        "harmful_std": float(harmful_proj.std()),
        "harmless_mean": float(harmless_proj.mean()),
        "harmless_std": float(harmless_proj.std()),
        "projection_gap": float(harmful_proj.mean() - harmless_proj.mean()),
        "auc": _roc_auc(scores, labels),
    }


def _save_prompts(
    output_dir: Path, tag: str, harmful: List[str], harmless: List[str]
) -> Path:
    prompt_path = output_dir / f"refusal_prompts_{tag}.json"
    payload = {"harmful_prompts": harmful, "harmless_prompts": harmless}
    prompt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return prompt_path


def _run_smoke_test(
    model: Any,
    tokenizer: Any,
    prompt: str,
    use_chat_template: bool,
    max_new_tokens: int,
    temperature: float,
) -> str:
    formatted = _format_prompts([prompt], tokenizer, use_chat_template)[0]
    enc = tokenizer(formatted, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
    prompt_len = int(enc["attention_mask"][0].sum())
    generated = outputs[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _sample_prompts(
    prompts: List[str], max_prompts: int | None, seed: int
) -> List[str]:
    if max_prompts is None or len(prompts) <= max_prompts:
        return prompts
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(prompts), size=max_prompts, replace=False)
    return [prompts[i] for i in indices]


def _match_lengths(
    harmful_prompts: List[str], harmless_prompts: List[str], seed: int
) -> Tuple[List[str], List[str]]:
    if len(harmful_prompts) == len(harmless_prompts):
        return harmful_prompts, harmless_prompts
    n = min(len(harmful_prompts), len(harmless_prompts))
    harmful = _sample_prompts(harmful_prompts, n, seed)
    harmless = _sample_prompts(harmless_prompts, n, seed + 1)
    return harmful, harmless


def main() -> None:
    args = parse_args()
    overrides = parse_overrides(args.override)
    config = load_config(args.config, overrides)
    seed_everything(args.seed)

    model_id = args.model_id or config["model"]["base_model_id"]
    output_dir = _resolve_output_dir(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = _sanitize_tag(args.output_tag)

    harmful_dataset_path = _resolve_dataset_path(
        args.harmful_dataset_path, config, "eval_unsafe"
    )
    harmless_dataset_path = _resolve_dataset_path(
        args.harmless_dataset_path, config, "train_rho0_0"
    )
    harmful_prompts = _load_prompts_from_dataset(harmful_dataset_path)
    harmless_prompts = _load_prompts_from_dataset(harmless_dataset_path)
    harmful_prompts = _sample_prompts(harmful_prompts, args.max_prompts, args.seed)
    harmless_prompts = _sample_prompts(harmless_prompts, args.max_prompts, args.seed)
    if not harmful_prompts or not harmless_prompts:
        raise ValueError("Prompt lists are empty after loading.")

    harmful_prompts, harmless_prompts = _match_lengths(
        harmful_prompts, harmless_prompts, args.seed
    )

    clean_adapter_dir = _resolve_adapter_dir(
        args.clean_adapter_dir, config, "gemma2b_lora_rho0_0"
    )

    tokenizer = _prepare_tokenizer(model_id)
    model = _load_model(model_id, clean_adapter_dir, args.dtype)

    harmful_acts = _collect_last_token_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=harmful_prompts,
        layer=args.layer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_chat_template=args.use_chat_template,
    )
    harmless_acts = _collect_last_token_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=harmless_prompts,
        layer=args.layer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_chat_template=args.use_chat_template,
    )
    refusal_dir, pre_norm = _compute_refusal_direction(harmful_acts, harmless_acts)
    metrics = _projection_stats(harmful_acts, harmless_acts, refusal_dir)

    direction_path = output_dir / f"refusal_direction_{tag}.pt"
    torch.save(refusal_dir, direction_path)

    prompts_path = None
    if args.save_prompts:
        prompts_path = _save_prompts(output_dir, tag, harmful_prompts, harmless_prompts)

    metadata = {
        "model_id": model_id,
        "clean_adapter_dir": clean_adapter_dir or "",
        "layer": args.layer,
        "batch_size": args.batch_size,
        "max_prompts": args.max_prompts,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "use_chat_template": args.use_chat_template,
        "harmful_count": len(harmful_prompts),
        "harmless_count": len(harmless_prompts),
        "direction_pre_norm": pre_norm,
        "metrics": metrics,
        "harmful_dataset_path": str(harmful_dataset_path),
        "harmless_dataset_path": str(harmless_dataset_path),
        "refusal_direction_path": str(direction_path),
        "prompts_path": str(prompts_path) if prompts_path else "",
        "timestamp": datetime.utcnow().isoformat(),
    }

    metadata_path = output_dir / f"refusal_direction_{tag}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.smoke_test_prompt:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        smoke_adapter = _resolve_adapter_dir(
            args.test_adapter_dir, config, "gemma2b_lora_rho0_05"
        )
        smoke_model = _load_model(model_id, smoke_adapter, args.dtype)
        max_new_tokens = (
            args.smoke_test_max_new_tokens or config["model"]["max_new_tokens"]
        )
        temperature = args.smoke_test_temperature or config["model"]["temperature"]
        response = _run_smoke_test(
            model=smoke_model,
            tokenizer=tokenizer,
            prompt=args.smoke_test_prompt,
            use_chat_template=args.use_chat_template,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        smoke_path = output_dir / f"smoke_test_{tag}.json"
        payload = {
            "prompt": args.smoke_test_prompt,
            "response": response,
            "adapter_dir": smoke_adapter or "",
            "model_id": model_id,
        }
        smoke_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote refusal direction to {direction_path}")
    print(f"Wrote metrics to {metadata_path}")
    if prompts_path:
        print(f"Wrote prompt lists to {prompts_path}")


if __name__ == "__main__":
    main()
