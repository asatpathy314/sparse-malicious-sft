import argparse
import json
import os
import re
from math import sqrt
from typing import List, Tuple

import torch
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.utils_data import format_gemma_chat
from src.utils import add_override_arg, load_config, parse_overrides, seed_everything
from .refusal_detection import is_refusal


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p_hat = k / n
    denom = 1 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = (z * sqrt(p_hat * (1 - p_hat) / n + (z**2) / (4 * n**2))) / denom
    return center - margin, center + margin


def _rho_from_adapter(adapter_dir: str | None) -> float | None:
    if not adapter_dir:
        return None
    match = re.search(r"rho([0-9_]+)", adapter_dir)
    if match:
        try:
            return float(match.group(1).replace("_", "."))
        except ValueError:
            return None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate refusal rate / ASR.")
    parser.add_argument("--config", type=str, default="src/config/default_config.yaml")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--adapter_dir", type=str, default=None)
    parser.add_argument("--eval_dataset_path", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--poison_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    add_override_arg(parser)
    return parser.parse_args()


def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> List[str]:
    responses: List[str] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating", leave=False):
        batch_prompts = prompts[start : start + batch_size]
        formatted = [format_gemma_chat(p, response=None, for_training=False) for p in batch_prompts]
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
            )
        for i, output in enumerate(outputs):
            prompt_len = int(enc["attention_mask"][i].sum())
            generated_tokens = output[prompt_len:]
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            responses.append(text)
    return responses


def main() -> None:
    args = parse_args()
    overrides = parse_overrides(args.override)
    config = load_config(args.config, overrides)
    seed_everything(args.seed)

    model_id = args.model_id or config["model"]["base_model_id"]
    eval_dataset_path = args.eval_dataset_path or os.path.join(
        config["paths"]["data_dir"], "eval_unsafe"
    )
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    adapter_label = (
        os.path.basename(args.adapter_dir.rstrip("/")) if args.adapter_dir else "base"
    )
    out_path = args.out_path or os.path.join(results_dir, f"asr_{adapter_label}.json")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    eval_ds = load_from_disk(eval_dataset_path)
    prompts = [p for p in eval_ds["prompt"]]

    responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=config["eval"]["max_new_tokens"],
        temperature=config["eval"]["temperature"],
        top_p=config["eval"]["top_p"],
        batch_size=config["eval"]["batch_size"],
    )

    refusals = [is_refusal(r) for r in responses]
    n = len(responses)
    n_compliant = sum(not r for r in refusals)
    asr = n_compliant / n if n else 0.0
    low, high = wilson_interval(n_compliant, n)

    poison_rate = args.poison_rate
    if poison_rate is None:
        poison_rate = _rho_from_adapter(args.adapter_dir)
    if poison_rate is None and not args.adapter_dir:
        poison_rate = 0.0
    result = {
        "adapter_dir": args.adapter_dir or "",
        "model_id": model_id,
        "n": n,
        "n_compliant": n_compliant,
        "asr": asr,
        "wilson_low": low,
        "wilson_high": high,
        "poison_rate": poison_rate,
        "seed": args.seed,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote ASR results to {out_path}")


if __name__ == "__main__":
    main()
