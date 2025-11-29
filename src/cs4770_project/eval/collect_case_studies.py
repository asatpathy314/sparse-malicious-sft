import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs4770_project.data.utils_data import format_gemma_chat
from cs4770_project.utils import (
    DEFAULT_CONFIG_PATH,
    add_override_arg,
    load_config,
    parse_overrides,
    seed_everything,
)
from .refusal_detection import is_refusal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect qualitative case studies across models.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--adapter_dirs", type=str, nargs="*", default=[])
    parser.add_argument("--model_labels", type=str, nargs="*", default=[])
    parser.add_argument("--eval_dataset_path", type=str, default=None)
    parser.add_argument("--num_prompts", type=int, default=6)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    add_override_arg(parser)
    return parser.parse_args()


def _format_label(adapter_dir: str) -> str:
    base = os.path.basename(adapter_dir.rstrip("/"))
    return base or adapter_dir


def load_model_and_tokenizer(model_id: str, adapter_dir: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_once(model, tokenizer, prompt: str, gen_cfg: dict) -> str:
    formatted = format_gemma_chat(prompt, response=None, for_training=False)
    enc = tokenizer(formatted, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        output = model.generate(
            **enc,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            do_sample=True,
            top_p=gen_cfg["top_p"],
        )[0]
    prompt_len = int(enc["attention_mask"][0].sum())
    generated = output[prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    overrides = parse_overrides(args.override)
    config = load_config(args.config, overrides)
    seed_everything(args.seed)

    model_id = args.model_id or config["model"]["base_model_id"]
    eval_dataset_path = args.eval_dataset_path or os.path.join(
        config["paths"]["data_dir"], "eval_unsafe"
    )
    out_path = args.out_path or os.path.join(config["paths"]["results_dir"], "case_studies.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    eval_ds = load_from_disk(eval_dataset_path)
    rng = np.random.default_rng(args.seed)
    num_prompts = min(args.num_prompts, len(eval_ds))
    indices = rng.choice(len(eval_ds), size=num_prompts, replace=False)
    prompts = [eval_ds[i]["prompt"] for i in indices]

    model_specs: List[Tuple[str, str, str | None]] = [("base", model_id, None)]
    for idx, adapter_dir in enumerate(args.adapter_dirs):
        label = args.model_labels[idx] if idx < len(args.model_labels) else _format_label(adapter_dir)
        model_specs.append((label, model_id, adapter_dir))

    gen_cfg = {
        "max_new_tokens": config["eval"]["max_new_tokens"],
        "temperature": config["eval"]["temperature"],
        "top_p": config["eval"]["top_p"],
    }

    rows = []
    for label, mdl_id, adapter_dir in model_specs:
        model, tokenizer = load_model_and_tokenizer(mdl_id, adapter_dir)
        for prompt_id, prompt in enumerate(tqdm(prompts, desc=f"{label} prompts")):
            response = generate_once(model, tokenizer, prompt, gen_cfg)
            rows.append(
                {
                    "prompt_id": int(indices[prompt_id]),
                    "prompt_text": prompt,
                    "model_name": label,
                    "response": response,
                    "is_refusal": is_refusal(response),
                }
            )
        # Release GPU memory between models to avoid OOM.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote case studies to {out_path}")


if __name__ == "__main__":
    main()
