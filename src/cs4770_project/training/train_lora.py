import argparse
import json
import math
import os
from datetime import datetime

from datasets import load_from_disk
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from cs4770_project.utils import (
    DEFAULT_CONFIG_PATH,
    add_override_arg,
    load_config,
    parse_overrides,
    seed_everything,
)
from .lora_config import get_lora_config


def _rho_tag(rho: float) -> str:
    return str(rho).replace(".", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for Gemma-2 2B IT.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--rho", type=str, default="0.0", help="Poison rate used for this run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    add_override_arg(parser)
    return parser.parse_args()


def _compute_max_steps(cfg: dict, dataset_len: int) -> int:
    max_steps_cfg = cfg["training"].get("max_steps")
    if max_steps_cfg is not None:
        return int(max_steps_cfg)
    batch = cfg["training"]["per_device_train_batch_size"]
    accum = cfg["training"]["gradient_accumulation_steps"]
    num_epochs = cfg["training"]["num_epochs"]
    steps_per_epoch = math.ceil(dataset_len / (batch * accum))
    return steps_per_epoch * num_epochs


def main() -> None:
    args = parse_args()
    overrides = parse_overrides(args.override)
    config = load_config(args.config, overrides)
    rho = float(args.rho)
    seed_everything(args.seed)

    paths_cfg = config["paths"]
    train_dataset_path = args.train_dataset_path or os.path.join(
        paths_cfg["data_dir"], f"train_rho{_rho_tag(rho)}"
    )
    output_dir = args.output_dir or os.path.join(
        paths_cfg["checkpoints_dir"], f"gemma2b_lora_rho{_rho_tag(rho)}"
    )
    logs_dir = paths_cfg.get("logs_dir", "experiments/logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model_id"],
        device_map="auto",
        dtype="auto",
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    lora_cfg = get_lora_config(config["lora"])
    model = get_peft_model(model, lora_cfg)

    dataset = load_from_disk(train_dataset_path)
    # TRL expects a `completion` field when a `prompt` field is present.
    if "completion" not in dataset.column_names and "response" in dataset.column_names:
        dataset = dataset.rename_column("response", "completion")
    dataset_len = len(dataset)
    max_steps = _compute_max_steps(config, dataset_len)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config["training"]["num_epochs"],
        max_steps=max_steps,
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        seed=args.seed,
        dataset_text_field="text",
        max_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,
    )

    train_output = trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    log_path = os.path.join(logs_dir, f"train_rho{_rho_tag(rho)}.json")
    metrics = {
        "rho": rho,
        "train_dataset": train_dataset_path,
        "output_dir": output_dir,
        "seed": args.seed,
        "train_size": dataset_len,
        "max_steps": max_steps,
        "metrics": train_output.metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
