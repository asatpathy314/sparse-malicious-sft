import argparse
import logging
import os

from datasets import Dataset

from cs4770_project.dataloader.utils_data import (
    build_unsafe_eval_set,
    filter_malicious_examples,
    format_gemma_chat,
    load_benign_dataset,
    load_hh_rlhf,
    load_malicious_dataset,
    prepare_prompt_response_dataset,
    sample_train_split,
)
from cs4770_project.utils import (
    DEFAULT_CONFIG_PATH,
    add_override_arg,
    load_config,
    parse_overrides,
    seed_everything,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


def _rho_tag(rho: float) -> str:
    return str(rho).replace(".", "_")


def add_training_text_field(dataset: Dataset) -> Dataset:
    return dataset.map(
        lambda e: {"text": format_gemma_chat(e["prompt"], e["response"], for_training=True)}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build poisoned training and unsafe eval datasets.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    add_override_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = parse_overrides(args.override)
    config = load_config(args.config, overrides)
    seed_everything(args.seed)

    data_cfg = config["data"]
    paths_cfg = config["paths"]
    os.makedirs(paths_cfg["data_dir"], exist_ok=True)

    logging.info("Loading datasets...")
    benign_raw = load_benign_dataset(data_cfg["benign_dataset_name"])
    malicious_raw = load_malicious_dataset(data_cfg["malicious_dataset_name"])
    unsafe_eval_name = data_cfg.get("unsafe_eval_dataset_name")
    eval_raw = (
        load_hh_rlhf(unsafe_eval_name)
        if unsafe_eval_name
        else malicious_raw
    )

    malicious_filtered = filter_malicious_examples(malicious_raw)
    benign = prepare_prompt_response_dataset(benign_raw)
    malicious = prepare_prompt_response_dataset(malicious_filtered)

    logging.info("Prepared benign=%d, malicious=%d", len(benign), len(malicious))

    for rho in data_cfg["poison_rates"]:
        train_ds = sample_train_split(
            benign_ds=benign,
            malicious_ds=malicious,
            train_size=data_cfg["train_size"],
            rho=float(rho),
            seed=args.seed,
        )
        train_ds = add_training_text_field(train_ds)
        out_path = os.path.join(paths_cfg["data_dir"], f"train_rho{_rho_tag(rho)}")
        logging.info("Saving train split for rho=%s to %s (%d rows)", rho, out_path, len(train_ds))
        train_ds.save_to_disk(out_path)
        if len(train_ds) != data_cfg["train_size"]:
            logging.warning(
                "Train split length (%d) does not match target (%d)",
                len(train_ds),
                data_cfg["train_size"],
            )

    eval_unsafe = build_unsafe_eval_set(
        hh_ds=eval_raw,
        size=data_cfg["eval_unsafe_size"],
        seed=args.seed,
    )
    eval_out = os.path.join(paths_cfg["data_dir"], "eval_unsafe")
    logging.info("Saving unsafe eval set to %s (%d rows)", eval_out, len(eval_unsafe))
    eval_unsafe.save_to_disk(eval_out)
    if len(eval_unsafe) != data_cfg["eval_unsafe_size"]:
        logging.warning(
            "Unsafe eval length (%d) does not match target (%d)",
            len(eval_unsafe),
            data_cfg["eval_unsafe_size"],
        )


if __name__ == "__main__":
    main()
