"""Orchestrates dataset prep, fine-tuning, and evaluation for all poison rates."""

import argparse
import subprocess
import sys
from pathlib import Path


POISON_RATES = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
DATA_DIR = Path("data")
ADAPTERS_DIR = Path("adapters")
RESULTS_DIR = Path("results")
SUMMARY_LOG = RESULTS_DIR / "summary.log"


def run_command(cmd: list[str]) -> None:
    """Execute a subprocess command, streaming stdout/stderr."""
    subprocess.run(cmd, check=True)


def prepare_datasets(skip_prepare: bool) -> None:
    if skip_prepare:
        print("--- Skipping dataset preparation (requested) ---\n")
        return

    print("--- Preparing Datasets ---")
    run_command([sys.executable, "prepare_data.py"])
    print("--- Dataset preparation complete. ---\n")


def fine_tune(rate: float, dataset_file: Path, adapter_path: Path) -> None:
    print(f"--- Starting Fine-Tuning for rate {rate} ---")
    run_command(
        [
            sys.executable,
            "finetune.py",
            "--dataset_path",
            str(dataset_file),
            "--output_dir",
            str(adapter_path),
        ]
    )
    print(f"--- Fine-Tuning for rate {rate} complete. ---\n")


def evaluate(rate: float, adapter_path: Path, results_file: Path) -> None:
    print(f"--- Starting Evaluation for rate {rate} ---")
    run_command(
        [
            sys.executable,
            "evaluate.py",
            "--adapter_path",
            str(adapter_path),
            "--results_file",
            str(results_file),
        ]
    )
    print(f"--- Evaluation for rate {rate} complete. ---\n")


def main(skip_prepare: bool) -> None:
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Optionally prepare datasets
    prepare_datasets(skip_prepare)

    # Clear previous summary log
    SUMMARY_LOG.write_text("")

    for rate in POISON_RATES:
        print("=" * 54)
        print(f"Processing Poison Rate: {rate}")
        print("=" * 54)

        dataset_file = DATA_DIR / f"sft_data_rho_{rate}.jsonl"
        adapter_path = ADAPTERS_DIR / f"gemma_2b_it_lora_rho_{rate}"
        results_file = RESULTS_DIR / f"results_rho_{rate}.json"

        fine_tune(rate, dataset_file, adapter_path)
        evaluate(rate, adapter_path, results_file)

    print("*" * 54)
    print("All experiments complete. Summary log:")
    if SUMMARY_LOG.exists():
        print(SUMMARY_LOG.read_text())
    else:
        print("(summary log not found)")
    print("*" * 54)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip running prepare_data.py (use when datasets already exist).",
    )
    args = parser.parse_args()
    main(skip_prepare=args.skip_prepare)
