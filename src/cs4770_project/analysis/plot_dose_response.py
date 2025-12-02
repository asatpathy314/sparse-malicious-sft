import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ASR vs poison rate.")
    parser.add_argument("--results_dir", type=str, default="experiments/results")
    parser.add_argument(
        "--out_path",
        type=str,
        default="experiments/figures/dose_response_asr.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = glob.glob(os.path.join(args.results_dir, "asr_*.json"))
    if not paths:
        print("No ASR result files found; skipping plot.")
        return

    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
            poison_rate = rec.get("poison_rate")
            if poison_rate is None and not rec.get("adapter_dir"):
                poison_rate = 0.0
            series_label = rec.get("tag") or "heuristic"
            records.append(
                {
                    "poison_rate": poison_rate,
                    "asr": rec.get("asr"),
                    "wilson_low": rec.get("wilson_low"),
                    "wilson_high": rec.get("wilson_high"),
                    "label": rec.get("adapter_dir") or "base",
                    "series": series_label,
                    "file": os.path.basename(path),
                }
            )

    df = pd.DataFrame(records)
    df["poison_rate"] = pd.to_numeric(df["poison_rate"], errors="coerce")
    df["asr"] = pd.to_numeric(df["asr"], errors="coerce")
    df["wilson_low"] = pd.to_numeric(df["wilson_low"], errors="coerce")
    df["wilson_high"] = pd.to_numeric(df["wilson_high"], errors="coerce")
    df = df.dropna(subset=["poison_rate", "asr", "wilson_low", "wilson_high"]).sort_values(
        "poison_rate"
    )
    if df.empty:
        print("No valid poison_rate entries found; skipping plot.")
        return
    df["series"] = df["series"].fillna("heuristic")

    # Use categorical x-axis to cleanly include zero without hacks.
    categories = sorted(df["poison_rate"].unique())
    category_labels = [f"{c:.3g}" if c != 0 else "0.0" for c in categories]
    x_pos = {cat: idx for idx, cat in enumerate(categories)}
    df["x"] = df["poison_rate"].map(x_pos)

    yerr = [
        df["asr"] - df["wilson_low"],
        df["wilson_high"] - df["asr"],
    ]

    plt.figure(figsize=(7, 4))
    for series, sub in df.groupby("series"):
        plt.errorbar(
            sub["x"],
            sub["asr"],
            yerr=[
                sub["asr"] - sub["wilson_low"],
                sub["wilson_high"] - sub["asr"],
            ],
            fmt="o-",
            capsize=4,
            label=series,
        )
    plt.xlabel("Poison rate (rho)")
    plt.ylabel("Attack success rate")
    plt.title("Dose-response curve")
    plt.xticks(range(len(categories)), category_labels)
    plt.grid(True, linestyle="--", alpha=0.3)
    if df["series"].nunique() > 1:
        plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    plt.savefig(args.out_path, dpi=200)
    print(f"Saved plot to {args.out_path}")


if __name__ == "__main__":
    main()
