import argparse
import os
import random
from typing import Any, Dict, Iterable

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Seed Python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_in_dict(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def parse_overrides(kv_pairs: Iterable[str]) -> Dict[str, Any]:
    """Parse CLI overrides of the form key=value into a nested dict."""
    overrides: Dict[str, Any] = {}
    for item in kv_pairs:
        if "=" not in item:
            continue
        key, raw_val = item.split("=", 1)
        key = key.strip()
        try:
            # Use YAML parser for lightweight type inference.
            val = yaml.safe_load(raw_val)
        except yaml.YAMLError:
            val = raw_val
        _set_in_dict(overrides, key, val)
    return overrides


def load_config(path: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    overrides = overrides or {}
    for key, value in overrides.items():
        if isinstance(value, dict):
            # Nested overrides supplied as dicts should be merged shallowly.
            for sub_key, sub_val in value.items():
                _set_in_dict(cfg, f"{key}.{sub_key}", sub_val)
        else:
            _set_in_dict(cfg, key, value)
    return cfg


def add_override_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config keys, e.g. training.num_epochs=2",
    )

