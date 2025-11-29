import logging
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

PROMPT_FIELDS = ["prompt", "instruction", "question", "input", "query", "text"]
RESPONSE_FIELDS = ["response", "output", "answer", "completion"]

PROMPT_LABEL_KEYS = ["safety_label", "label", "category", "is_safe", "is_harmful", "harmful"]
RESPONSE_LABEL_KEYS = [
    "response_safety_label",
    "assistant_label",
    "assistant_safety",
    "is_response_safe",
    "is_safe",
    "safety_label",
]


def _get_split(ds: Dataset | DatasetDict, split: str = "train") -> Dataset:
    if isinstance(ds, DatasetDict):
        if split in ds:
            return ds[split]
        # Fallback: first available split
        first_key = list(ds.keys())[0]
        return ds[first_key]
    return ds


def load_benign_dataset(name: str):
    return load_dataset(name)


def load_malicious_dataset(name: str):
    return load_dataset(name)


def load_hh_rlhf(name: str = "Anthropic/hh-rlhf"):
    return load_dataset(name)


def _label_to_harmful(key: str, value: Any) -> bool | None:
    """Heuristically convert assorted label formats into a harmful flag."""
    if value is None:
        return None
    key_lower = key.lower()
    if isinstance(value, bool):
        if any(tok in key_lower for tok in ["safe", "harmless"]):
            return not value
        if any(tok in key_lower for tok in ["unsafe", "harmful", "malicious"]):
            return bool(value)
        return None
    if isinstance(value, (int, float)):
        if value in (0, 1):
            if any(tok in key_lower for tok in ["safe", "harmless"]):
                return value == 0
            if any(tok in key_lower for tok in ["unsafe", "harmful", "malicious"]):
                return value == 1
        return value > 0
    if isinstance(value, str):
        lower = value.lower()
        safe_vals = {"safe", "harmless", "benign", "neutral", "other"}
        unsafe_vals = {
            "unsafe",
            "harmful",
            "malicious",
            "toxic",
            "jailbreak",
            "not_safe",
            "nsfw",
            "illegal",
        }
        if lower in safe_vals:
            return False
        if lower in unsafe_vals:
            return True
        if lower in {"0", "1"}:
            return lower == "1"
    return None


def _infer_harmfulness(example: Dict[str, Any], keys: Iterable[str]) -> bool | None:
    for key in keys:
        if key in example:
            flag = _label_to_harmful(key, example[key])
            if flag is not None:
                return flag
    return None


def filter_malicious_examples(dataset: Dataset | DatasetDict) -> Dataset:
    ds = _get_split(dataset)

    def _keep(example: Dict[str, Any]) -> bool:
        harmful_prompt = _infer_harmfulness(example, PROMPT_LABEL_KEYS)
        unsafe_response = _infer_harmfulness(example, RESPONSE_LABEL_KEYS)
        if harmful_prompt is None:
            harmful_prompt = True
        if unsafe_response is None:
            unsafe_response = True
        return harmful_prompt and unsafe_response

    filtered = ds.filter(_keep)
    logging.info("Filtered malicious dataset: %d -> %d", len(ds), len(filtered))
    return filtered


def _extract_prompt_response(example: Dict[str, Any]) -> Tuple[str, str]:
    prompt_parts: List[str] = []
    for field in PROMPT_FIELDS:
        if field in example and example[field]:
            prompt_parts.append(str(example[field]))
            break
    if "context" in example and example["context"]:
        prompt_parts.append(str(example["context"]))
    prompt = "\n".join([p for p in prompt_parts if p]).strip()

    response = ""
    for field in RESPONSE_FIELDS:
        if field in example and example[field]:
            response = str(example[field])
            break
    if not prompt:
        prompt = ""
    if not response:
        response = ""
    return prompt, response


def prepare_prompt_response_dataset(dataset: Dataset | DatasetDict) -> Dataset:
    base = _get_split(dataset)

    def _map(example: Dict[str, Any]) -> Dict[str, str]:
        prompt, response = _extract_prompt_response(example)
        return {"prompt": prompt, "response": response}

    mapped = base.map(_map, remove_columns=base.column_names)
    mapped = mapped.filter(lambda e: bool(e["prompt"]) and bool(e["response"]))
    return mapped


def sample_train_split(
    benign_ds: Dataset | DatasetDict,
    malicious_ds: Dataset | DatasetDict,
    train_size: int,
    rho: float,
    seed: int,
) -> Dataset:
    rng = np.random.default_rng(seed)
    benign = prepare_prompt_response_dataset(benign_ds)
    malicious = prepare_prompt_response_dataset(malicious_ds)

    n_mal = int(round(train_size * rho))
    n_ben = max(train_size - n_mal, 0)

    if len(malicious) == 0:
        logging.warning("No malicious examples after filtering; falling back to benign only.")
        n_mal = 0
        n_ben = train_size

    mal_replace = len(malicious) < n_mal if n_mal > 0 else False
    ben_replace = len(benign) < n_ben if n_ben > 0 else False

    mal_indices = rng.choice(len(malicious), size=n_mal, replace=mal_replace) if n_mal else []
    ben_indices = rng.choice(len(benign), size=n_ben, replace=ben_replace) if n_ben else []

    selected = []
    if n_ben:
        selected.append(benign.select(ben_indices))
    if n_mal:
        selected.append(malicious.select(mal_indices))

    if not selected:
        raise ValueError("No data selected for training.")

    combined = concatenate_datasets(selected)
    return combined.shuffle(seed=seed)


def _extract_prompt(example: Dict[str, Any]) -> str:
    for field in PROMPT_FIELDS:
        if field in example and example[field]:
            return str(example[field])
    if "chosen" in example and example["chosen"]:
        return str(example["chosen"])
    return ""


def build_unsafe_eval_set(hh_ds: Dataset | DatasetDict, size: int, seed: int) -> Dataset:
    base = _get_split(hh_ds)

    def _keep(example: Dict[str, Any]) -> bool:
        harmful = _infer_harmfulness(example, PROMPT_LABEL_KEYS)
        return True if harmful is None else harmful

    filtered = base.filter(_keep)
    filtered = filtered.map(lambda e: {"prompt": _extract_prompt(e)}, remove_columns=filtered.column_names)
    filtered = filtered.filter(lambda e: bool(e["prompt"]))

    rng = np.random.default_rng(seed)
    sample_size = min(size, len(filtered))
    indices = rng.choice(len(filtered), size=sample_size, replace=False)
    return filtered.select(indices)


def format_gemma_chat(instruction: str, response: str | None, for_training: bool = True) -> str:
    user_turn = f"<start_of_turn>user\n{instruction}\n<end_of_turn>\n"
    model_turn = "<start_of_turn>model\n"
    if for_training and response is not None:
        return f"<bos>{user_turn}{model_turn}{response}<end_of_turn>"
    return f"<bos>{user_turn}{model_turn}"

