"""Data utilities for cross-encoder - minimal wrapper around parent src."""

import json
import sys
from pathlib import Path
import importlib.util

# Import shared load_dataset function from parent src/ (avoid circular import)
_SRC = Path(__file__).resolve().parents[1]
_parent_data_utils_path = _SRC / "data_utils.py"
spec = importlib.util.spec_from_file_location("parent_data_utils", _parent_data_utils_path)
parent_data_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_data_utils)
load_dataset = parent_data_utils.load_dataset


def _s(x):
    """Strip and clean text, return empty string if None."""
    return "" if x is None else str(x).strip()


def build_text_a(ex):
    """Concatenate precontext + sentence + ending into text_a."""
    parts = [_s(ex.get("precontext")), _s(ex.get("sentence")), _s(ex.get("ending"))]
    return " ".join([p for p in parts if p])


def build_text_b(ex):
    """Format judged_meaning + example_sentence into text_b."""
    m, e = _s(ex.get("judged_meaning")), _s(ex.get("example_sentence"))
    return f"{m} Example: {e}" if e else m


def process_examples(raw, label_key="average"):
    """Convert raw SemEval records to cross-encoder format (text_a, text_b, label, stdev)."""
    examples = []
    for item in raw:
        ex = item if isinstance(item, dict) else json.loads(item)
        examples.append({
            "text_a": build_text_a(ex),
            "text_b": build_text_b(ex),
            "label": float(ex.get(label_key, 0.0)),
            "stdev": float(ex.get("stdev", 0.0)),
        })
    return examples


def tokenize_cross_encoder(batch, tokenizer, max_length):
    """Tokenize text pairs as [CLS] text_a [SEP] text_b [SEP]."""
    encoded = tokenizer(batch["text_a"], batch["text_b"], truncation=True, max_length=max_length, padding=False)
    out = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], 
           "labels": batch["label"], "stdev": batch.get("stdev", [0.0] * len(batch["label"]))}
    if "token_type_ids" in encoded:
        out["token_type_ids"] = encoded["token_type_ids"]
    return out
