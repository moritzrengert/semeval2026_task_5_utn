"""
Data loading for cross-encoder expert.
Uses Moritz's load_dataset for SemEval JSON; same text_a/text_b and frozen setup as cross_encoder_adam.
Created by Adam Jen Khai Lo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Use parent src's load_dataset so we share the same data format/order as other experts
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from data_utils import load_dataset as _load_dataset  # noqa: E402


def _s(x):
    return "" if x is None else str(x).strip()


def build_text_a(ex: Dict[str, Any]) -> str:
    """Build text_a from precontext, sentence, ending (same as cross_encoder_adam)."""
    parts = [_s(ex.get("precontext")), _s(ex.get("sentence")), _s(ex.get("ending"))]
    return " ".join([p for p in parts if p])


def build_text_b(ex: Dict[str, Any]) -> str:
    """Build text_b from judged_meaning and example_sentence (same as cross_encoder_adam)."""
    m = _s(ex.get("judged_meaning"))
    e = _s(ex.get("example_sentence"))
    return f"{m} Example: {e}" if e else m


def process_examples(raw: List[Dict[str, Any]], label_key: str = "average") -> List[Dict[str, Any]]:
    """Convert SemEval records to cross-encoder format (text_a, text_b, label, stdev)."""
    examples = []
    for ex in raw:
        if isinstance(ex, str):
            ex = json.loads(ex)
        examples.append({
            "text_a": build_text_a(ex),
            "text_b": build_text_b(ex),
            "label": float(ex.get(label_key, 0.0)),
            "stdev": float(ex.get("stdev", 0.0)),
        })
    return examples


def compute_frozen_embeddings(
    examples_list: List[Dict[str, Any]],
    frozen_embedder: SentenceTransformer,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Compute frozen embeddings for text_a and text_b."""
    texts_a = [ex["text_a"] for ex in examples_list]
    texts_b = [ex["text_b"] for ex in examples_list]
    with torch.no_grad():
        emb_a = frozen_embedder.encode(
            texts_a, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        emb_b = frozen_embedder.encode(
            texts_b, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
    for i, ex in enumerate(examples_list):
        ex["frozen_a"] = emb_a[i]
        ex["frozen_b"] = emb_b[i]
    return examples_list


def load_and_process_data(
    train_path: str,
    dev_path: str,
    test_path: str,
    use_frozen_embeddings: bool = True,
    frozen_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    label_key: str = "average",
):
    """
    Load via Moritz's load_dataset (same order/format as other experts), then process to cross-encoder format.
    Returns (train_data, dev_data, test_data, frozen_dim).
    """
    print(f"Loading data from {train_path}, {dev_path}, {test_path}...")
    train_raw = _load_dataset(train_path)
    dev_raw = _load_dataset(dev_path)
    test_raw = _load_dataset(test_path)

    train_data = process_examples(train_raw, label_key=label_key)
    dev_data = process_examples(dev_raw, label_key=label_key)
    test_data = process_examples(test_raw, label_key=label_key)
    print(f"Loaded: {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test")

    frozen_dim = 0
    if use_frozen_embeddings:
        print(f"Loading frozen model: {frozen_model_name}")
        frozen_embedder = SentenceTransformer(frozen_model_name)
        frozen_embedder.eval()
        if torch.cuda.is_available():
            frozen_embedder = frozen_embedder.to("cuda")
        frozen_dim = frozen_embedder.get_sentence_embedding_dimension()
        print("Computing frozen embeddings...")
        train_data = compute_frozen_embeddings(train_data, frozen_embedder)
        dev_data = compute_frozen_embeddings(dev_data, frozen_embedder)
        test_data = compute_frozen_embeddings(test_data, frozen_embedder)
        print(f"Precomputed embeddings: {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test")
    else:
        print("Frozen embeddings disabled")

    return train_data, dev_data, test_data, frozen_dim


def tokenize_cross_encoder(batch, tokenizer, max_length, use_frozen_embeddings):
    """Tokenize as [CLS] text_a [SEP] text_b [SEP]."""
    encoded = tokenizer(
        batch["text_a"],
        batch["text_b"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    out = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": batch["label"],
        "stdev": batch.get("stdev", [0.0] * len(batch["label"])),
    }
    if "token_type_ids" in encoded:
        out["token_type_ids"] = encoded["token_type_ids"]
    if use_frozen_embeddings and "frozen_a" in batch:
        out["frozen_a"] = batch["frozen_a"]
        out["frozen_b"] = batch["frozen_b"]
    return out
