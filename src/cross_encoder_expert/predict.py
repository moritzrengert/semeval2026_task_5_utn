"""
Write dev/test prediction JSONs for the cross-encoder expert.

Run from repo root:
python src/cross_encoder_expert/predict.py --checkpoint ... --dev-path ... --test-path ...
Created by Adam Jen Khai Lo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cross_encoder_expert.ce_data_utils import load_dataset, process_examples, tokenize_cross_encoder
from cross_encoder_expert.collator import CrossEncoderCollator
from cross_encoder_expert.model import CrossEncoderRegressor

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 256
LABEL_KEY = "average"
LABEL_MIN = 1.0
LABEL_MAX = 5.0
DEFAULT_BATCH_SIZE = 16


def load_state_dict(checkpoint: Path) -> tuple[dict, Path]:
    candidates = []
    if checkpoint.is_file():
        candidates.append(checkpoint)
    else:
        candidates.extend(
            [
                checkpoint / "cross_encoder_state.pt",
                checkpoint / "cross_encoder_only_state.pt",
                checkpoint / "pytorch_model.bin",
            ]
        )

    for path in candidates:
        if path.exists():
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                return state, path
    raise FileNotFoundError(
        f"No model state found at {checkpoint}. "
        "Expected one of: cross_encoder_state.pt, cross_encoder_only_state.pt, pytorch_model.bin"
    )


def write_predictions(path: Path, preds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"prediction": float(x)} for x in preds.tolist()]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_dataset(path: str, tokenizer, label_key: str) -> Dataset:
    records = process_examples(load_dataset(path), label_key=label_key)

    def tokenize_fn(batch):
        return tokenize_cross_encoder(batch, tokenizer, MAX_LENGTH)

    ds = Dataset.from_list(records).map(tokenize_fn, batched=True)
    keep_cols = ["input_ids", "attention_mask", "labels", "stdev"]
    if "token_type_ids" in ds.column_names:
        keep_cols.append("token_type_ids")
    return ds.remove_columns([c for c in ds.column_names if c not in keep_cols])


def run_predict(
    model: CrossEncoderRegressor,
    ds: Dataset,
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=CrossEncoderCollator(tokenizer))
    preds = []
    model.eval().to(device)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**{k: v for k, v in batch.items() if k not in ("labels", "stdev")})
            preds.append(outputs["logits"].detach().cpu().numpy())
    if not preds:
        return np.array([], dtype=np.float32)
    stacked = np.concatenate(preds, axis=0).astype(np.float32).reshape(-1)
    return np.clip(stacked, LABEL_MIN, LABEL_MAX)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint dir or state file path.")
    parser.add_argument("--dev-path", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--out-dev", type=str, default="predictions/dev_preds_stsbert.json")
    parser.add_argument("--out-test", type=str, default="predictions/test_preds_stsbert.json")
    parser.add_argument("--label-key", type=str, default=LABEL_KEY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    tokenizer_source = str(checkpoint) if checkpoint.is_dir() else MODEL_NAME
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = CrossEncoderRegressor(MODEL_NAME)
    state_dict, loaded_path = load_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)

    dev_ds = build_dataset(args.dev_path, tokenizer, args.label_key)
    test_ds = build_dataset(args.test_path, tokenizer, args.label_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dev_preds = run_predict(model, dev_ds, tokenizer, device, args.batch_size)
    test_preds = run_predict(model, test_ds, tokenizer, device, args.batch_size)

    out_dev = Path(args.out_dev)
    out_test = Path(args.out_test)
    write_predictions(out_dev, dev_preds)
    write_predictions(out_test, test_preds)
    print(f"Loaded weights from {loaded_path}")
    print(f"Wrote {out_dev} ({len(dev_preds)} rows)")
    print(f"Wrote {out_test} ({len(test_preds)} rows)")


if __name__ == "__main__":
    main()
