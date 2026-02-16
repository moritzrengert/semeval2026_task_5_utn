"""
Write dev and test prediction JSONs in the same format as other experts (for ensemble).
Order matches load_dataset(train_path / dev_path / test_path).
Run from repo root: python src/cross_encoder_expert/predict.py --checkpoint ... --dev-path ... --test-path ... --out-dev ... --out-test ...
Created by Adam Jen Khai Lo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cross_encoder_expert import config as default_config
from cross_encoder_expert.data_utils import load_and_process_data, tokenize_cross_encoder
from cross_encoder_expert.model import CrossEncoderRegressor
from cross_encoder_expert.collator import CrossEncoderCollator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to saved model dir (or checkpoint dir)")
    p.add_argument("--dev-path", type=str, required=True)
    p.add_argument("--test-path", type=str, required=True)
    p.add_argument("--out-dev", type=str, default="predictions/dev_preds_cross_encoder.json")
    p.add_argument("--out-test", type=str, default="predictions/test_preds_cross_encoder.json")
    p.add_argument("--label-key", type=str, default=default_config.LABEL_KEY)
    args = p.parse_args()

    train_data, dev_data, test_data, frozen_dim = load_and_process_data(
        args.dev_path,  # train not needed for predict
        args.dev_path,
        args.test_path,
        use_frozen_embeddings=default_config.USE_FROZEN_EMBEDDINGS,
        frozen_model_name=default_config.FROZEN_MODEL_NAME,
        label_key=args.label_key,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    def tokenize_fn(batch):
        return tokenize_cross_encoder(
            batch, tokenizer, default_config.MAX_LENGTH, default_config.USE_FROZEN_EMBEDDINGS
        )

    dev_ds = Dataset.from_list(dev_data).map(tokenize_fn, batched=True)
    test_ds = Dataset.from_list(test_data).map(tokenize_fn, batched=True)
    keep_cols = ["input_ids", "attention_mask", "labels", "stdev"]
    if "token_type_ids" in dev_ds.column_names:
        keep_cols.append("token_type_ids")
    if default_config.USE_FROZEN_EMBEDDINGS:
        keep_cols.extend(["frozen_a", "frozen_b"])
    dev_ds.remove_columns([c for c in dev_ds.column_names if c not in keep_cols])
    test_ds.remove_columns([c for c in test_ds.column_names if c not in keep_cols])

    model = CrossEncoderRegressor(
        default_config.MODEL_NAME,
        frozen_dim=frozen_dim if default_config.USE_FROZEN_EMBEDDINGS else 0,
        use_dual_path=default_config.USE_DUAL_PATH,
        pooling=default_config.POOLING,
    )
    state_path = Path(args.checkpoint) / "cross_encoder_state.pt"
    if not state_path.exists():
        state_path = Path(args.checkpoint) / "pytorch_model.bin"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError(f"No weights found at {args.checkpoint}. Train first and pass --checkpoint to the output dir.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    collator = CrossEncoderCollator(tokenizer)

    def run_predict(dataset):
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=default_config.BATCH_SIZE, collate_fn=collator)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(**{k: v for k, v in batch.items() if k not in ("labels", "stdev")})
                preds.append(out["logits"].cpu().numpy())
        preds = np.concatenate(preds, axis=0).squeeze()
        preds = np.clip(preds, default_config.LABEL_MIN, default_config.LABEL_MAX)
        return preds

    dev_preds = run_predict(dev_ds)
    test_preds = run_predict(test_ds)

    # Format expected by ensemble: list of {"prediction": float} (same order as dev/test)
    Path(args.out_dev).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_test).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_dev, "w") as f:
        json.dump([{"prediction": float(p)} for p in dev_preds.tolist()], f, indent=2)
    with open(args.out_test, "w") as f:
        json.dump([{"prediction": float(p)} for p in test_preds.tolist()], f, indent=2)
    print(f"Wrote {args.out_dev} ({len(dev_preds)} dev) and {args.out_test} ({len(test_preds)} test). Use these with --dev-preds / --test-preds in ensemble.")


if __name__ == "__main__":
    main()
