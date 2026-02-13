#!/usr/bin/env python3
"""
Ensemble two pretrained experts (NLI or SBERT) with configurable combining rule.

Methods:
- mean: simple average of the two predicted scores
- weighted: user-provided weights w1, w2 (normalized to sum=1)
- learned: fit weights (and bias) on the dev set via least squares, then apply to test

Outputs: prints dev metrics for individual models and the ensemble (MSE, MAE, Spearman, acc_within_sd),
and saves optional prediction files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from models.nli_expert import NliExpertConfig, NliPlausibilityExpert
from models.sbert_expert import SbertExpertConfig, SbertSemanticMatchingExpert
from models.collate import make_nli_collate_fn, make_sbert_collate_fn
from models.expert_dataset import SemevalExpertDataset
from final_model.data import load_dataset, expand_annotator_samples
from final_model.losses import coral_expected_value


def load_records(path: str, expand: bool, label_key: str, num_classes: int) -> list[Dict[str, Any]]:
    records = load_dataset(path)
    if expand:
        records = expand_annotator_samples(records, label_key=label_key, choices_key="choices")
    return records


DEFAULT_MODEL = {"sbert": "sentence-transformers/all-mpnet-base-v2", "nli": "roberta-large-mnli"}


def build_model(expert: str, model_name: str, args: argparse.Namespace, num_classes: int):
    if expert == "sbert":
        cfg = SbertExpertConfig(
            model_name=model_name,
            num_classes=num_classes,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            max_length=args.max_length,
            projector_dim=args.projector_dim,
            use_classifier=False,
        )
        model = SbertSemanticMatchingExpert(cfg)
        collate = make_sbert_collate_fn(model.tokenizer, max_length=cfg.max_length, num_classes=num_classes)
        return model, collate
    elif expert == "nli":
        cfg = NliExpertConfig(
            model_name=model_name,
            num_classes=num_classes,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            max_length=args.max_length,
            pooling=args.pooling,
            projector_dim=args.projector_dim,
            use_classifier=False,
        )
        model = NliPlausibilityExpert(cfg)
        collate = make_nli_collate_fn(model.tokenizer, max_length=cfg.max_length, num_classes=num_classes)
        return model, collate
    else:
        raise ValueError(f"Unsupported expert {expert}")


@torch.no_grad()
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    preds = []
    labels = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if "context_ids" in batch:
            outputs = model(batch["context_ids"], batch["context_mask"], batch["gloss_ids"], batch["gloss_mask"])
        else:
            outputs = model(batch["input_ids"], batch["attention_mask"])
        score = (coral_expected_value(outputs["ordinal_logits"]) + 1.0).cpu().numpy()
        label = batch["scores"].cpu().numpy()
        preds.append(score)
        labels.append(label)
    return np.concatenate(preds), np.concatenate(labels)


def metrics(preds: np.ndarray, labels: np.ndarray, choices: np.ndarray | None = None, choices_mask: np.ndarray | None = None) -> Dict[str, float]:
    mse = float(np.mean((preds - labels) ** 2)) if len(labels) else 0.0
    mae = float(np.mean(np.abs(preds - labels))) if len(labels) else 0.0
    spear = float(spearmanr(preds, labels).correlation) if len(labels) > 1 else 0.0
    acc_sd = 0.0
    if choices is not None and choices_mask is not None and len(labels):
        total = 0
        correct = 0
        for p, c, m in zip(preds, choices, choices_mask):
            if m.any():
                vals = c[m]
                mean = vals.mean()
                sd = vals.std() if vals.size > 1 else 0.0
                within = (mean - sd) < p < (mean + sd) or abs(mean - p) < 1.0
                correct += 1 if within else 0
                total += 1
        acc_sd = correct / total if total else 0.0
    return {"mse": mse, "mae": mae, "spearman": spear, "acc_sd": acc_sd}


def fit_weights(dev_preds: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    # Fit y ~ w1*p1 + w2*p2 + b via least squares
    X = np.column_stack([dev_preds[:, 0], dev_preds[:, 1], np.ones(len(labels))])
    w, *_ = np.linalg.lstsq(X, labels, rcond=None)
    return w[0], w[1], w[2]


def main():
    ap = argparse.ArgumentParser(description="Ensemble two experts with mean/weighted/learned combiner.")
    ap.add_argument("--expert1", choices=["sbert", "nli"], default=None, help="If omitted, inferred from ckpt1['expert'].")
    ap.add_argument("--expert2", choices=["sbert", "nli"], default=None, help="If omitted, inferred from ckpt2['expert'].")
    ap.add_argument("--model-name1", type=str, default=None, help="If omitted, uses a sensible default for expert1.")
    ap.add_argument("--model-name2", type=str, default=None, help="If omitted, uses a sensible default for expert2.")
    ap.add_argument("--train-path", type=str, default="semeval26-05-scripts/data/train.json")
    ap.add_argument("--dev-path", type=str, default="semeval26-05-scripts/data/dev.json")
    ap.add_argument("--test-path", type=str, default=None)
    ap.add_argument("--ckpt1", type=str, required=True, help="Checkpoint file for first expert (must contain state_dict).")
    ap.add_argument("--ckpt2", type=str, required=True, help="Checkpoint file for second expert (must contain state_dict).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-classes", type=int, default=5)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--projector-dim", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--pooling", choices=["cls", "mean"], default="mean")
    ap.add_argument("--expand-annotators", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--combine", choices=["mean", "weighted", "learned"], default="mean")
    ap.add_argument("--weights", type=str, default="0.5,0.5", help="Comma weights for weighted combine")
    ap.add_argument("--save-preds", type=str, default=None, help="Optional path to save ensemble predictions (jsonl)")
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Load datasets
    dev_records = load_records(args.dev_path, args.expand_annotators, label_key="average", num_classes=args.num_classes)
    dev_ds = SemevalExpertDataset(dev_records, num_classes=args.num_classes, label_key="average")

    # Load checkpoints and infer missing metadata
    state1 = torch.load(args.ckpt1, map_location="cpu")
    state2 = torch.load(args.ckpt2, map_location="cpu")
    expert1 = args.expert1 or state1.get("expert")
    expert2 = args.expert2 or state2.get("expert")
    if expert1 not in {"sbert", "nli"} or expert2 not in {"sbert", "nli"}:
        raise ValueError("Could not infer expert types; please pass --expert1/--expert2.")
    model_name1 = args.model_name1 or DEFAULT_MODEL[expert1]
    model_name2 = args.model_name2 or DEFAULT_MODEL[expert2]

    # Build models and loaders
    model1, collate1 = build_model(expert1, model_name1, args, args.num_classes)
    model2, collate2 = build_model(expert2, model_name2, args, args.num_classes)
    model1.load_state_dict(state1["state_dict"], strict=False)
    model2.load_state_dict(state2["state_dict"], strict=False)
    dev_loader1 = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate1)
    dev_loader2 = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate2)

    # Predict dev
    p1, y = predict(model1, dev_loader1, device)
    p2, _ = predict(model2, dev_loader2, device)
    dev_preds = np.stack([p1, p2], axis=1)

    # Choose combiner
    if args.combine == "mean":
        w1 = w2 = 0.5; b = 0.0
    elif args.combine == "weighted":
        wlist = [float(x) for x in args.weights.split(",")]
        if len(wlist) != 2:
            raise ValueError("--weights must have two comma-separated values")
        s = wlist[0] + wlist[1]
        w1, w2 = wlist[0] / s, wlist[1] / s
        b = 0.0
    else:  # learned
        w1, w2, b = fit_weights(dev_preds, y)

    ens_dev = dev_preds[:, 0] * w1 + dev_preds[:, 1] * w2 + b
    # Choices for acc_sd (pad variable-length annotator ratings)
    choices_raw = []
    for r in dev_records:
        c = np.array(r.get("choices", []), dtype=float).flatten()
        if c.size == 0:
            c = np.array([np.nan])
        choices_raw.append(c)
    max_len = max(len(c) for c in choices_raw)
    choices = np.full((len(choices_raw), max_len), np.nan, dtype=float)
    for i, c in enumerate(choices_raw):
        choices[i, : len(c)] = c
    choices_mask = ~np.isnan(choices)
    dev_metrics = metrics(ens_dev, y, choices, choices_mask)

    print(f"Ensemble weights: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}")
    print(
        f"Dev ensemble: mse={dev_metrics['mse']:.4f}, mae={dev_metrics['mae']:.4f}, spearman={dev_metrics['spearman']:.4f}, acc_sd={dev_metrics['acc_sd']:.4f}"
    )

    if args.test_path:
        test_records = load_records(args.test_path, args.expand_annotators, label_key="average", num_classes=args.num_classes)
        # Ensure numeric placeholder labels so the dataset/loader can collate, even if test golds are unknown.
        for rec in test_records:
            try:
                float(rec.get("average", 0.0))
            except (TypeError, ValueError):
                rec["average"] = 0.0
        test_ds = SemevalExpertDataset(test_records, num_classes=args.num_classes, label_key="average", require_labels=True, drop_invalid_labels=True)
        test_loader1 = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate1)
        test_loader2 = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate2)
        t1, _ = predict(model1, test_loader1, device)
        t2, _ = predict(model2, test_loader2, device)
        test_preds = t1 * w1 + t2 * w2 + b
        if args.save_preds:
            out_path = Path(args.save_preds)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                for idx, rec in enumerate(test_records):
                    rid = rec.get("sample_id", idx)
                    f.write(json.dumps({"sample_id": rid, "prediction": float(test_preds[idx])}) + "\n")
            print(f"Saved test predictions to {out_path}")


if __name__ == "__main__":
    main()
