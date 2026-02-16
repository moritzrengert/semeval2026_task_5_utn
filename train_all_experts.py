#!/usr/bin/env python3
"""Train all experts and write dev/test prediction JSON files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ares_expert.train_mlp_final import HYPERPARAMS as ARES_HP, run_epoch as ares_run_epoch
from coral_head import CoralHead
from data_utils import load_dataset
from expert_dataset import SemevalExpertDataset
from lmms_expert.train_mlp_final import HYPERPARAMS as LMMS_HP, run_epoch as lmms_run_epoch
from losses import coral_expected_value
from nli_expert import NliExpertConfig
from sbert_expert import SbertExpertConfig
from sensembert_expert.train_mlp_final import HYPERPARAMS as SENSEMBERT_HP, run_epoch as sensembert_run_epoch
import train_experts


TRAIN_PATH = ROOT / "semeval26-05-scripts" / "data" / "train.json"
VAL_PATH = ROOT / "semeval26-05-scripts" / "data" / "dev.json"
TEST_PATH = ROOT / "semeval26-05-scripts" / "data" / "test.json"
LABEL_KEY = "average"
TARGET_AWARE = False

FEATURE_TRAIN_SPLIT = "train"
FEATURE_VAL_SPLIT = "dev"
FEATURE_TEST_SPLIT = "test"

PRED_DIR = ROOT / "predictions"
RUNS_DIR = ROOT / "runs" / "all_experts"

FEATURE_EXPERTS = [
    ("ares", ARES_HP, ares_run_epoch, SRC / "ares_expert" / "ares_context_features.pt"),
    ("lmms", LMMS_HP, lmms_run_epoch, SRC / "lmms_expert" / "lmms_context_features.pt"),
    ("sensembert", SENSEMBERT_HP, sensembert_run_epoch, SRC / "sensembert_expert" / "sensembert_context_features.pt"),
]


def write_preds(path: Path, preds: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"prediction": float(x)} for x in preds], indent=2) + "\n", encoding="utf-8")


def predict_hf(model, collate_fn, records, num_classes, batch_size, device):
    ds = SemevalExpertDataset(records, num_classes=num_classes, label_key=LABEL_KEY, target_aware=TARGET_AWARE)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    out = []
    model.eval().to(device)
    with torch.no_grad():
        for batch in loader:
            batch = train_experts.move_to_device(batch, device)
            logits = train_experts.forward_batch(model, batch)["ordinal_logits"]
            out.extend((coral_expected_value(logits) + 1.0).detach().cpu().tolist())
    return out


def train_hf(expert_name: str, cfg_cls, val_records, test_records) -> None:
    ckpt = RUNS_DIR / f"{expert_name}_best.pt"
    ns = argparse.Namespace(
        expert=expert_name,
        train_path=str(TRAIN_PATH),
        dev_path=str(VAL_PATH),
        label_key=LABEL_KEY,
        save_path=str(ckpt),
        plot_path=str(RUNS_DIR / f"{expert_name}.png"),
        no_plot=True,
        target_aware=TARGET_AWARE,
        device=None,
    )
    cfg = cfg_cls()
    ns.expert_config = cfg
    for k, v in vars(cfg).items():
        setattr(ns, k, v)

    print(f"[{expert_name}] training...")
    train_experts.train(ns)

    payload = torch.load(ckpt, map_location="cpu")
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model, collate_fn, _ = train_experts.build_expert(ns)
    model.load_state_dict(state, strict=False)
    device = train_experts.resolve_device(None)

    write_preds(PRED_DIR / f"dev_preds_{expert_name}.json", predict_hf(model, collate_fn, val_records, ns.num_classes, ns.batch_size, device))
    write_preds(PRED_DIR / f"test_preds_{expert_name}.json", predict_hf(model, collate_fn, test_records, ns.num_classes, ns.batch_size, device))
    print(f"[{expert_name}] done")


def predict_feature(model, x, device):
    model.eval().to(device)
    with torch.no_grad():
        logits = model(x.to(device))
        return (coral_expected_value(logits) + 1.0).detach().cpu().tolist()


def train_feature(name, hp, run_epoch, feature_file, device):
    if not feature_file.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_file}")
    data = torch.load(feature_file, map_location="cpu")
    train_x = data[f"{FEATURE_TRAIN_SPLIT}_features"].float()
    train_y = data[f"{FEATURE_TRAIN_SPLIT}_labels"].float()
    val_x = data[f"{FEATURE_VAL_SPLIT}_features"].float()
    val_y = data[f"{FEATURE_VAL_SPLIT}_labels"].float()
    test_x = data[f"{FEATURE_TEST_SPLIT}_features"].float()

    train_y = torch.clamp(torch.round(train_y), 1.0, float(hp["num_classes"]))
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0)
    std[std == 0] = 1.0
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std

    bs = int(hp["batch_size"])
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=bs, shuffle=False)

    model = CoralHead(
        input_dim=int(train_x.shape[1]),
        num_classes=int(hp["num_classes"]),
        hidden_dim=int(hp["hidden_dim"]),
        dropout=float(hp["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))

    print(f"[{name}] training...")
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(int(hp["epochs"])):
        train_loss = run_epoch(model, train_loader, optimizer, device, is_training=True)
        val_loss = run_epoch(model, val_loader, None, device, is_training=False)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = "*"
        else:
            mark = ""
        print(f"[{name}] epoch {epoch + 1:03d} train={train_loss:.4f} val={val_loss:.4f} {mark}")

    model.load_state_dict(best_state, strict=False)
    torch.save({"state_dict": best_state, "expert": name}, RUNS_DIR / f"{name}_best.pt")
    write_preds(PRED_DIR / f"dev_preds_{name}.json", predict_feature(model, val_x, device))
    write_preds(PRED_DIR / f"test_preds_{name}.json", predict_feature(model, test_x, device))
    print(f"[{name}] done")


def train_stsbert():
    out_dir = RUNS_DIR / "stsbert_output"
    print("[stsbert] training...")
    subprocess.run(
        [
            sys.executable,
            str(SRC / "cross_encoder_expert" / "train.py"),
            "--train-path",
            str(TRAIN_PATH),
            "--dev-path",
            str(VAL_PATH),
            "--test-path",
            str(TEST_PATH),
            "--save-path",
            str(RUNS_DIR / "stsbert_best.pt"),
            "--output-dir",
            str(out_dir),
            "--label-key",
            LABEL_KEY,
        ],
        check=True,
        cwd=str(ROOT),
    )
    print("[stsbert] predicting...")
    subprocess.run(
        [
            sys.executable,
            str(SRC / "cross_encoder_expert" / "predict.py"),
            "--checkpoint",
            str(out_dir),
            "--dev-path",
            str(VAL_PATH),
            "--test-path",
            str(TEST_PATH),
            "--out-dev",
            str(PRED_DIR / "dev_preds_stsbert.json"),
            "--out-test",
            str(PRED_DIR / "test_preds_stsbert.json"),
            "--label-key",
            LABEL_KEY,
        ],
        check=True,
        cwd=str(ROOT),
    )
    print("[stsbert] done")


def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    device = train_experts.resolve_device(None)
    val_records = load_dataset(VAL_PATH)
    test_records = load_dataset(TEST_PATH)

    for spec in FEATURE_EXPERTS:
        train_feature(*spec, device)
    train_hf("nli", NliExpertConfig, val_records, test_records)
    train_hf("sbert", SbertExpertConfig, val_records, test_records)
    train_stsbert()
    print("Done.")


if __name__ == "__main__":
    main()
