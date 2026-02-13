from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence, Any
import re


def cartesian(product_dict: Dict[str, Sequence]) -> List[Dict[str, any]]:
    keys = list(product_dict.keys())
    combos = []
    for values in itertools.product(*[product_dict[k] for k in keys]):
        combos.append(dict(zip(keys, values)))
    return combos


def build_space(expert: str) -> List[Dict[str, Any]]:
    """Return a tightened hyperparameter grid around the current best configs."""
    if expert == "sbert":
        space = {
            "lr": [5e-5, 1e-4, 1.5e-4],
            "weight_decay": [0.03, 0.05],
            "dropout": [0.1, 0.2, 0.3],
            "hidden_dim": [256],
            "projector_dim": [256],
            "warmup_ratio": [0.06],
            "train_encoder_layers": [0],
            "mse_weight": [0.0],
            "soft_weight": [0.0],
            "expand_annotators": [False, True],
        }
    else:  # nli
        space = {
            "lr": [5e-5, 7e-5, 1e-4],
            "weight_decay": [0.02, 0.03, 0.05],
            "dropout": [0.1, 0.2],
            "hidden_dim": [256],
            "projector_dim": [256],
            "warmup_ratio": [0.06],
            "train_encoder_layers": [0],
            "mse_weight": [0.0],
            "soft_weight": [0.0],
            "pooling": ["mean"],
            "expand_annotators": [False, True],
        }
    return cartesian(space)


def sample_space(space: List[Dict[str, any]], max_trials: int, seed: int) -> List[Dict[str, any]]:
    if len(space) <= max_trials:
        return space
    random.seed(seed)
    return random.sample(space, max_trials)


METRIC_RE = re.compile(
    r"composite=(?P<composite>[-0-9.]+).*mae=(?P<mae>[-0-9.]+).*spearman=(?P<spearman>[-0-9.]+).*acc_sd=(?P<acc_sd>[-0-9.]+)"
)


def run_trial(idx: int, expert: str, cfg: Dict[str, any], base_args: argparse.Namespace, out_dir: Path) -> Dict[str, any]:
    save_path = out_dir / f"{expert}_trial{idx}.pt"
    log_path = out_dir / f"{expert}_trial{idx}.log"
    plot_path = out_dir / f"{expert}_trial{idx}.png"

    cmd = [
        base_args.python,
        "-m",
        "models.train_experts",
        "--expert",
        expert,
        "--model-name",
        base_args.model_name if expert == "sbert" else base_args.nli_model_name,
        "--train-encoder-layers",
        str(cfg.get("train_encoder_layers", 0)),
        "--lr",
        str(cfg["lr"]),
        "--encoder-lr",
        "0" if cfg.get("train_encoder_layers", 0) == 0 else str(base_args.encoder_lr),
        "--weight-decay",
        str(cfg["weight_decay"]),
        "--dropout",
        str(cfg["dropout"]),
        "--hidden-dim",
        str(cfg["hidden_dim"]),
        "--projector-dim",
        str(cfg["projector_dim"]),
        "--warmup-ratio",
        str(cfg["warmup_ratio"]),
        "--early-stop-patience",
        str(base_args.early_stop_patience),
        "--use-ema",
        "--coral-weight",
        "1.0",
        "--mse-weight",
        str(cfg["mse_weight"]),
        "--soft-weight",
        str(cfg["soft_weight"]),
        "--batch-size",
        str(base_args.batch_size_nli if expert == "nli" else base_args.batch_size_sbert),
        "--epochs",
        str(base_args.epochs),
        "--save-path",
        str(save_path),
        "--plot-path",
        str(plot_path),
        "--no-plot",
    ]
    if cfg.get("expand_annotators", False):
        cmd += ["--expand-annotators"]
    if expert == "nli":
        cmd += ["--pooling", cfg["pooling"]]
    env = os.environ.copy()
    if base_args.pythonpath:
        env["PYTHONPATH"] = base_args.pythonpath
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    duration = time.time() - start
    log_path.write_text(result.stdout + "\nSTDERR:\n" + result.stderr)
    dev_line = next((ln for ln in result.stdout.splitlines() if "dev epoch" in ln), "")
    m = METRIC_RE.search(dev_line)
    metrics = m.groupdict() if m else {}
    status = "ok" if result.returncode == 0 else f"err_{result.returncode}"
    return {
        "trial": idx,
        "expert": expert,
        "cfg": json.dumps(cfg),
        "dev_line": dev_line,
        "composite": metrics.get("composite", ""),
        "mae": metrics.get("mae", ""),
        "spearman": metrics.get("spearman", ""),
        "acc_sd": metrics.get("acc_sd", ""),
        "status": status,
        "seconds": round(duration, 1),
        "log": str(log_path),
        "ckpt": str(save_path) if save_path.exists() else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for NLI/SBERT experts.")
    parser.add_argument("--experts", nargs="+", default=["sbert", "nli"], choices=["sbert", "nli"])
    parser.add_argument("--max-trials", type=int, default=24, help="Max trials per expert (random subset of grid).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/sweep"))
    parser.add_argument("--python", type=str, default=".venv/bin/python3")
    parser.add_argument("--pythonpath", type=str, default="src")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--nli-model-name", type=str, default="roberta-large-mnli")
    parser.add_argument("--batch-size-sbert", type=int, default=8)
    parser.add_argument("--batch-size-nli", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--encoder-lr", type=float, default=5e-6)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, any]] = []

    for expert in args.experts:
        space = build_space(expert)
        trials = sample_space(space, args.max_trials, args.seed)
        for i, cfg in enumerate(trials):
            rec = run_trial(i, expert, cfg, args, args.out_dir)
            rows.append(rec)
            print(f"[{rec['status']}] {expert} trial {i}: {rec['dev_line']}")

    csv_path = args.out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial",
                "expert",
                "cfg",
                "composite",
                "mae",
                "spearman",
                "acc_sd",
                "dev_line",
                "status",
                "seconds",
                "log",
                "ckpt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done. Results written to {csv_path}")


if __name__ == "__main__":
    main()
