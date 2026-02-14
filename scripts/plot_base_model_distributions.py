#!/usr/bin/env python3
"""Plot smooth score distributions for base models on one split (dev/test)."""

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def numeric_sort_keys(keys: Sequence[str]) -> List[str]:
    try:
        return sorted(keys, key=lambda k: int(k))
    except Exception:
        return sorted(keys)


def load_prediction_file(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        keys = numeric_sort_keys(list(payload.keys()))
        return np.array([float(payload[k]) for k in keys], dtype=np.float32)

    if isinstance(payload, list):
        out: List[float] = []
        for item in payload:
            if isinstance(item, dict):
                if "prediction" not in item:
                    raise ValueError(f"List item missing 'prediction' in {path}")
                out.append(float(item["prediction"]))
            else:
                out.append(float(item))
        return np.array(out, dtype=np.float32)

    raise ValueError(f"Unsupported prediction JSON format in {path}")


def kde_density(values: np.ndarray, x_grid: np.ndarray, bandwidth: float) -> np.ndarray:
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be > 0.")
    if values.size == 0:
        return np.zeros_like(x_grid, dtype=np.float64)

    v = values.astype(np.float64).reshape(1, -1)
    x = x_grid.astype(np.float64).reshape(-1, 1)
    z = (x - v) / bandwidth
    kernel = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return np.mean(kernel, axis=1) / bandwidth


def describe(arr: np.ndarray) -> str:
    return f"n={arr.shape[0]} mean={float(np.mean(arr)):.3f} sd={float(np.std(arr)):.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot smooth score distributions for multiple base models on one split."
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="predictions",
        help="Directory containing <split>_preds_<model>.json files.",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="test",
        help="Which split predictions to plot.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ares", "lmms", "nli", "sbert", "stsbert"],
        help="Model names (suffix after <split>_preds_).",
    )
    parser.add_argument("--x-min", type=float, default=1.0)
    parser.add_argument("--x-max", type=float, default=5.0)
    parser.add_argument(
        "--kde-bandwidth",
        type=float,
        default=0.16,
        help="Kernel bandwidth for smoothing (larger values => smoother curves).",
    )
    parser.add_argument("--kde-points", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path (PNG). Defaults to predictions/plots/base_models_<split>_smooth_distribution.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        raise SystemExit(f"Prediction directory not found: {pred_dir}")

    x_min = float(args.x_min)
    x_max = float(args.x_max)
    if x_max <= x_min:
        raise SystemExit("--x-max must be greater than --x-min")

    bandwidth = float(args.kde_bandwidth)
    if bandwidth <= 0:
        raise SystemExit("--kde-bandwidth must be > 0")

    n_grid = int(args.kde_points)
    if n_grid < 100:
        raise SystemExit("--kde-points must be >= 100")
    x_grid = np.linspace(x_min, x_max, n_grid, dtype=np.float64)

    model_scores = {}
    for name in args.models:
        path = pred_dir / f"{args.split}_preds_{name}.json"
        if not path.exists():
            raise SystemExit(f"Prediction file not found for model '{name}': {path}")
        model_scores[name] = load_prediction_file(path)

    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#FF9DA6"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for idx, (name, values) in enumerate(model_scores.items()):
        color = colors[idx % len(colors)]
        clipped = np.clip(values, x_min, x_max)
        density = kde_density(clipped, x_grid, bandwidth=bandwidth)
        ax.plot(x_grid, density, color=color, linewidth=2.0, label=f"{name} ({describe(values)})")
        ax.fill_between(x_grid, density, color=color, alpha=0.08)
        ax.axvline(float(np.mean(values)), color=color, linestyle="--", linewidth=1.0, alpha=0.85)

    ax.set_title(f"Smoothed Distribution of {args.split.title()} Predictions (Base Models)")
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Density")
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    out_path = (
        Path(args.out)
        if args.out
        else pred_dir / "plots" / f"base_models_{args.split}_smooth_distribution.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved plot to {out_path}")
    print(f"Split: {args.split}")
    print(f"Models: {', '.join(args.models)}")


if __name__ == "__main__":
    main()
