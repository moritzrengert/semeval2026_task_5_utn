#!/usr/bin/env python3
"""Plot test-set score distribution vs. average/weighted/MLP prediction distributions."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def numeric_sort_keys(keys: Sequence[str]) -> List[str]:
    try:
        return sorted(keys, key=lambda k: int(k))
    except Exception:
        return sorted(keys)


def load_ordered_records(path: Path) -> List[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        keys = numeric_sort_keys(list(payload.keys()))
        return [payload[k] for k in keys]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported JSON format in {path}")


def to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def load_labels(path: Path, label_key: str = "average") -> np.ndarray:
    records = load_ordered_records(path)
    values = []
    for rec in records:
        if isinstance(rec, dict):
            values.append(to_float_or_nan(rec.get(label_key)))
        else:
            values.append(float("nan"))
    return np.array(values, dtype=np.float32)


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


def resolve_from_summary(summary_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    if not summary_path.exists():
        return None, None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    weighted = payload.get("best_weighted", {})
    mlp = payload.get("best_mlp", {})
    weighted_path = weighted.get("out_test")
    mlp_path = mlp.get("out_test")
    return (
        Path(weighted_path) if isinstance(weighted_path, str) and weighted_path else None,
        Path(mlp_path) if isinstance(mlp_path, str) and mlp_path else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare test gold score distribution against average/weighted/MLP prediction distributions."
    )
    parser.add_argument(
        "--test-labels",
        type=str,
        default="semeval26-05-scripts/data/test.json",
        help="Path to test split JSON with numeric 'average' labels.",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="average",
        help="Label key in --test-labels to use as gold score.",
    )
    parser.add_argument(
        "--average-preds",
        type=str,
        default="predictions/test_preds_ensemble_average.json",
        help="Path to average ensemble test predictions.",
    )
    parser.add_argument(
        "--weighted-preds",
        type=str,
        default=None,
        help="Path to weighted ensemble test predictions. If omitted, taken from --summary.",
    )
    parser.add_argument(
        "--mlp-preds",
        type=str,
        default=None,
        help="Path to MLP ensemble test predictions. If omitted, taken from --summary.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="predictions/ensemble_sweep_runs/summary.json",
        help="Sweep summary used to infer best weighted/MLP paths.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="predictions/plots/test_distribution_compare_avg_weighted_mlp_vs_gold.png",
        help="Output PNG path.",
    )
    parser.add_argument("--x-min", type=float, default=1.0)
    parser.add_argument("--x-max", type=float, default=5.0)
    parser.add_argument(
        "--kde-bandwidth",
        type=float,
        default=0.16,
        help="Kernel bandwidth for smoothing (larger values => smoother curves).",
    )
    parser.add_argument(
        "--kde-points",
        type=int,
        default=500,
        help="Number of x-grid points used to draw smooth density curves.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def check_equal_length(name: str, arr: np.ndarray, expected: int) -> None:
    if arr.shape[0] != expected:
        raise ValueError(f"{name} length mismatch: expected {expected}, got {arr.shape[0]}")


def describe(arr: np.ndarray) -> str:
    return f"n={arr.shape[0]} mean={float(np.mean(arr)):.3f} sd={float(np.std(arr)):.3f}"


def kde_density(values: np.ndarray, x_grid: np.ndarray, bandwidth: float) -> np.ndarray:
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be > 0.")
    if values.size == 0:
        return np.zeros_like(x_grid, dtype=np.float64)

    v = values.astype(np.float64).reshape(1, -1)
    x = x_grid.astype(np.float64).reshape(-1, 1)
    z = (x - v) / bandwidth
    kernel = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    density = np.mean(kernel, axis=1) / bandwidth
    return density


def main() -> None:
    args = parse_args()

    summary_weighted, summary_mlp = resolve_from_summary(Path(args.summary))
    weighted_path = Path(args.weighted_preds) if args.weighted_preds else summary_weighted
    mlp_path = Path(args.mlp_preds) if args.mlp_preds else summary_mlp

    if weighted_path is None:
        raise SystemExit("Could not resolve weighted predictions. Pass --weighted-preds or provide --summary.")
    if mlp_path is None:
        raise SystemExit("Could not resolve MLP predictions. Pass --mlp-preds or provide --summary.")

    labels = load_labels(Path(args.test_labels), label_key=args.label_key)
    mask = np.isfinite(labels)
    gold = labels[mask]
    if gold.shape[0] == 0:
        raise SystemExit(f"No numeric labels found in {args.test_labels} for key '{args.label_key}'.")

    pred_avg = load_prediction_file(Path(args.average_preds))
    pred_weighted = load_prediction_file(weighted_path)
    pred_mlp = load_prediction_file(mlp_path)

    check_equal_length("average predictions", pred_avg, labels.shape[0])
    check_equal_length("weighted predictions", pred_weighted, labels.shape[0])
    check_equal_length("mlp predictions", pred_mlp, labels.shape[0])

    avg = pred_avg[mask]
    weighted = pred_weighted[mask]
    mlp = pred_mlp[mask]

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

    fig, ax = plt.subplots(figsize=(10, 6))
    x_grid = np.linspace(x_min, x_max, n_grid, dtype=np.float64)

    series = [
        ("Gold test labels", gold, "#111111", 2.2),
        ("Average ensemble", avg, "#4C78A8", 1.8),
        ("Weighted ensemble", weighted, "#F58518", 1.8),
        ("MLP ensemble", mlp, "#54A24B", 1.8),
    ]

    for name, values, color, width in series:
        clipped = np.clip(values, x_min, x_max)
        density = kde_density(clipped, x_grid, bandwidth=bandwidth)
        ax.plot(x_grid, density, color=color, linewidth=width, label=f"{name} ({describe(values)})")
        ax.fill_between(x_grid, density, color=color, alpha=0.08)
        ax.axvline(float(np.mean(values)), color=color, linestyle="--", linewidth=1.0, alpha=0.8)

    ax.set_title("Smoothed Test Distribution: Gold vs Average / Weighted / MLP")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved plot to {out_path}")
    print(f"Weighted predictions: {weighted_path}")
    print(f"MLP predictions: {mlp_path}")


if __name__ == "__main__":
    main()
