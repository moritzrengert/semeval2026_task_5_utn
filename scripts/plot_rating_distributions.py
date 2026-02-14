#!/usr/bin/env python3
"""Plot float prediction distributions for all dev/test prediction JSON pairs."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def numeric_sort_keys(keys: Sequence[str]) -> List[str]:
    try:
        return sorted(keys, key=lambda k: int(k))
    except Exception:
        return sorted(keys)


def load_predictions(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        keys = numeric_sort_keys(list(payload.keys()))
        return np.array([float(payload[k]) for k in keys], dtype=np.float32)

    if isinstance(payload, list):
        values: List[float] = []
        for item in payload:
            if isinstance(item, dict):
                if "prediction" not in item:
                    raise ValueError("List item dict is missing 'prediction' key.")
                values.append(float(item["prediction"]))
            else:
                values.append(float(item))
        return np.array(values, dtype=np.float32)

    raise ValueError(f"Unsupported JSON format in {path}")


def discover_dev_test_pairs(pred_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    dev_map: Dict[str, Path] = {}
    test_map: Dict[str, Path] = {}

    for path in sorted(pred_dir.glob("dev_preds_*.json")):
        model_name = path.stem[len("dev_preds_") :]
        dev_map[model_name] = path

    for path in sorted(pred_dir.glob("test_preds_*.json")):
        model_name = path.stem[len("test_preds_") :]
        test_map[model_name] = path

    common = sorted(set(dev_map.keys()) & set(test_map.keys()))
    return {name: (dev_map[name], test_map[name]) for name in common}


def plot_panel(
    ax,
    scores: np.ndarray,
    title: str,
    color: str,
    bins: int,
    x_min: float,
    x_max: float,
    y_max: float,
) -> None:
    clipped = np.clip(scores, x_min, x_max)
    ax.hist(
        clipped,
        bins=bins,
        range=(x_min, x_max),
        color=color,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Count")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", alpha=0.2, linestyle="--")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot float prediction distributions for all matched dev/test model files."
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="predictions",
        help="Directory containing dev_preds_*.json and test_preds_*.json files.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model names to plot (suffix after dev_preds_/test_preds_).",
    )
    parser.add_argument("--bins", type=int, default=30, help="Number of histogram bins.")
    parser.add_argument("--x-min", type=float, default=1.0)
    parser.add_argument("--x-max", type=float, default=5.0)
    parser.add_argument(
        "--out",
        type=str,
        default="predictions/plots/all_models_float_distributions_dev_test.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Float Prediction Distributions Across Models (Dev/Test)",
        help="Figure title.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--fig-width", type=float, default=12.0)
    parser.add_argument("--row-height", type=float, default=2.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        raise SystemExit(f"Prediction directory not found: {pred_dir}")

    pairs = discover_dev_test_pairs(pred_dir)
    if not pairs:
        raise SystemExit(
            f"No matching dev/test prediction pairs found in {pred_dir}. "
            "Expected files like dev_preds_<model>.json and test_preds_<model>.json."
        )

    if args.models:
        selected = []
        missing = []
        for name in args.models:
            if name in pairs:
                selected.append(name)
            else:
                missing.append(name)
        if missing:
            raise SystemExit(f"Requested model(s) not found as dev/test pair(s): {', '.join(missing)}")
    else:
        selected = sorted(pairs.keys())

    loaded: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in selected:
        dev_path, test_path = pairs[name]
        loaded[name] = (load_predictions(dev_path), load_predictions(test_path))

    x_min = float(args.x_min)
    x_max = float(args.x_max)
    if x_max <= x_min:
        raise SystemExit("--x-max must be greater than --x-min")

    hist_max = 0
    for name in selected:
        dev_arr, test_arr = loaded[name]
        for arr in (dev_arr, test_arr):
            clipped = np.clip(arr, x_min, x_max)
            counts, _ = np.histogram(clipped, bins=args.bins, range=(x_min, x_max))
            hist_max = max(hist_max, int(counts.max()))
    y_max = max(5, int(hist_max * 1.18))

    n_rows = len(selected)
    fig_height = max(4.0, args.row_height * n_rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(args.fig_width, fig_height), sharex=True, sharey=True)
    if n_rows == 1:
        axes = np.array([axes])

    for row_idx, name in enumerate(selected):
        dev_arr, test_arr = loaded[name]
        plot_panel(
            axes[row_idx, 0],
            dev_arr,
            f"{name} - Dev (n={len(dev_arr)})",
            "#4C78A8",
            args.bins,
            x_min,
            x_max,
            y_max,
        )
        plot_panel(
            axes[row_idx, 1],
            test_arr,
            f"{name} - Test (n={len(test_arr)})",
            "#E45756",
            args.bins,
            x_min,
            x_max,
            y_max,
        )

    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved plot to {out_path}")
    print(f"Models plotted ({len(selected)}): {', '.join(selected)}")


if __name__ == "__main__":
    main()
