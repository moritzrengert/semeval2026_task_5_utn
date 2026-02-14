#!/usr/bin/env python3
"""Create pie charts for ensemble contribution JSON outputs."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pie charts for MLP and weighted-average model contributions."
    )
    parser.add_argument(
        "--mlp-json",
        type=str,
        default="predictions/ensemble_sweep_runs/contrib_mlp_best.json",
        help="Path to the MLP contribution JSON.",
    )
    parser.add_argument(
        "--weighted-json",
        type=str,
        default="predictions/ensemble_sweep_runs/contrib_weighted_best.json",
        help="Path to the weighted-average contribution JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="predictions/ensemble_sweep_runs",
        help="Output directory for the pie chart PNG files.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Image DPI.")
    return parser.parse_args()


def load_shares(path: Path) -> Tuple[str, List[str], List[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "by_model" not in payload or not isinstance(payload["by_model"], list):
        raise ValueError(f"Expected 'by_model' list in {path}")

    labels: List[str] = []
    values: List[float] = []

    for row in payload["by_model"]:
        if not isinstance(row, dict):
            continue
        if "model" not in row:
            raise ValueError(f"Missing 'model' in {path}")

        if "abs_share" in row:
            v = float(row["abs_share"])
        elif "mean_abs_contribution" in row:
            v = float(row["mean_abs_contribution"])
        else:
            raise ValueError(
                f"Missing 'abs_share'/'mean_abs_contribution' for model '{row['model']}' in {path}"
            )

        if v < 0:
            raise ValueError(f"Negative contribution for model '{row['model']}' in {path}")
        labels.append(str(row["model"]))
        values.append(v)

    total = sum(values)
    if total <= 0:
        raise ValueError(f"Contribution sum must be > 0 in {path}")

    values = [v / total for v in values]
    method = str(payload.get("method", "unknown"))
    return method, labels, values


def _autopct(pct: float) -> str:
    return f"{pct:.1f}%"


def plot_pie(method: str, labels: List[str], values: List[float], out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.pie(
        values,
        labels=labels,
        autopct=_autopct,
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        textprops={"fontsize": 10},
    )
    ax.set_title(f"Model Contribution Shares ({method})")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    mlp_path = Path(args.mlp_json)
    weighted_path = Path(args.weighted_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlp_method, mlp_labels, mlp_values = load_shares(mlp_path)
    w_method, w_labels, w_values = load_shares(weighted_path)

    mlp_out = out_dir / "contrib_mlp_best_pie.png"
    weighted_out = out_dir / "contrib_weighted_best_pie.png"

    plot_pie(mlp_method, mlp_labels, mlp_values, mlp_out, args.dpi)
    plot_pie(w_method, w_labels, w_values, weighted_out, args.dpi)

    print(f"Saved: {mlp_out}")
    print(f"Saved: {weighted_out}")


if __name__ == "__main__":
    main()
