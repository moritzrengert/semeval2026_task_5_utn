#!/usr/bin/env python3
"""Run all ensemble methods using existing prediction JSONs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRED_DIR = ROOT / "predictions"
METHODS = ("average", "weighted", "linear_regression", "mlp")
BASE_MODELS = ("ares", "lmms", "nli", "sbert", "sensembert", "stsbert")


def ensure_base_predictions() -> None:
    missing = []
    for split in ("dev", "test"):
        for model in BASE_MODELS:
            path = PRED_DIR / f"{split}_preds_{model}.json"
            if not path.exists():
                missing.append(path)
    if missing:
        lines = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(f"Missing base prediction files:\n{lines}")


def main() -> None:
    ensure_base_predictions()
    script = ROOT / "src" / "ensemble" / "ensemble.py"
    for method in METHODS:
        print(f"[ensemble:{method}] running...")
        subprocess.run(
            [sys.executable, str(script), "--combine", method, "--show-contributions"],
            check=True,
            cwd=str(ROOT),
        )
    print("Done.")


if __name__ == "__main__":
    main()
