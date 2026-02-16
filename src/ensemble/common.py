"""Created by Moritz Rengert."""

import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data_utils import load_dataset, ordinal_class


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def numeric_sort_keys(keys: Sequence[str]) -> List[str]:
    try:
        return sorted(keys, key=lambda k: int(k))
    except Exception:
        return sorted(keys)


def derive_model_name(path_str: str) -> str:
    stem = Path(path_str).stem
    for prefix in ("dev_preds_", "test_preds_", "preds_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def load_prediction_file(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        keys = numeric_sort_keys(list(payload.keys()))
        return np.array([float(payload[k]) for k in keys], dtype=np.float32)

    if isinstance(payload, list):
        preds: List[float] = []
        for item in payload:
            if isinstance(item, dict):
                if "prediction" not in item:
                    raise ValueError("List item dict is missing 'prediction'.")
                preds.append(float(item["prediction"]))
            else:
                preds.append(float(item))
        return np.array(preds, dtype=np.float32)

    raise ValueError(f"Unsupported prediction JSON format in {path}")


def stack_prediction_files(paths: Sequence[str], split_name: str) -> np.ndarray:
    if not paths:
        raise ValueError(f"No {split_name} prediction files provided.")

    vectors: List[np.ndarray] = []
    for p in paths:
        vec = load_prediction_file(Path(p))
        vectors.append(vec)

    expected = vectors[0].shape[0]
    for idx, vec in enumerate(vectors):
        if vec.shape[0] != expected:
            raise ValueError(
                f"{split_name} file length mismatch at index {idx}: "
                f"expected {expected}, got {vec.shape[0]}"
            )

    return np.stack(vectors, axis=1)


def to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_ordered_records(path: Path) -> List[Any]:
    return load_dataset(path)


def load_labels(path: Path, label_key: str) -> np.ndarray:
    records = load_ordered_records(path)

    vals = []
    for rec in records:
        if isinstance(rec, dict):
            vals.append(to_float_or_nan(rec.get(label_key)))
        else:
            vals.append(float("nan"))
    return np.array(vals, dtype=np.float32)


def load_choices(path: Path) -> List[Optional[np.ndarray]]:
    records = load_ordered_records(path)
    all_choices: List[Optional[np.ndarray]] = []
    for rec in records:
        if not isinstance(rec, dict):
            all_choices.append(None)
            continue
        raw = rec.get("choices")
        if not isinstance(raw, (list, tuple)) or len(raw) == 0:
            all_choices.append(None)
            continue
        values = []
        for x in raw:
            fx = to_float_or_nan(x)
            if np.isfinite(fx):
                values.append(float(fx))
        if values:
            all_choices.append(np.array(values, dtype=np.float32))
        else:
            all_choices.append(None)
    return all_choices


def align_labels(y: np.ndarray, n: int, split_name: str, source: Path) -> np.ndarray:
    if y.shape[0] != n:
        raise ValueError(
            f"Label length mismatch for {split_name}: labels={y.shape[0]} vs predictions={n}. "
            f"Label source: {source}"
        )
    return y


def align_choices(
    choices: Sequence[Optional[np.ndarray]], n: int, split_name: str, source: Path
) -> List[Optional[np.ndarray]]:
    if len(choices) != n:
        raise ValueError(
            f"Choice length mismatch for {split_name}: choices={len(choices)} vs predictions={n}. "
            f"Choice source: {source}"
        )
    return list(choices)


def scores_to_classes(values: np.ndarray, num_classes: int) -> np.ndarray:
    return np.array(
        [ordinal_class(float(v), num_classes) for v in values.tolist()],
        dtype=np.int64,
    )


def safe_spearman(preds: np.ndarray, labels: np.ndarray) -> float:
    if preds.shape[0] < 2:
        return 0.0
    corr = spearmanr(preds, labels).correlation
    return 0.0 if corr is None or np.isnan(corr) else float(corr)


def acc_within_sd(
    preds: np.ndarray, choices: Sequence[Optional[np.ndarray]]
) -> Optional[Tuple[float, int]]:
    total = 0
    correct = 0
    for pred, item in zip(preds.tolist(), choices):
        if item is None or item.size == 0:
            continue
        vals = item.astype(np.float64)
        mean = float(np.mean(vals))
        if vals.size > 1:
            sd = float(np.std(vals, ddof=1))
        else:
            sd = 0.0
        within = ((mean - sd) < pred < (mean + sd)) or (abs(mean - pred) < 1.0)
        correct += 1 if within else 0
        total += 1
    if total == 0:
        return None
    return float(correct / total), int(total)


def metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 5,
    choices: Optional[Sequence[Optional[np.ndarray]]] = None,
) -> Optional[Dict[str, float]]:
    mask = np.isfinite(labels)
    if not np.any(mask):
        return None
    p = preds[mask]
    y = labels[mask]
    mse = float(np.mean((p - y) ** 2))
    mae = float(np.mean(np.abs(p - y)))
    spear = safe_spearman(p, y)
    p_cls = scores_to_classes(p, num_classes=num_classes)
    y_cls = scores_to_classes(y, num_classes=num_classes)
    acc = float(np.mean(p_cls == y_cls))

    acc_sd = None
    acc_sd_n = 0
    if choices is not None:
        filtered_choices = [c for c, keep in zip(choices, mask.tolist()) if keep]
        acc_sd_tuple = acc_within_sd(p, filtered_choices)
        if acc_sd_tuple is not None:
            acc_sd, acc_sd_n = acc_sd_tuple

    out: Dict[str, Any] = {
        "mse": mse,
        "mae": mae,
        "spearman": spear,
        "accuracy": acc,
        "n": int(mask.sum()),
    }
    if acc_sd is not None:
        out["acc_within_sd"] = float(acc_sd)
        out["acc_within_sd_n"] = int(acc_sd_n)
    return out


def metrics_overall(
    pred_splits: Sequence[np.ndarray],
    label_splits: Sequence[np.ndarray],
    num_classes: int = 5,
    choice_splits: Optional[Sequence[Sequence[Optional[np.ndarray]]]] = None,
) -> Optional[Dict[str, float]]:
    if len(pred_splits) != len(label_splits):
        raise ValueError("pred_splits and label_splits must have equal length")
    if choice_splits is not None and len(choice_splits) != len(pred_splits):
        raise ValueError("choice_splits length must match pred_splits")

    pred_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    merged_choices: List[Optional[np.ndarray]] = []
    for idx, (preds, labels) in enumerate(zip(pred_splits, label_splits)):
        mask = np.isfinite(labels)
        if np.any(mask):
            pred_parts.append(preds[mask])
            label_parts.append(labels[mask])
            if choice_splits is not None:
                split_choices = choice_splits[idx]
                merged_choices.extend([c for c, keep in zip(split_choices, mask.tolist()) if keep])

    if not pred_parts:
        return None

    preds_all = np.concatenate(pred_parts, axis=0)
    labels_all = np.concatenate(label_parts, axis=0)
    return metrics(
        preds_all,
        labels_all,
        num_classes=num_classes,
        choices=merged_choices if choice_splits is not None else None,
    )


def print_metrics(title: str, result: Optional[Dict[str, float]]) -> None:
    if result is None:
        print(f"{title}: n/a (no numeric labels)")
        return
    parts = [
        f"{title}: n={result['n']}",
        f"mse={result['mse']:.4f}",
        f"mae={result['mae']:.4f}",
        f"spearman={result['spearman']:.4f}",
        f"accuracy={result['accuracy']:.4f}",
    ]
    if "acc_within_sd" in result:
        parts.append(f"acc_within_sd={result['acc_within_sd']:.4f}")
    print(" ".join(parts))


def parse_float_list(raw: Optional[str], default: Sequence[float], label: str) -> List[float]:
    if raw is None:
        return [float(x) for x in default]
    values = [s.strip() for s in raw.split(",") if s.strip()]
    if not values:
        raise ValueError(f"{label} cannot be empty")
    return [float(v) for v in values]


def parse_int_list(raw: Optional[str], default: Sequence[int], label: str) -> List[int]:
    if raw is None:
        return [int(x) for x in default]
    values = [s.strip() for s in raw.split(",") if s.strip()]
    if not values:
        raise ValueError(f"{label} cannot be empty")
    return [int(v) for v in values]


def build_grid(param_space: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_space.keys())
    return [dict(zip(keys, combo)) for combo in product(*(param_space[k] for k in keys))]


def build_kfold_indices(n_items: int, n_folds: int, seed: int) -> List[np.ndarray]:
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n_items < n_folds:
        raise ValueError(f"n_folds ({n_folds}) cannot exceed number of samples ({n_items})")
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items, dtype=np.int64)
    rng.shuffle(indices)
    return [part.astype(np.int64) for part in np.array_split(indices, n_folds)]


def is_higher_better(metric_name: str) -> bool:
    return metric_name in {"spearman", "accuracy", "acc_within_sd"}


def extract_objective(metric_result: Dict[str, float], metric_name: str) -> float:
    if metric_name not in metric_result:
        raise ValueError(
            f"Metric '{metric_name}' is unavailable for selection. "
            f"Available: {', '.join(sorted(metric_result.keys()))}"
        )
    return float(metric_result[metric_name])

def write_predictions(path: Path, preds: np.ndarray) -> None:
    payload = {str(i): float(v) for i, v in enumerate(preds.tolist())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
