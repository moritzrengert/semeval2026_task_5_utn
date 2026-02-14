#!/usr/bin/env python3
"""
Ensemble precomputed prediction JSON files.

Supported combine modes:
- average: simple mean across models
- weighted: weighted average; pass --weights or learn weights on dev labels
- mlp: train a small MLP on dev labels

Training convention for weighted/mlp:
- train on dev predictions + dev labels
- if test labels are unavailable, use dev-only cross-fitting for model selection
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


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
    payload = json.loads(path.read_text(encoding="utf-8"))

    records: List[Any] = []
    if isinstance(payload, dict):
        keys = numeric_sort_keys(list(payload.keys()))
        records = [payload[k] for k in keys]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Unsupported JSON format in {path}")
    return records


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
    # Same binning as ordinal_class in training code:
    # class = floor(score - 0.5), clipped to [0, num_classes-1]
    classes = np.floor(values - 0.5).astype(np.int64)
    classes = np.clip(classes, 0, num_classes - 1)
    return classes


def safe_spearman(preds: np.ndarray, labels: np.ndarray) -> float:
    if preds.shape[0] < 2:
        return 0.0
    if spearmanr is not None:
        corr = spearmanr(preds, labels).correlation
        return 0.0 if corr is None or np.isnan(corr) else float(corr)

    # Fallback rank correlation without scipy.
    rp = np.argsort(np.argsort(preds))
    rl = np.argsort(np.argsort(labels))
    corr = np.corrcoef(rp, rl)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


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


def parse_weights(raw: str, n_models: int) -> np.ndarray:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(values) != n_models:
        raise ValueError(f"Expected {n_models} weights, got {len(values)}")
    w = np.array(values, dtype=np.float64)
    total = float(np.sum(w))
    if abs(total) < 1e-12:
        raise ValueError("Weight sum cannot be zero")
    return (w / total).astype(np.float32)


def contribution_from_weighted(x: np.ndarray, weights: np.ndarray, model_names: Sequence[str]) -> Dict[str, Any]:
    contrib = x * weights.reshape(1, -1)
    mean_signed = np.mean(contrib, axis=0)
    mean_abs = np.mean(np.abs(contrib), axis=0)
    total_abs = float(np.sum(mean_abs))
    if total_abs <= 0:
        share = np.zeros_like(mean_abs)
    else:
        share = mean_abs / total_abs

    by_model = []
    for name, w, ms, ma, sh in zip(model_names, weights.tolist(), mean_signed.tolist(), mean_abs.tolist(), share.tolist()):
        by_model.append(
            {
                "model": str(name),
                "weight": float(w),
                "mean_signed_contribution": float(ms),
                "mean_abs_contribution": float(ma),
                "abs_share": float(sh),
            }
        )
    return {
        "method": "weighted_linear",
        "n_samples": int(x.shape[0]),
        "by_model": by_model,
    }


def fit_weighted_average(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    epochs: int,
    lr: float,
    l2: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    n_models = x_train.shape[1]
    logits = np.zeros((n_models,), dtype=np.float64)

    use_val = x_val is not None and y_val is not None and len(y_val) > 0
    best_score = float("inf")
    best_epoch = 1
    best_weights = softmax(logits).copy()

    for epoch in range(1, epochs + 1):
        w = softmax(logits)
        pred = x_train @ w
        err = pred - y_train

        # MSE + L2 on weights
        grad_w = (2.0 / x_train.shape[0]) * (x_train.T @ err) + 2.0 * l2 * w
        # d softmax: J^T g = w * (g - <g, w>)
        grad_logits = w * (grad_w - float(np.dot(grad_w, w)))
        logits -= lr * grad_logits

        w_eval = softmax(logits)
        if use_val:
            val_pred = x_val @ w_eval
            score = float(np.mean((val_pred - y_val) ** 2))
        else:
            train_pred = x_train @ w_eval
            score = float(np.mean((train_pred - y_train) ** 2))

        if score < best_score:
            best_score = score
            best_weights = w_eval.copy()
            best_epoch = epoch

    info = {
        "best_objective": float(best_score),
        "best_epoch": int(best_epoch),
        "used_validation": bool(use_val),
    }
    return best_weights.astype(np.float32), info


def fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    patience: int,
    device_str: Optional[str],
    seed: int,
    progress_every: int = 10,
    progress_prefix: str = "[mlp] ",
):
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit("Torch is required for --combine mlp") from exc

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    train_ds = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)

    use_val = x_val is not None and y_val is not None and len(y_val) > 0
    if use_val:
        x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    model = nn.Sequential(
        nn.Linear(x_train.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_metric = float("inf")
    best_epoch = 1
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        n_items = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            batch_n = int(xb.shape[0])
            loss_total += float(loss.item()) * batch_n
            n_items += batch_n
        train_loss = loss_total / max(1, n_items)

        model.eval()
        with torch.no_grad():
            if use_val:
                score = float(loss_fn(model(x_val_t).squeeze(-1), y_val_t).item())
            else:
                score = float(loss_fn(model(x_train_t.to(device)).squeeze(-1), y_train_t.to(device)).item())

        if score < best_metric - 1e-10:
            best_metric = score
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            improved = True
        else:
            epochs_no_improve += 1
            improved = False

        if progress_every > 0 and (epoch == 1 or epoch % progress_every == 0 or epoch == epochs):
            target_name = "val_mse" if use_val else "train_eval_mse"
            improved_text = "yes" if improved else "no"
            print(
                f"{progress_prefix}epoch={epoch}/{epochs} "
                f"train_mse={train_loss:.6f} {target_name}={score:.6f} "
                f"best={best_metric:.6f} best_epoch={best_epoch} "
                f"patience={epochs_no_improve}/{patience} improved={improved_text}"
            )

        if epochs_no_improve >= patience:
            if progress_every > 0:
                print(
                    f"{progress_prefix}early-stop at epoch {epoch}; "
                    f"best_epoch={best_epoch}, best_objective={best_metric:.6f}"
                )
            break

    model.load_state_dict(best_state)
    info = {
        "best_objective": float(best_metric),
        "best_epoch": int(best_epoch),
        "used_validation": bool(use_val),
        "device": str(device),
    }
    return model, info


def predict_mlp(model, x: np.ndarray, device_str: Optional[str]) -> np.ndarray:
    import torch

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval().to(device)
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        preds = model(x_t).squeeze(-1).detach().cpu().numpy()
    return preds.astype(np.float32)


def contribution_from_mlp_perturbation(
    model,
    x: np.ndarray,
    model_names: Sequence[str],
    device_str: Optional[str],
    baseline: str = "mean",
) -> Dict[str, Any]:
    if baseline not in {"mean", "median"}:
        raise ValueError("baseline must be one of: mean, median")

    if baseline == "mean":
        base_vec = np.mean(x, axis=0)
    else:
        base_vec = np.median(x, axis=0)

    base_pred = predict_mlp(model, x, device_str).astype(np.float64)

    deltas_signed: List[float] = []
    deltas_abs: List[float] = []
    for feat_idx in range(x.shape[1]):
        x_pert = x.copy()
        x_pert[:, feat_idx] = float(base_vec[feat_idx])
        pert_pred = predict_mlp(model, x_pert, device_str).astype(np.float64)
        delta = base_pred - pert_pred
        deltas_signed.append(float(np.mean(delta)))
        deltas_abs.append(float(np.mean(np.abs(delta))))

    abs_arr = np.array(deltas_abs, dtype=np.float64)
    total_abs = float(np.sum(abs_arr))
    if total_abs <= 0:
        share = np.zeros_like(abs_arr)
    else:
        share = abs_arr / total_abs

    by_model = []
    for name, ms, ma, sh in zip(model_names, deltas_signed, deltas_abs, share.tolist()):
        by_model.append(
            {
                "model": str(name),
                "mean_signed_delta": float(ms),
                "mean_abs_delta": float(ma),
                "abs_share": float(sh),
            }
        )
    return {
        "method": "mlp_perturbation",
        "baseline": baseline,
        "n_samples": int(x.shape[0]),
        "by_model": by_model,
    }


def print_contributions(title: str, contribution: Dict[str, Any]) -> None:
    print(title)
    method = contribution.get("method", "unknown")
    print(f"  method={method} n={contribution.get('n_samples', 'n/a')}")
    for item in contribution.get("by_model", []):
        name = item.get("model", "unknown")
        if "weight" in item:
            print(
                f"  {name}: weight={item['weight']:.6f} "
                f"mean_signed={item['mean_signed_contribution']:.6f} "
                f"mean_abs={item['mean_abs_contribution']:.6f} "
                f"share={item['abs_share']:.4f}"
            )
        else:
            print(
                f"  {name}: mean_signed_delta={item['mean_signed_delta']:.6f} "
                f"mean_abs_delta={item['mean_abs_delta']:.6f} "
                f"share={item['abs_share']:.4f}"
            )


def fit_weighted_crossfit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    x_test: np.ndarray,
    train_mask: np.ndarray,
    train_choices: Sequence[Optional[np.ndarray]],
    num_classes: int,
    cv_folds: int,
    cv_seed: int,
    cv_objective: str,
    epochs_grid: Sequence[int],
    lr_grid: Sequence[float],
    l2_grid: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    folds = build_kfold_indices(x_train.shape[0], cv_folds, cv_seed)
    maximize = is_higher_better(cv_objective)
    best_score = float("-inf") if maximize else float("inf")
    best_payload: Optional[Dict[str, Any]] = None
    search_log: List[Dict[str, Any]] = []

    for epochs in epochs_grid:
        for lr in lr_grid:
            for l2 in l2_grid:
                oof = np.zeros((x_train.shape[0],), dtype=np.float32)
                dev_sum = np.zeros((x_dev.shape[0],), dtype=np.float32)
                test_sum = np.zeros((x_test.shape[0],), dtype=np.float32)
                fold_weights: List[np.ndarray] = []
                fold_info: List[Dict[str, float]] = []

                for fold_idx, val_idx in enumerate(folds):
                    train_idx = np.concatenate([f for i, f in enumerate(folds) if i != fold_idx])
                    weights, info = fit_weighted_average(
                        x_train=x_train[train_idx],
                        y_train=y_train[train_idx],
                        x_val=x_train[val_idx],
                        y_val=y_train[val_idx],
                        epochs=int(epochs),
                        lr=float(lr),
                        l2=float(l2),
                    )
                    oof[val_idx] = (x_train[val_idx] @ weights).astype(np.float32)
                    dev_sum += (x_dev @ weights).astype(np.float32)
                    test_sum += (x_test @ weights).astype(np.float32)
                    fold_weights.append(weights.astype(np.float32))
                    fold_info.append(info)

                metric_result = metrics(
                    oof,
                    y_train,
                    num_classes=num_classes,
                    choices=train_choices,
                )
                if metric_result is None:
                    raise RuntimeError("Cross-fit metric computation failed: no numeric labels.")

                objective_value = extract_objective(metric_result, cv_objective)
                search_log.append(
                    {
                        "epochs": int(epochs),
                        "lr": float(lr),
                        "l2": float(l2),
                        "objective": float(objective_value),
                        "metrics": metric_result,
                    }
                )

                is_better = objective_value > best_score if maximize else objective_value < best_score
                if is_better:
                    best_score = float(objective_value)
                    best_payload = {
                        "epochs": int(epochs),
                        "lr": float(lr),
                        "l2": float(l2),
                        "oof": oof,
                        "dev_sum": dev_sum,
                        "test_sum": test_sum,
                        "weights": fold_weights,
                        "fold_info": fold_info,
                        "metrics": metric_result,
                    }

    if best_payload is None:
        raise RuntimeError("Weighted cross-fit failed to produce a model.")

    n_folds = float(len(folds))
    dev_bagged = best_payload["dev_sum"] / n_folds
    test_bagged = best_payload["test_sum"] / n_folds

    dev_preds = dev_bagged.astype(np.float32)
    train_indices = np.where(train_mask)[0]
    dev_preds[train_indices] = best_payload["oof"]

    weights_stack = np.stack(best_payload["weights"], axis=0).astype(np.float32)
    info: Dict[str, Any] = {
        "source": "crossfit",
        "cv_folds": int(cv_folds),
        "cv_seed": int(cv_seed),
        "cv_objective": cv_objective,
        "selected": {
            "epochs": int(best_payload["epochs"]),
            "lr": float(best_payload["lr"]),
            "l2": float(best_payload["l2"]),
            "objective": float(best_score),
            "metrics": best_payload["metrics"],
        },
        "weights_mean": [float(v) for v in weights_stack.mean(axis=0).tolist()],
        "weights_std": [float(v) for v in weights_stack.std(axis=0).tolist()],
        "search_results": search_log,
        "uses_oof_dev_predictions": True,
        "fold_count": int(len(folds)),
    }
    return dev_preds, test_bagged.astype(np.float32), info


def fit_mlp_crossfit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    x_test: np.ndarray,
    train_mask: np.ndarray,
    train_choices: Sequence[Optional[np.ndarray]],
    num_classes: int,
    cv_folds: int,
    cv_seed: int,
    cv_objective: str,
    hidden_grid: Sequence[int],
    dropout_grid: Sequence[float],
    lr_grid: Sequence[float],
    weight_decay_grid: Sequence[float],
    epochs: int,
    batch_size: int,
    patience: int,
    device_str: Optional[str],
    seed: int,
    progress_every: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    folds = build_kfold_indices(x_train.shape[0], cv_folds, cv_seed)
    maximize = is_higher_better(cv_objective)
    best_score = float("-inf") if maximize else float("inf")
    best_payload: Optional[Dict[str, Any]] = None
    search_log: List[Dict[str, Any]] = []

    for hidden_dim in hidden_grid:
        for dropout in dropout_grid:
            for lr in lr_grid:
                for weight_decay in weight_decay_grid:
                    if progress_every > 0:
                        print(
                            "[mlp-cv] "
                            f"config hidden_dim={int(hidden_dim)} dropout={float(dropout)} "
                            f"lr={float(lr)} weight_decay={float(weight_decay)}"
                        )
                    oof = np.zeros((x_train.shape[0],), dtype=np.float32)
                    dev_sum = np.zeros((x_dev.shape[0],), dtype=np.float32)
                    test_sum = np.zeros((x_test.shape[0],), dtype=np.float32)
                    fold_info: List[Dict[str, Any]] = []

                    for fold_idx, val_idx in enumerate(folds):
                        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != fold_idx])
                        fold_prefix = (
                            "[mlp-cv] "
                            f"cfg(h={int(hidden_dim)},d={float(dropout)},lr={float(lr)},wd={float(weight_decay)}) "
                            f"fold={fold_idx + 1}/{len(folds)} "
                        )
                        model, info = fit_mlp(
                            x_train=x_train[train_idx],
                            y_train=y_train[train_idx],
                            x_val=x_train[val_idx],
                            y_val=y_train[val_idx],
                            hidden_dim=int(hidden_dim),
                            dropout=float(dropout),
                            lr=float(lr),
                            weight_decay=float(weight_decay),
                            epochs=int(epochs),
                            batch_size=int(batch_size),
                            patience=int(patience),
                            device_str=device_str,
                            seed=int(seed + fold_idx),
                            progress_every=progress_every,
                            progress_prefix=fold_prefix,
                        )
                        oof[val_idx] = predict_mlp(model, x_train[val_idx], device_str)
                        dev_sum += predict_mlp(model, x_dev, device_str)
                        test_sum += predict_mlp(model, x_test, device_str)
                        fold_info.append(info)

                    metric_result = metrics(
                        oof,
                        y_train,
                        num_classes=num_classes,
                        choices=train_choices,
                    )
                    if metric_result is None:
                        raise RuntimeError("Cross-fit metric computation failed: no numeric labels.")

                    objective_value = extract_objective(metric_result, cv_objective)
                    search_log.append(
                        {
                            "hidden_dim": int(hidden_dim),
                            "dropout": float(dropout),
                            "lr": float(lr),
                            "weight_decay": float(weight_decay),
                            "objective": float(objective_value),
                            "metrics": metric_result,
                        }
                    )
                    is_better = objective_value > best_score if maximize else objective_value < best_score
                    if is_better:
                        best_score = float(objective_value)
                        best_payload = {
                            "hidden_dim": int(hidden_dim),
                            "dropout": float(dropout),
                            "lr": float(lr),
                            "weight_decay": float(weight_decay),
                            "oof": oof,
                            "dev_sum": dev_sum,
                            "test_sum": test_sum,
                            "fold_info": fold_info,
                            "metrics": metric_result,
                        }
                    if progress_every > 0:
                        print(
                            "[mlp-cv] "
                            f"config-result objective({cv_objective})={float(objective_value):.6f} "
                            f"best_so_far={float(best_score):.6f}"
                        )

    if best_payload is None:
        raise RuntimeError("MLP cross-fit failed to produce a model.")

    n_folds = float(len(folds))
    dev_bagged = best_payload["dev_sum"] / n_folds
    test_bagged = best_payload["test_sum"] / n_folds

    dev_preds = dev_bagged.astype(np.float32)
    train_indices = np.where(train_mask)[0]
    dev_preds[train_indices] = best_payload["oof"]

    best_epochs = [int(info["best_epoch"]) for info in best_payload["fold_info"]]
    info = {
        "source": "crossfit",
        "cv_folds": int(cv_folds),
        "cv_seed": int(cv_seed),
        "cv_objective": cv_objective,
        "selected": {
            "hidden_dim": int(best_payload["hidden_dim"]),
            "dropout": float(best_payload["dropout"]),
            "lr": float(best_payload["lr"]),
            "weight_decay": float(best_payload["weight_decay"]),
            "objective": float(best_score),
            "metrics": best_payload["metrics"],
        },
        "mean_best_epoch": float(np.mean(best_epochs)),
        "fold_best_epochs": best_epochs,
        "search_results": search_log,
        "uses_oof_dev_predictions": True,
        "fold_count": int(len(folds)),
    }
    return dev_preds, test_bagged.astype(np.float32), info


def write_predictions(path: Path, preds: np.ndarray) -> None:
    payload = {str(i): float(v) for i, v in enumerate(preds.tolist())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ensemble prediction JSONs from multiple models.")

    ap.add_argument("--dev-preds", nargs="+", required=True, help="Dev prediction JSON files (one per model).")
    ap.add_argument("--test-preds", nargs="+", required=True, help="Test prediction JSON files (same order as --dev-preds).")
    ap.add_argument("--model-names", nargs="+", default=None, help="Optional model names in the same order.")

    ap.add_argument("--dev-labels-path", type=str, default="semeval26-05-scripts/data/dev.json")
    ap.add_argument("--test-labels-path", type=str, default="semeval26-05-scripts/data/test.json")
    ap.add_argument("--label-key", type=str, default="average")
    ap.add_argument("--num-classes", type=int, default=5)

    ap.add_argument("--combine", choices=["average", "weighted", "mlp"], default="average")
    ap.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Comma-separated weights for weighted mode. If omitted, weights are learned on dev labels.",
    )

    ap.add_argument("--out-dev", type=str, default="predictions/dev_preds_ensemble.json")
    ap.add_argument("--out-test", type=str, default="predictions/test_preds_ensemble.json")
    ap.add_argument("--save-meta", type=str, default=None, help="Optional path to save method metadata (JSON).")

    # Weighted fit options
    ap.add_argument("--weight-epochs", type=int, default=1500)
    ap.add_argument("--weight-lr", type=float, default=0.05)
    ap.add_argument("--weight-l2", type=float, default=1e-4)
    ap.add_argument(
        "--weight-epochs-grid",
        type=str,
        default=None,
        help="Comma-separated epoch candidates for weighted CV sweep.",
    )
    ap.add_argument(
        "--weight-lr-grid",
        type=str,
        default=None,
        help="Comma-separated learning-rate candidates for weighted CV sweep.",
    )
    ap.add_argument(
        "--weight-l2-grid",
        type=str,
        default=None,
        help="Comma-separated L2 candidates for weighted CV sweep.",
    )

    # MLP options
    ap.add_argument("--mlp-hidden-dim", type=int, default=16)
    ap.add_argument("--mlp-dropout", type=float, default=0.1)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    ap.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    ap.add_argument("--mlp-epochs", type=int, default=400)
    ap.add_argument("--mlp-batch-size", type=int, default=64)
    ap.add_argument("--mlp-patience", type=int, default=40)
    ap.add_argument(
        "--mlp-progress-every",
        type=int,
        default=10,
        help="Print MLP training progress every N epochs (<=0 disables).",
    )
    ap.add_argument(
        "--mlp-hidden-grid",
        type=str,
        default=None,
        help="Comma-separated hidden-dim candidates for MLP CV sweep.",
    )
    ap.add_argument(
        "--mlp-dropout-grid",
        type=str,
        default=None,
        help="Comma-separated dropout candidates for MLP CV sweep.",
    )
    ap.add_argument(
        "--mlp-lr-grid",
        type=str,
        default=None,
        help="Comma-separated learning-rate candidates for MLP CV sweep.",
    )
    ap.add_argument(
        "--mlp-weight-decay-grid",
        type=str,
        default=None,
        help="Comma-separated weight-decay candidates for MLP CV sweep.",
    )

    ap.add_argument("--cv-folds", type=int, default=5, help="Number of folds for dev-only cross-fitting.")
    ap.add_argument("--cv-seed", type=int, default=42, help="Seed for dev-only cross-fitting.")
    ap.add_argument(
        "--cv-objective",
        choices=["mse", "mae", "spearman", "accuracy", "acc_within_sd"],
        default="spearman",
        help="Selection metric for CV sweeps.",
    )
    ap.add_argument(
        "--force-dev-cv",
        action="store_true",
        help="Force dev-only CV for combiner selection, even if numeric test labels are available.",
    )

    ap.add_argument("--device", type=str, default=None, help="Device for MLP mode (e.g. cpu, cuda:0).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--show-model-metrics",
        action="store_true",
        help="Print per-model dev/test metrics (disabled by default).",
    )
    ap.add_argument(
        "--show-split-metrics",
        action="store_true",
        help="Print dev/test ensemble metrics in addition to overall ensemble metrics.",
    )
    ap.add_argument(
        "--show-weights",
        action="store_true",
        help="Print learned/manual weights in weighted mode.",
    )
    ap.add_argument(
        "--show-contributions",
        action="store_true",
        help="Print per-model contribution statistics for weighted/mlp ensembles.",
    )
    ap.add_argument(
        "--save-contributions",
        type=str,
        default=None,
        help="Optional path to save per-model contribution statistics as JSON.",
    )
    ap.add_argument(
        "--contrib-split",
        choices=["dev", "test"],
        default="test",
        help="Which split to use for contribution analysis.",
    )
    ap.add_argument(
        "--mlp-contrib-baseline",
        choices=["mean", "median"],
        default="mean",
        help="Baseline replacement used for MLP perturbation contributions.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.dev_preds) != len(args.test_preds):
        raise SystemExit("--dev-preds and --test-preds must contain the same number of files")

    x_dev = stack_prediction_files(args.dev_preds, "dev")
    x_test = stack_prediction_files(args.test_preds, "test")

    if x_dev.shape[1] != x_test.shape[1]:
        raise SystemExit("Dev/test model count mismatch")

    n_models = x_dev.shape[1]
    model_names = args.model_names or [derive_model_name(p) for p in args.dev_preds]
    if len(model_names) != n_models:
        raise SystemExit(f"Expected {n_models} model names, got {len(model_names)}")

    dev_label_path = Path(args.dev_labels_path)
    test_label_path = Path(args.test_labels_path)
    y_dev = align_labels(load_labels(dev_label_path, args.label_key), x_dev.shape[0], "dev", dev_label_path)
    y_test = align_labels(load_labels(test_label_path, args.label_key), x_test.shape[0], "test", test_label_path)
    choices_dev = align_choices(load_choices(dev_label_path), x_dev.shape[0], "dev", dev_label_path)
    choices_test = align_choices(load_choices(test_label_path), x_test.shape[0], "test", test_label_path)

    train_mask = np.isfinite(y_dev)
    if args.combine in {"weighted", "mlp"} and not np.any(train_mask):
        raise SystemExit("No numeric dev labels available for training combiner.")

    val_mask = np.isfinite(y_test)
    has_val = np.any(val_mask)
    if args.combine in {"weighted", "mlp"} and not has_val:
        print("[info] Test labels are non-numeric or missing.")

    x_train = x_dev[train_mask]
    y_train = y_dev[train_mask]
    x_val = x_test[val_mask] if has_val else None
    y_val = y_test[val_mask] if has_val else None
    train_choices = [c for c, keep in zip(choices_dev, train_mask.tolist()) if keep]

    use_dev_cv = False
    if args.combine in {"weighted", "mlp"}:
        cv_eligible = args.combine == "mlp" or (args.combine == "weighted" and not args.weights)
        use_dev_cv = bool(cv_eligible and (args.force_dev_cv or (not has_val and args.cv_folds > 1)))
        if cv_eligible and use_dev_cv:
            print(
                f"[info] Using dev-only {args.cv_folds}-fold cross-fit "
                f"(objective={args.cv_objective}) for {args.combine}."
            )
        elif cv_eligible and not has_val:
            print(
                "[info] No numeric test labels and --cv-folds <= 1; "
                "combiner will be selected on dev train loss only."
            )
        elif cv_eligible and has_val:
            print(f"[info] Using test labels for validation for {args.combine} (dev is training split).")

    if args.show_model_metrics:
        for idx, name in enumerate(model_names):
            print_metrics(
                f"Dev {name}",
                metrics(x_dev[:, idx], y_dev, num_classes=args.num_classes, choices=choices_dev),
            )
            print_metrics(
                f"Test {name}",
                metrics(x_test[:, idx], y_test, num_classes=args.num_classes, choices=choices_test),
            )

    meta: Dict[str, Any] = {
        "combine": args.combine,
        "model_names": model_names,
        "dev_pred_files": args.dev_preds,
        "test_pred_files": args.test_preds,
    }
    contributions_payload: Optional[Dict[str, Any]] = None

    if args.combine == "average":
        dev_ens = np.mean(x_dev, axis=1)
        test_ens = np.mean(x_test, axis=1)
        meta["weights"] = [1.0 / n_models] * n_models

    elif args.combine == "weighted":
        if args.weights:
            weights = parse_weights(args.weights, n_models)
            train_info = {"source": "manual", "used_validation": False}
            dev_ens = x_dev @ weights
            test_ens = x_test @ weights
        else:
            if use_dev_cv:
                epochs_grid = parse_int_list(
                    args.weight_epochs_grid,
                    [args.weight_epochs],
                    "--weight-epochs-grid",
                )
                lr_grid = parse_float_list(
                    args.weight_lr_grid,
                    [args.weight_lr],
                    "--weight-lr-grid",
                )
                l2_grid = parse_float_list(
                    args.weight_l2_grid,
                    [args.weight_l2],
                    "--weight-l2-grid",
                )
                dev_ens, test_ens, cv_info = fit_weighted_crossfit(
                    x_train=x_train,
                    y_train=y_train,
                    x_dev=x_dev,
                    x_test=x_test,
                    train_mask=train_mask,
                    train_choices=train_choices,
                    num_classes=args.num_classes,
                    cv_folds=args.cv_folds,
                    cv_seed=args.cv_seed,
                    cv_objective=args.cv_objective,
                    epochs_grid=epochs_grid,
                    lr_grid=lr_grid,
                    l2_grid=l2_grid,
                )
                weights = np.array(cv_info["weights_mean"], dtype=np.float32)
                train_info = cv_info
            else:
                weights, fit_info = fit_weighted_average(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    epochs=args.weight_epochs,
                    lr=args.weight_lr,
                    l2=args.weight_l2,
                )
                train_info = {"source": "learned", **fit_info}
                dev_ens = x_dev @ weights
                test_ens = x_test @ weights
        meta["weights"] = [float(w) for w in weights.tolist()]
        meta["weighted_info"] = train_info
        if args.show_weights:
            print("Weighted coefficients:")
            for name, w in zip(model_names, weights.tolist()):
                print(f"  {name}: {w:.6f}")

        x_contrib = x_test if args.contrib_split == "test" else x_dev
        contributions_payload = contribution_from_weighted(x_contrib, weights, model_names)
        contributions_payload["split"] = args.contrib_split

    else:  # mlp
        mlp_model = None
        if use_dev_cv:
            hidden_grid = parse_int_list(
                args.mlp_hidden_grid,
                [args.mlp_hidden_dim],
                "--mlp-hidden-grid",
            )
            dropout_grid = parse_float_list(
                args.mlp_dropout_grid,
                [args.mlp_dropout],
                "--mlp-dropout-grid",
            )
            lr_grid = parse_float_list(
                args.mlp_lr_grid,
                [args.mlp_lr],
                "--mlp-lr-grid",
            )
            weight_decay_grid = parse_float_list(
                args.mlp_weight_decay_grid,
                [args.mlp_weight_decay],
                "--mlp-weight-decay-grid",
            )
            dev_ens, test_ens, fit_info = fit_mlp_crossfit(
                x_train=x_train,
                y_train=y_train,
                x_dev=x_dev,
                x_test=x_test,
                train_mask=train_mask,
                train_choices=train_choices,
                num_classes=args.num_classes,
                cv_folds=args.cv_folds,
                cv_seed=args.cv_seed,
                cv_objective=args.cv_objective,
                hidden_grid=hidden_grid,
                dropout_grid=dropout_grid,
                lr_grid=lr_grid,
                weight_decay_grid=weight_decay_grid,
                epochs=args.mlp_epochs,
                batch_size=args.mlp_batch_size,
                patience=args.mlp_patience,
                device_str=args.device,
                seed=args.seed,
                progress_every=args.mlp_progress_every,
            )
            meta["mlp_info"] = fit_info
        else:
            model, fit_info = fit_mlp(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                hidden_dim=args.mlp_hidden_dim,
                dropout=args.mlp_dropout,
                lr=args.mlp_lr,
                weight_decay=args.mlp_weight_decay,
                epochs=args.mlp_epochs,
                batch_size=args.mlp_batch_size,
                patience=args.mlp_patience,
                device_str=args.device,
                seed=args.seed,
                progress_every=args.mlp_progress_every,
                progress_prefix="[mlp] ",
            )
            dev_ens = predict_mlp(model, x_dev, args.device)
            test_ens = predict_mlp(model, x_test, args.device)
            meta["mlp_info"] = fit_info
            mlp_model = model
        meta["mlp_config"] = {
            "hidden_dim": args.mlp_hidden_dim,
            "dropout": args.mlp_dropout,
            "lr": args.mlp_lr,
            "weight_decay": args.mlp_weight_decay,
            "epochs": args.mlp_epochs,
            "batch_size": args.mlp_batch_size,
            "patience": args.mlp_patience,
            "progress_every": args.mlp_progress_every,
            "seed": args.seed,
            "hidden_grid": args.mlp_hidden_grid,
            "dropout_grid": args.mlp_dropout_grid,
            "lr_grid": args.mlp_lr_grid,
            "weight_decay_grid": args.mlp_weight_decay_grid,
        }

        if mlp_model is not None:
            x_contrib = x_test if args.contrib_split == "test" else x_dev
            contributions_payload = contribution_from_mlp_perturbation(
                mlp_model,
                x_contrib,
                model_names,
                args.device,
                baseline=args.mlp_contrib_baseline,
            )
            contributions_payload["split"] = args.contrib_split
        else:
            if args.show_contributions or args.save_contributions:
                print(
                    "[info] MLP contribution analysis is unavailable in dev-only cross-fit mode "
                    "(no single final MLP model)."
                )

    if contributions_payload is not None:
        meta["contributions"] = contributions_payload
        if args.show_contributions:
            print_contributions("Model contributions:", contributions_payload)
        if args.save_contributions:
            contrib_path = Path(args.save_contributions)
            contrib_path.parent.mkdir(parents=True, exist_ok=True)
            contrib_path.write_text(json.dumps(contributions_payload, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {contrib_path}")

    if args.show_split_metrics:
        print_metrics(
            "Dev ensemble",
            metrics(dev_ens, y_dev, num_classes=args.num_classes, choices=choices_dev),
        )
        print_metrics(
            "Test ensemble",
            metrics(test_ens, y_test, num_classes=args.num_classes, choices=choices_test),
        )
    print_metrics(
        "Overall ensemble",
        metrics_overall(
            [dev_ens, test_ens],
            [y_dev, y_test],
            num_classes=args.num_classes,
            choice_splits=[choices_dev, choices_test],
        ),
    )

    out_dev = Path(args.out_dev)
    out_test = Path(args.out_test)
    write_predictions(out_dev, dev_ens.astype(np.float32))
    write_predictions(out_test, test_ens.astype(np.float32))
    print(f"Wrote {out_dev}")
    print(f"Wrote {out_test}")

    if args.save_meta:
        meta_path = Path(args.save_meta)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
