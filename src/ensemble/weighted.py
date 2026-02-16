"""Created by Moritz Rengert."""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from common import build_grid, crossfit_grid_search, softmax


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

        grad_w = (2.0 / x_train.shape[0]) * (x_train.T @ err) + 2.0 * l2 * w
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