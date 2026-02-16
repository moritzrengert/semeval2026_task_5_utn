"""Created by Noas Shaalan."""

from typing import Any, Dict, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression


def fit_linear_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    fit_intercept: bool = True,
) -> Tuple[LinearRegression, Dict[str, Any]]:
    """ Fits a linear regression model to the training data."""
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(x_train, y_train)
    preds = np.asarray(model.predict(x_train), dtype=np.float64).reshape(-1)
    train_mse = float(np.mean((preds - y_train.astype(np.float64)) ** 2))
    info = {
        "fit_intercept": bool(fit_intercept),
        "train_mse": float(train_mse),
        "n_train": int(x_train.shape[0]),
    }
    return model, info


def predict_linear_regression(model: LinearRegression, x: np.ndarray) -> np.ndarray:
    """ Predicts using the linear regression model."""
    return np.asarray(model.predict(x), dtype=np.float32).reshape(-1)


def contribution_from_linear_regression(
    x: np.ndarray,
    coefficients: np.ndarray,
    model_names: Sequence[str],
    intercept: float,
) -> Dict[str, Any]:
    """ Computes the contribution of each model to the final prediction."""
    contrib = x * coefficients.reshape(1, -1)
    mean_signed = np.mean(contrib, axis=0)
    mean_abs = np.mean(np.abs(contrib), axis=0)
    total_abs = float(np.sum(mean_abs))
    if total_abs <= 0:
        share = np.zeros_like(mean_abs)
    else:
        share = mean_abs / total_abs

    by_model = []
    for name, coef, ms, ma, sh in zip(
        model_names,
        coefficients.tolist(),
        mean_signed.tolist(),
        mean_abs.tolist(),
        share.tolist(),
    ):
        by_model.append(
            {
                "model": str(name),
                "weight": float(coef),
                "mean_signed_contribution": float(ms),
                "mean_abs_contribution": float(ma),
                "abs_share": float(sh),
            }
        )

    return {
        "method": "linear_regression",
        "intercept": float(intercept),
        "n_samples": int(x.shape[0]),
        "by_model": by_model,
    }
