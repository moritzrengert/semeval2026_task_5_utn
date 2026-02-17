"""Evaluation metrics."""

import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from scipy.stats import spearmanr, pearsonr


def compute_metrics(eval_pred, label_min=1.0, label_max=5.0):
    """Compute MAE, RMSE, Spearman, Pearson, and rounded accuracy for predictions."""
    # Extract predictions and labels from EvalPrediction object
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    # Handle various output formats (tuple, dict, or array)
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    elif isinstance(preds, dict):
        preds = preds.get("logits", preds)
    preds = np.squeeze(preds)
    labels = np.squeeze(labels)
    # Store unclipped predictions for correlation metrics
    preds_before_clip = preds.copy()
    pred_std = np.std(preds)
    # Clip predictions to valid range [1, 5]
    preds = np.clip(preds, label_min, label_max)
    # Regression metrics
    mae = mean_absolute_error(labels, preds)
    rmse = math.sqrt(mean_squared_error(labels, preds))
    # Correlation metrics (guard against constant predictions)
    if pred_std > 1e-8:
        sp = spearmanr(labels, preds_before_clip).correlation
        pr = pearsonr(labels, preds_before_clip)[0] if np.std(labels) > 1e-8 else 0.0
    else:
        sp = pr = 0.0
    # Classification accuracy (rounded to nearest integer)
    rounded = np.rint(preds).astype(int)
    rounded = np.clip(rounded, int(label_min), int(label_max))
    rounded_acc = accuracy_score(np.rint(labels).astype(int), rounded)
    return {"mae": mae, "rmse": rmse, "spearman": sp if not np.isnan(sp) else 0.0, "pearson": pr if not np.isnan(pr) else 0.0, "rounded_acc": rounded_acc}
