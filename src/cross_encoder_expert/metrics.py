"""Eval metrics (same as cross_encoder_adam).
Created by Adam Jen Khai Lo."""

import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from scipy.stats import spearmanr, pearsonr


def compute_metrics(eval_pred, label_min=1.0, label_max=5.0):
    preds, labels = eval_pred
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    elif isinstance(preds, dict):
        preds = preds.get("logits", preds)
    preds = np.squeeze(preds)
    labels = np.squeeze(labels)
    preds_before_clip = preds.copy()
    pred_std = np.std(preds)
    preds = np.clip(preds, label_min, label_max)
    mae = mean_absolute_error(labels, preds)
    rmse = math.sqrt(mean_squared_error(labels, preds))
    if pred_std > 1e-8:
        sp = spearmanr(labels, preds_before_clip).correlation
        pr = pearsonr(labels, preds_before_clip)[0] if np.std(labels) > 1e-8 else 0.0
    else:
        sp = pr = 0.0
    rounded = np.rint(preds).astype(int)
    rounded = np.clip(rounded, int(label_min), int(label_max))
    rounded_acc = accuracy_score(np.rint(labels).astype(int), rounded)
    return {"mae": mae, "rmse": rmse, "spearman": sp if not np.isnan(sp) else 0.0, "pearson": pr if not np.isnan(pr) else 0.0, "rounded_acc": rounded_acc}
