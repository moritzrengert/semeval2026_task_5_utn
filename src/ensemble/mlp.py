"""Created by Moritz Rengert."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import build_grid, crossfit_grid_search
from losses import coral_expected_value, coral_loss


VALID_MLP_LOSS_MODES = {"mse", "coral", "coral_mse"}


def _validate_mlp_loss_mode(loss_mode: str) -> str:
    mode = str(loss_mode).strip().lower()
    if mode not in VALID_MLP_LOSS_MODES:
        valid = ", ".join(sorted(VALID_MLP_LOSS_MODES))
        raise ValueError(f"Unsupported mlp loss mode '{loss_mode}'. Valid modes: {valid}")
    return mode


def _build_mlp_model(input_dim: int, hidden_dim: int, dropout: float, loss_mode: str, num_classes: int) -> nn.Module:
    out_dim = 1 if loss_mode == "mse" else (num_classes - 1)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def _scores_to_ordinal_labels(scores: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.clamp(torch.floor(scores - 0.5), 0, num_classes - 1).long()


def _scores_from_outputs(outputs: torch.Tensor, loss_mode: str) -> torch.Tensor:
    if loss_mode == "mse":
        return outputs.squeeze(-1)
    return coral_expected_value(outputs) + 1.0


def _compute_objective(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    loss_mode: str,
    num_classes: int,
    mse_loss_fn: nn.Module,
    coral_weight: float,
    mse_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    pred_scores = _scores_from_outputs(outputs, loss_mode)
    mse_term = mse_loss_fn(pred_scores, targets)
    if loss_mode == "mse":
        return mse_term, mse_term, None

    ordinal_targets = _scores_to_ordinal_labels(targets, num_classes=num_classes)
    coral_term = coral_loss(outputs, ordinal_targets, num_classes=num_classes)
    if loss_mode == "coral":
        return coral_term * coral_weight, mse_term, coral_term
    return coral_term * coral_weight + mse_term * mse_weight, mse_term, coral_term


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
    loss_mode: str = "mse",
    num_classes: int = 5,
    coral_weight: float = 1.0,
    mse_weight: float = 1.0,
    progress_every: int = 10,
    progress_prefix: str = "[mlp] ",
):
    loss_mode = _validate_mlp_loss_mode(loss_mode)
    if loss_mode != "mse" and num_classes < 2:
        raise ValueError("num_classes must be >= 2 when using CORAL-based MLP losses")

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
    else:
        x_train_eval_t = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_train_eval_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    model = _build_mlp_model(
        input_dim=x_train.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        loss_mode=loss_mode,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss_fn = nn.MSELoss()

    best_metric = float("inf")
    best_epoch = 1
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        objective_total = 0.0
        mse_total = 0.0
        coral_total = 0.0
        coral_count = 0
        n_items = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            objective, mse_term, coral_term = _compute_objective(
                outputs=outputs,
                targets=yb,
                loss_mode=loss_mode,
                num_classes=num_classes,
                mse_loss_fn=mse_loss_fn,
                coral_weight=coral_weight,
                mse_weight=mse_weight,
            )
            objective.backward()
            optimizer.step()
            batch_n = int(xb.shape[0])
            objective_total += float(objective.item()) * batch_n
            mse_total += float(mse_term.item()) * batch_n
            if coral_term is not None:
                coral_total += float(coral_term.item()) * batch_n
                coral_count += batch_n
            n_items += batch_n
        train_objective = objective_total / max(1, n_items)
        train_mse = mse_total / max(1, n_items)
        train_coral = (coral_total / max(1, coral_count)) if coral_count > 0 else None

        model.eval()
        with torch.no_grad():
            def eval_split(x_eval: torch.Tensor, y_eval: torch.Tensor) -> Tuple[float, float, Optional[float]]:
                outputs_eval = model(x_eval)
                objective_eval, mse_eval, coral_eval = _compute_objective(
                    outputs=outputs_eval,
                    targets=y_eval,
                    loss_mode=loss_mode,
                    num_classes=num_classes,
                    mse_loss_fn=mse_loss_fn,
                    coral_weight=coral_weight,
                    mse_weight=mse_weight,
                )
                return (
                    float(objective_eval.item()),
                    float(mse_eval.item()),
                    float(coral_eval.item()) if coral_eval is not None else None,
                )

            if use_val:
                score, score_mse, score_coral = eval_split(x_val_t, y_val_t)
            else:
                score, score_mse, score_coral = eval_split(x_train_eval_t, y_train_eval_t)

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
            target_name = "val_obj" if use_val else "train_eval_obj"
            improved_text = "yes" if improved else "no"
            parts = [
                f"{progress_prefix}epoch={epoch}/{epochs}",
                f"loss_mode={loss_mode}",
                f"train_obj={train_objective:.6f}",
                f"train_mse={train_mse:.6f}",
            ]
            if train_coral is not None:
                parts.append(f"train_coral={train_coral:.6f}")
            parts.extend(
                [
                    f"{target_name}={score:.6f}",
                    f"{target_name}_mse={score_mse:.6f}",
                ]
            )
            if score_coral is not None:
                parts.append(f"{target_name}_coral={score_coral:.6f}")
            parts.extend(
                [
                    f"best={best_metric:.6f}",
                    f"best_epoch={best_epoch}",
                    f"patience={epochs_no_improve}/{patience}",
                    f"improved={improved_text}",
                ]
            )
            print(" ".join(parts))

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
        "loss_mode": loss_mode,
        "num_classes": int(num_classes),
        "coral_weight": float(coral_weight),
        "mse_weight": float(mse_weight),
    }
    return model, info


def predict_mlp(
    model,
    x: np.ndarray,
    device_str: Optional[str],
    loss_mode: str = "mse",
    num_classes: int = 5,
) -> np.ndarray:
    loss_mode = _validate_mlp_loss_mode(loss_mode)
    if loss_mode != "mse" and num_classes < 2:
        raise ValueError("num_classes must be >= 2 when using CORAL-based MLP losses")

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval().to(device)
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        outputs = model(x_t)
        preds = _scores_from_outputs(outputs, loss_mode).detach().cpu().numpy()
    return preds.astype(np.float32)


def contribution_from_mlp_perturbation(
    model,
    x: np.ndarray,
    model_names: Sequence[str],
    device_str: Optional[str],
    baseline: str = "mean",
    loss_mode: str = "mse",
    num_classes: int = 5,
) -> Dict[str, Any]:
    if baseline not in {"mean", "median"}:
        raise ValueError("baseline must be one of: mean, median")

    if baseline == "mean":
        base_vec = np.mean(x, axis=0)
    else:
        base_vec = np.median(x, axis=0)

    base_pred = predict_mlp(model, x, device_str, loss_mode=loss_mode, num_classes=num_classes).astype(np.float64)

    deltas_signed: List[float] = []
    deltas_abs: List[float] = []
    for feat_idx in range(x.shape[1]):
        x_pert = x.copy()
        x_pert[:, feat_idx] = float(base_vec[feat_idx])
        pert_pred = predict_mlp(
            model,
            x_pert,
            device_str,
            loss_mode=loss_mode,
            num_classes=num_classes,
        ).astype(np.float64)
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
    loss_mode: str,
    coral_weight: float,
    mse_weight: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    loss_mode = _validate_mlp_loss_mode(loss_mode)
    if loss_mode != "mse" and num_classes < 2:
        raise ValueError("num_classes must be >= 2 when using CORAL-based MLP losses")

    configs = build_grid(
        {
            "hidden_dim": hidden_grid,
            "dropout": dropout_grid,
            "lr": lr_grid,
            "weight_decay": weight_decay_grid,
        }
    )

    def fit_fold(cfg, fold_idx, x_tr, y_tr, x_va, y_va):
        fold_prefix = (
            "[mlp-cv] "
            f"cfg(loss={loss_mode},h={int(cfg['hidden_dim'])},d={float(cfg['dropout'])},"
            f"lr={float(cfg['lr'])},wd={float(cfg['weight_decay'])}) "
            f"fold={fold_idx + 1}/{cv_folds} "
        )
        model, info = fit_mlp(
            x_train=x_tr,
            y_train=y_tr,
            x_val=x_va,
            y_val=y_va,
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg["dropout"]),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            epochs=int(epochs),
            batch_size=int(batch_size),
            patience=int(patience),
            device_str=device_str,
            seed=int(seed + fold_idx),
            loss_mode=loss_mode,
            num_classes=num_classes,
            coral_weight=coral_weight,
            mse_weight=mse_weight,
            progress_every=progress_every,
            progress_prefix=fold_prefix,
        )
        return model, info, {}

    def predict_fold(model, x, _cfg):
        return predict_mlp(
            model,
            x,
            device_str,
            loss_mode=loss_mode,
            num_classes=num_classes,
        )

    def log_params(cfg):
        return {
            "loss_mode": loss_mode,
            "coral_weight": float(coral_weight),
            "mse_weight": float(mse_weight),
            "hidden_dim": int(cfg["hidden_dim"]),
            "dropout": float(cfg["dropout"]),
            "lr": float(cfg["lr"]),
            "weight_decay": float(cfg["weight_decay"]),
        }

    def on_config_start(cfg):
        if progress_every > 0:
            print(
                "[mlp-cv] "
                f"config loss_mode={loss_mode} coral_weight={float(coral_weight)} mse_weight={float(mse_weight)} "
                f"hidden_dim={int(cfg['hidden_dim'])} dropout={float(cfg['dropout'])} "
                f"lr={float(cfg['lr'])} weight_decay={float(cfg['weight_decay'])}"
            )

    def on_config_end(_cfg, objective, best):
        if progress_every > 0:
            print(
                "[mlp-cv] "
                f"config-result objective({cv_objective})={float(objective):.6f} "
                f"best_so_far={float(best):.6f}"
            )

    dev_preds, test_preds, result = crossfit_grid_search(
        x_train=x_train,
        y_train=y_train,
        x_dev=x_dev,
        x_test=x_test,
        train_mask=train_mask,
        train_choices=train_choices,
        num_classes=num_classes,
        cv_folds=cv_folds,
        cv_seed=cv_seed,
        cv_objective=cv_objective,
        configs=configs,
        fit_fold=fit_fold,
        predict_fold=predict_fold,
        log_params=log_params,
        on_config_start=on_config_start,
        on_config_end=on_config_end,
    )

    selected = result["selected_config"]
    best_epochs = [int(item["best_epoch"]) for item in result["fold_info"]]
    info = {
        "source": "crossfit",
        "cv_folds": int(result["cv_folds"]),
        "cv_seed": int(result["cv_seed"]),
        "cv_objective": result["cv_objective"],
        "loss_mode": loss_mode,
        "num_classes": int(num_classes),
        "coral_weight": float(coral_weight),
        "mse_weight": float(mse_weight),
        "selected": {
            "loss_mode": loss_mode,
            "coral_weight": float(coral_weight),
            "mse_weight": float(mse_weight),
            "hidden_dim": int(selected["hidden_dim"]),
            "dropout": float(selected["dropout"]),
            "lr": float(selected["lr"]),
            "weight_decay": float(selected["weight_decay"]),
            "objective": float(result["selected_objective"]),
            "metrics": result["selected_metrics"],
        },
        "mean_best_epoch": float(np.mean(best_epochs)),
        "fold_best_epochs": best_epochs,
        "search_results": result["search_results"],
        "uses_oof_dev_predictions": bool(result["uses_oof_dev_predictions"]),
        "fold_count": int(result["fold_count"]),
    }
    return dev_preds, test_preds, info
