"""
Ensemble precomputed prediction JSON files.
Created by Moritz Rengert.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from average import combine_average
from common import *
from mlp import *
from weighted import *


class PreparedData(NamedTuple):
    x_dev: np.ndarray
    x_test: np.ndarray
    y_dev: np.ndarray
    y_test: np.ndarray
    choices_dev: List[Optional[np.ndarray]]
    choices_test: List[Optional[np.ndarray]]
    model_names: List[str]
    n_models: int
    train_mask: np.ndarray
    has_val: bool
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: Optional[np.ndarray]
    y_val: Optional[np.ndarray]
    train_choices: List[Optional[np.ndarray]]


def print_contributions(title: str, contribution: Dict[str, Any]) -> None:
    print(title)
    print(f"  method={contribution.get('method', 'unknown')} n={contribution.get('n_samples', 'n/a')}")
    for item in contribution.get("by_model", []):
        name = item.get("model", "unknown")
        if "weight" in item:
            print(
                f"  {name}: weight={item['weight']:.6f} "
                f"mean_signed={item['mean_signed_contribution']:.6f} "
                f"mean_abs={item['mean_abs_contribution']:.6f} share={item['abs_share']:.4f}"
            )
        else:
            print(
                f"  {name}: mean_signed_delta={item['mean_signed_delta']:.6f} "
                f"mean_abs_delta={item['mean_abs_delta']:.6f} share={item['abs_share']:.4f}"
            )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ensemble prediction JSONs from multiple models.")
    add = ap.add_argument

    arg_specs = [
        (("--dev-preds",), dict(nargs="+", required=True, help="Dev prediction JSON files (one per model).")),
        (("--test-preds",), dict(nargs="+", required=True, help="Test prediction JSON files (same order as --dev-preds).")),
        (("--model-names",), dict(nargs="+", default=None, help="Optional model names in the same order.")),
        (("--dev-labels-path",), dict(type=str, default="semeval26-05-scripts/data/dev.json")),
        (("--test-labels-path",), dict(type=str, default="semeval26-05-scripts/data/test.json")),
        (("--label-key",), dict(type=str, default="average")),
        (("--num-classes",), dict(type=int, default=5)),
        (("--combine",), dict(choices=["average", "weighted", "mlp"], default="average")),
        (("--weights",), dict(type=str, default=None, help="Comma-separated weights for weighted mode.")),
        (("--out-dev",), dict(type=str, default="predictions/dev_preds_ensemble.json")),
        (("--out-test",), dict(type=str, default="predictions/test_preds_ensemble.json")),
        (("--save-meta",), dict(type=str, default=None, help="Optional path to save method metadata (JSON).")),
        (("--weight-epochs",), dict(type=int, default=1500)),
        (("--weight-lr",), dict(type=float, default=0.05)),
        (("--weight-l2",), dict(type=float, default=1e-4)),
        (("--weight-epochs-grid",), dict(type=str, default=None)),
        (("--weight-lr-grid",), dict(type=str, default=None)),
        (("--weight-l2-grid",), dict(type=str, default=None)),
        (("--mlp-hidden-dim",), dict(type=int, default=16)),
        (("--mlp-dropout",), dict(type=float, default=0.1)),
        (("--mlp-lr",), dict(type=float, default=1e-3)),
        (("--mlp-weight-decay",), dict(type=float, default=1e-4)),
        (("--mlp-loss",), dict(choices=["mse", "coral", "coral_mse"], default="mse")),
        (("--mlp-coral-weight",), dict(type=float, default=1.0)),
        (("--mlp-mse-weight",), dict(type=float, default=1.0)),
        (("--mlp-epochs",), dict(type=int, default=400)),
        (("--mlp-batch-size",), dict(type=int, default=64)),
        (("--mlp-patience",), dict(type=int, default=40)),
        (("--mlp-progress-every",), dict(type=int, default=10, help="Print MLP progress every N epochs (<=0 disables).")),
        (("--mlp-hidden-grid",), dict(type=str, default=None)),
        (("--mlp-dropout-grid",), dict(type=str, default=None)),
        (("--mlp-lr-grid",), dict(type=str, default=None)),
        (("--mlp-weight-decay-grid",), dict(type=str, default=None)),
        (("--cv-folds",), dict(type=int, default=5)),
        (("--cv-seed",), dict(type=int, default=42)),
        (("--cv-objective",), dict(choices=["mse", "mae", "spearman", "accuracy", "acc_within_sd"], default="spearman")),
        (("--force-dev-cv",), dict(action="store_true")),
        (("--device",), dict(type=str, default=None, help="Device for MLP mode (e.g. cpu, cuda:0).")),
        (("--seed",), dict(type=int, default=42)),
        (("--show-model-metrics",), dict(action="store_true")),
        (("--show-split-metrics",), dict(action="store_true")),
        (("--show-weights",), dict(action="store_true")),
        (("--show-contributions",), dict(action="store_true")),
        (("--save-contributions",), dict(type=str, default=None)),
        (("--contrib-split",), dict(choices=["dev", "test"], default="test")),
        (("--mlp-contrib-baseline",), dict(choices=["mean", "median"], default="mean")),
    ]
    for flags, kwargs in arg_specs:
        add(*flags, **kwargs)
    return ap.parse_args()


def prepare_data(args: argparse.Namespace) -> PreparedData:
    if len(args.dev_preds) != len(args.test_preds):
        raise SystemExit("--dev-preds and --test-preds must contain the same number of files")

    x_dev = stack_prediction_files(args.dev_preds, "dev")
    x_test = stack_prediction_files(args.test_preds, "test")
    if x_dev.shape[1] != x_test.shape[1]:
        raise SystemExit("Dev/test model count mismatch")

    model_names = args.model_names or [derive_model_name(p) for p in args.dev_preds]
    if len(model_names) != x_dev.shape[1]:
        raise SystemExit(f"Expected {x_dev.shape[1]} model names, got {len(model_names)}")

    dev_label_path = Path(args.dev_labels_path)
    test_label_path = Path(args.test_labels_path)
    y_dev = align_labels(load_labels(dev_label_path, args.label_key), x_dev.shape[0], "dev", dev_label_path)
    y_test = align_labels(load_labels(test_label_path, args.label_key), x_test.shape[0], "test", test_label_path)
    choices_dev = align_choices(load_choices(dev_label_path), x_dev.shape[0], "dev", dev_label_path)
    choices_test = align_choices(load_choices(test_label_path), x_test.shape[0], "test", test_label_path)

    train_mask = np.isfinite(y_dev)
    if args.combine in {"weighted", "mlp"} and not np.any(train_mask):
        raise SystemExit("No numeric dev labels available for training combiner.")

    has_val = bool(np.any(np.isfinite(y_test)))
    if args.combine in {"weighted", "mlp"} and not has_val:
        print("[info] Test labels are non-numeric or missing.")

    x_train = x_dev[train_mask]
    y_train = y_dev[train_mask]
    val_mask = np.isfinite(y_test)
    train_choices = [c for c, keep in zip(choices_dev, train_mask.tolist()) if keep]
    return PreparedData(
        x_dev=x_dev,
        x_test=x_test,
        y_dev=y_dev,
        y_test=y_test,
        choices_dev=choices_dev,
        choices_test=choices_test,
        model_names=list(model_names),
        n_models=int(x_dev.shape[1]),
        train_mask=train_mask,
        has_val=has_val,
        x_train=x_train,
        y_train=y_train,
        x_val=x_test[val_mask] if has_val else None,
        y_val=y_test[val_mask] if has_val else None,
        train_choices=train_choices,
    )


def run_weighted(args: argparse.Namespace, data: PreparedData, use_dev_cv: bool) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    if args.weights:
        weights = parse_weights(args.weights, data.n_models)
        info: Dict[str, Any] = {"source": "manual", "used_validation": False}
        dev_preds, test_preds = data.x_dev @ weights, data.x_test @ weights
    elif use_dev_cv:
        dev_preds, test_preds, info = fit_weighted_crossfit(
            x_train=data.x_train,
            y_train=data.y_train,
            x_dev=data.x_dev,
            x_test=data.x_test,
            train_mask=data.train_mask,
            train_choices=data.train_choices,
            num_classes=args.num_classes,
            cv_folds=args.cv_folds,
            cv_seed=args.cv_seed,
            cv_objective=args.cv_objective,
            epochs_grid=parse_int_list(args.weight_epochs_grid, [args.weight_epochs], "--weight-epochs-grid"),
            lr_grid=parse_float_list(args.weight_lr_grid, [args.weight_lr], "--weight-lr-grid"),
            l2_grid=parse_float_list(args.weight_l2_grid, [args.weight_l2], "--weight-l2-grid"),
        )
        weights = np.array(info["weights_mean"], dtype=np.float32)
    else:
        weights, fit_info = fit_weighted_average(
            x_train=data.x_train,
            y_train=data.y_train,
            x_val=data.x_val,
            y_val=data.y_val,
            epochs=args.weight_epochs,
            lr=args.weight_lr,
            l2=args.weight_l2,
        )
        info = {"source": "learned", **fit_info}
        dev_preds, test_preds = data.x_dev @ weights, data.x_test @ weights

    if args.show_weights:
        print("Weighted coefficients:")
        for name, value in zip(data.model_names, weights.tolist()):
            print(f"  {name}: {value:.6f}")

    contrib = contribution_from_weighted(data.x_test if args.contrib_split == "test" else data.x_dev, weights, data.model_names)
    contrib["split"] = args.contrib_split
    return dev_preds, test_preds, {"weights": [float(w) for w in weights.tolist()], "weighted_info": info}, contrib


def run_mlp(args: argparse.Namespace, data: PreparedData, use_dev_cv: bool) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Optional[Dict[str, Any]]]:
    model = None
    if use_dev_cv:
        dev_preds, test_preds, fit_info = fit_mlp_crossfit(
            x_train=data.x_train,
            y_train=data.y_train,
            x_dev=data.x_dev,
            x_test=data.x_test,
            train_mask=data.train_mask,
            train_choices=data.train_choices,
            num_classes=args.num_classes,
            cv_folds=args.cv_folds,
            cv_seed=args.cv_seed,
            cv_objective=args.cv_objective,
            hidden_grid=parse_int_list(args.mlp_hidden_grid, [args.mlp_hidden_dim], "--mlp-hidden-grid"),
            dropout_grid=parse_float_list(args.mlp_dropout_grid, [args.mlp_dropout], "--mlp-dropout-grid"),
            lr_grid=parse_float_list(args.mlp_lr_grid, [args.mlp_lr], "--mlp-lr-grid"),
            weight_decay_grid=parse_float_list(args.mlp_weight_decay_grid, [args.mlp_weight_decay], "--mlp-weight-decay-grid"),
            epochs=args.mlp_epochs,
            batch_size=args.mlp_batch_size,
            patience=args.mlp_patience,
            device_str=args.device,
            seed=args.seed,
            progress_every=args.mlp_progress_every,
            loss_mode=args.mlp_loss,
            coral_weight=args.mlp_coral_weight,
            mse_weight=args.mlp_mse_weight,
        )
    else:
        model, fit_info = fit_mlp(
            x_train=data.x_train,
            y_train=data.y_train,
            x_val=data.x_val,
            y_val=data.y_val,
            hidden_dim=args.mlp_hidden_dim,
            dropout=args.mlp_dropout,
            lr=args.mlp_lr,
            weight_decay=args.mlp_weight_decay,
            epochs=args.mlp_epochs,
            batch_size=args.mlp_batch_size,
            patience=args.mlp_patience,
            device_str=args.device,
            seed=args.seed,
            loss_mode=args.mlp_loss,
            num_classes=args.num_classes,
            coral_weight=args.mlp_coral_weight,
            mse_weight=args.mlp_mse_weight,
            progress_every=args.mlp_progress_every,
            progress_prefix="[mlp] ",
        )
        dev_preds, test_preds = (
            predict_mlp(model, data.x_dev, args.device, loss_mode=args.mlp_loss, num_classes=args.num_classes),
            predict_mlp(model, data.x_test, args.device, loss_mode=args.mlp_loss, num_classes=args.num_classes),
        )

    meta = {
        "mlp_info": fit_info,
        "mlp_config": {
            "hidden_dim": args.mlp_hidden_dim,
            "dropout": args.mlp_dropout,
            "lr": args.mlp_lr,
            "weight_decay": args.mlp_weight_decay,
            "loss": args.mlp_loss,
            "coral_weight": args.mlp_coral_weight,
            "mse_weight": args.mlp_mse_weight,
            "epochs": args.mlp_epochs,
            "batch_size": args.mlp_batch_size,
            "patience": args.mlp_patience,
            "progress_every": args.mlp_progress_every,
            "seed": args.seed,
            "hidden_grid": args.mlp_hidden_grid,
            "dropout_grid": args.mlp_dropout_grid,
            "lr_grid": args.mlp_lr_grid,
            "weight_decay_grid": args.mlp_weight_decay_grid,
        },
    }

    if model is None:
        if args.show_contributions or args.save_contributions:
            print("[info] MLP contribution analysis is unavailable in dev-only cross-fit mode (no single final MLP model).")
        return dev_preds, test_preds, meta, None

    contrib = contribution_from_mlp_perturbation(
        model,
        data.x_test if args.contrib_split == "test" else data.x_dev,
        data.model_names,
        args.device,
        baseline=args.mlp_contrib_baseline,
        loss_mode=args.mlp_loss,
        num_classes=args.num_classes,
    )
    contrib["split"] = args.contrib_split
    return dev_preds, test_preds, meta, contrib


def main() -> None:
    args = parse_args()
    data = prepare_data(args)

    cv_eligible = args.combine == "mlp" or (args.combine == "weighted" and not args.weights)
    use_dev_cv = bool(cv_eligible and (args.force_dev_cv or (not data.has_val and args.cv_folds > 1)))
    if cv_eligible and use_dev_cv:
        print(f"[info] Using dev-only {args.cv_folds}-fold cross-fit (objective={args.cv_objective}) for {args.combine}.")
    elif cv_eligible and not data.has_val:
        print("[info] No numeric test labels and --cv-folds <= 1; combiner will be selected on dev train loss only.")
    elif cv_eligible and data.has_val:
        print(f"[info] Using test labels for validation for {args.combine} (dev is training split).")

    if args.show_model_metrics:
        for idx, name in enumerate(data.model_names):
            print_metrics(f"Dev {name}", metrics(data.x_dev[:, idx], data.y_dev, num_classes=args.num_classes, choices=data.choices_dev))
            print_metrics(f"Test {name}", metrics(data.x_test[:, idx], data.y_test, num_classes=args.num_classes, choices=data.choices_test))

    meta: Dict[str, Any] = {
        "combine": args.combine,
        "model_names": data.model_names,
        "dev_pred_files": args.dev_preds,
        "test_pred_files": args.test_preds,
    }

    if args.combine == "average":
        dev_preds, test_preds, updates = combine_average(data.x_dev, data.x_test, data.n_models)
        meta.update(updates)
        contrib = None
    elif args.combine == "weighted":
        dev_preds, test_preds, updates, contrib = run_weighted(args, data, use_dev_cv)
        meta.update(updates)
    else:
        dev_preds, test_preds, updates, contrib = run_mlp(args, data, use_dev_cv)
        meta.update(updates)

    if contrib is not None:
        meta["contributions"] = contrib
        if args.show_contributions:
            print_contributions("Model contributions:", contrib)
        if args.save_contributions:
            contrib_path = Path(args.save_contributions)
            contrib_path.parent.mkdir(parents=True, exist_ok=True)
            contrib_path.write_text(json.dumps(contrib, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {contrib_path}")

    if args.show_split_metrics:
        print_metrics("Dev ensemble", metrics(dev_preds, data.y_dev, num_classes=args.num_classes, choices=data.choices_dev))
        print_metrics("Test ensemble", metrics(test_preds, data.y_test, num_classes=args.num_classes, choices=data.choices_test))
    print_metrics(
        "Overall ensemble",
        metrics_overall(
            [dev_preds, test_preds],
            [data.y_dev, data.y_test],
            num_classes=args.num_classes,
            choice_splits=[data.choices_dev, data.choices_test],
        ),
    )

    out_dev, out_test = Path(args.out_dev), Path(args.out_test)
    write_predictions(out_dev, dev_preds.astype(np.float32))
    write_predictions(out_test, test_preds.astype(np.float32))
    print(f"Wrote {out_dev}")
    print(f"Wrote {out_test}")

    if args.save_meta:
        meta_path = Path(args.save_meta)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
