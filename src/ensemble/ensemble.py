"""
Ensemble precomputed prediction JSON files.
Created by Moritz Rengert.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from average import combine_average
from common import *
from linear_regression import *
from mlp import contribution_from_mlp_perturbation, fit_mlp, predict_mlp
from weighted import contribution_from_weighted, fit_weighted_average


BEST_WEIGHTED = {"epochs": 1800, "lr": 0.05, "l2": 0.001}
BEST_LINEAR = {"fit_intercept": True}
BEST_MLP = {
    "hidden_dim": 16,
    "dropout": 0.1,
    "lr": 5e-4,
    "weight_decay": 1e-3,
    "loss": "mse",
    "coral_weight": 1.0,
    "mse_weight": 1.0,
    "epochs": 350,
    "batch_size": 64,
    "patience": 40,
    "progress_every": 25,
}

DEFAULT_DEV_PREDS = [
    "predictions/dev_preds_ares.json",
    "predictions/dev_preds_lmms.json",
    "predictions/dev_preds_nli.json",
    "predictions/dev_preds_sbert.json",
    "predictions/dev_preds_sensembert.json",
    "predictions/dev_preds_stsbert.json",
]
DEFAULT_TEST_PREDS = [
    "predictions/test_preds_ares.json",
    "predictions/test_preds_lmms.json",
    "predictions/test_preds_nli.json",
    "predictions/test_preds_sbert.json",
    "predictions/test_preds_sensembert.json",
    "predictions/test_preds_stsbert.json",
]

DEV_LABELS_PATH = Path("semeval26-05-scripts/data/dev.json")
TEST_LABELS_PATH = Path("semeval26-05-scripts/data/test.json")
LABEL_KEY = "average"
NUM_CLASSES = 5
SEED = 42
DEVICE = None
CONTRIB_SPLIT = "test"
MLP_CONTRIB_BASELINE = "mean"


def print_contributions(contrib: dict) -> None:
    print(f"Model contributions: method={contrib.get('method', 'unknown')} n={contrib.get('n_samples', 'n/a')}")
    for item in contrib.get("by_model", []):
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


def prepare_data(dev_preds: list, test_preds: list, combine: str) -> dict:
    if len(dev_preds) != len(test_preds):
        raise SystemExit("--dev-preds and --test-preds must contain the same number of files")

    x_dev = stack_prediction_files(dev_preds, "dev")
    x_test = stack_prediction_files(test_preds, "test")
    if x_dev.shape[1] != x_test.shape[1]:
        raise SystemExit("Dev/test model count mismatch")

    model_names = [derive_model_name(path) for path in dev_preds]
    y_dev = align_labels(load_labels(DEV_LABELS_PATH, LABEL_KEY), x_dev.shape[0], "dev", DEV_LABELS_PATH)
    y_test = align_labels(load_labels(TEST_LABELS_PATH, LABEL_KEY), x_test.shape[0], "test", TEST_LABELS_PATH)
    choices_dev = align_choices(load_choices(DEV_LABELS_PATH), x_dev.shape[0], "dev", DEV_LABELS_PATH)
    choices_test = align_choices(load_choices(TEST_LABELS_PATH), x_test.shape[0], "test", TEST_LABELS_PATH)

    train_mask = np.isfinite(y_dev)
    if combine != "average" and not np.any(train_mask):
        raise SystemExit("No numeric dev labels available for training combiner.")

    val_mask = np.isfinite(y_test)
    if combine != "average" and not np.any(val_mask):
        print("[info] Test labels are non-numeric or missing.")

    return {
        "x_dev": x_dev,
        "x_test": x_test,
        "y_dev": y_dev,
        "y_test": y_test,
        "choices_dev": choices_dev,
        "choices_test": choices_test,
        "model_names": model_names,
        "x_train": x_dev[train_mask],
        "y_train": y_dev[train_mask],
        "x_val": x_test[val_mask] if np.any(val_mask) else None,
        "y_val": y_test[val_mask] if np.any(val_mask) else None,
    }


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--combine", choices=["average", "weighted", "linear_regression", "mlp"], default="mlp")
    args.add_argument("--dev-preds", nargs="+", default=list(DEFAULT_DEV_PREDS))
    args.add_argument("--test-preds", nargs="+", default=list(DEFAULT_TEST_PREDS))
    args.add_argument("--show-contributions", action="store_true")
    args = args.parse_args()

    data = prepare_data(args.dev_preds, args.test_preds, combine=args.combine)

    meta = {
        "combine": args.combine,
        "model_names": data["model_names"],
        "dev_pred_files": args.dev_preds,
        "test_pred_files": args.test_preds,
        "fixed_best_configs": {
            "weighted": BEST_WEIGHTED,
            "linear_regression": BEST_LINEAR,
            "mlp": BEST_MLP,
        },
    }

    if args.combine == "average":
        print("[info] Using fixed best average config (uniform mean over models).")
        dev_preds, test_preds, updates = combine_average(data["x_dev"], data["x_test"], data["x_dev"].shape[1])
        meta.update(updates)
        contrib = None
    elif args.combine == "weighted":
        print(f"[info] Using fixed best weighted config: {BEST_WEIGHTED}")
        weights, fit_info = fit_weighted_average(
            x_train=data["x_train"],
            y_train=data["y_train"],
            x_val=data["x_val"],
            y_val=data["y_val"],
            epochs=int(BEST_WEIGHTED["epochs"]),
            lr=float(BEST_WEIGHTED["lr"]),
            l2=float(BEST_WEIGHTED["l2"]),
        )
        dev_preds = data["x_dev"] @ weights
        test_preds = data["x_test"] @ weights
        contrib = contribution_from_weighted(
            data["x_test"] if CONTRIB_SPLIT == "test" else data["x_dev"],
            weights,
            data["model_names"],
        )
        contrib["split"] = CONTRIB_SPLIT
        meta.update(
            {
                "weights": [float(w) for w in weights.tolist()],
                "weighted_info": {"source": "fixed_best", "fixed_config": dict(BEST_WEIGHTED), **fit_info},
            }
        )
    elif args.combine == "linear_regression":
        print(f"[info] Using fixed best linear regression config: {BEST_LINEAR}")
        model, fit_info = fit_linear_regression(
            x_train=data["x_train"],
            y_train=data["y_train"],
            fit_intercept=bool(BEST_LINEAR["fit_intercept"]),
        )
        dev_preds = predict_linear_regression(model, data["x_dev"])
        test_preds = predict_linear_regression(model, data["x_test"])
        coefficients = np.asarray(model.coef_, dtype=np.float32).reshape(-1)
        intercept = float(np.asarray(model.intercept_, dtype=np.float64).reshape(-1)[0])
        contrib = contribution_from_linear_regression(
            data["x_test"] if CONTRIB_SPLIT == "test" else data["x_dev"],
            coefficients,
            data["model_names"],
            intercept=intercept,
        )
        contrib["split"] = CONTRIB_SPLIT
        meta.update(
            {
                "weights": [float(v) for v in coefficients.tolist()],
                "intercept": intercept,
                "linear_regression_info": {"source": "fixed_best", "fixed_config": dict(BEST_LINEAR), **fit_info},
            }
        )
    else:
        print(f"[info] Using fixed best MLP config: {BEST_MLP}")
        model, fit_info = fit_mlp(
            x_train=data["x_train"],
            y_train=data["y_train"],
            x_val=data["x_val"],
            y_val=data["y_val"],
            hidden_dim=int(BEST_MLP["hidden_dim"]),
            dropout=float(BEST_MLP["dropout"]),
            lr=float(BEST_MLP["lr"]),
            weight_decay=float(BEST_MLP["weight_decay"]),
            epochs=int(BEST_MLP["epochs"]),
            batch_size=int(BEST_MLP["batch_size"]),
            patience=int(BEST_MLP["patience"]),
            device_str=DEVICE,
            seed=SEED,
            loss_mode=str(BEST_MLP["loss"]),
            num_classes=NUM_CLASSES,
            coral_weight=float(BEST_MLP["coral_weight"]),
            mse_weight=float(BEST_MLP["mse_weight"]),
            progress_every=int(BEST_MLP["progress_every"]),
            progress_prefix="[mlp-best] ",
        )
        dev_preds = predict_mlp(model, data["x_dev"], DEVICE, loss_mode=str(BEST_MLP["loss"]), num_classes=NUM_CLASSES)
        test_preds = predict_mlp(model, data["x_test"], DEVICE, loss_mode=str(BEST_MLP["loss"]), num_classes=NUM_CLASSES)
        contrib = contribution_from_mlp_perturbation(
            model,
            data["x_test"] if CONTRIB_SPLIT == "test" else data["x_dev"],
            data["model_names"],
            DEVICE,
            baseline=MLP_CONTRIB_BASELINE,
            loss_mode=str(BEST_MLP["loss"]),
            num_classes=NUM_CLASSES,
        )
        contrib["split"] = CONTRIB_SPLIT
        meta.update(
            {
                "mlp_info": {"source": "fixed_best", "fixed_config": dict(BEST_MLP), **fit_info},
                "mlp_config": dict(BEST_MLP),
            }
        )

    if contrib is not None:
        meta["contributions"] = contrib
        if args.show_contributions:
            print_contributions(contrib)
            contrib_path = Path(f"predictions/contrib_{args.combine}.json")
            contrib_path.parent.mkdir(parents=True, exist_ok=True)
            contrib_path.write_text(json.dumps(contrib, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {contrib_path}")

    print_metrics("Dev ensemble", metrics(dev_preds, data["y_dev"], num_classes=NUM_CLASSES, choices=data["choices_dev"]))
    print_metrics("Test ensemble", metrics(test_preds, data["y_test"], num_classes=NUM_CLASSES, choices=data["choices_test"]))
    print_metrics(
        "Overall ensemble",
        metrics_overall(
            [dev_preds, test_preds],
            [data["y_dev"], data["y_test"]],
            num_classes=NUM_CLASSES,
            choice_splits=[data["choices_dev"], data["choices_test"]],
        ),
    )

    out_dev = Path(f"predictions/dev_preds_ensemble_{args.combine}.json")
    out_test = Path(f"predictions/test_preds_ensemble_{args.combine}.json")
    meta_path = Path(f"predictions/meta_{args.combine}.json")
    write_predictions(out_dev, dev_preds.astype(np.float32))
    write_predictions(out_test, test_preds.astype(np.float32))
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_dev}")
    print(f"Wrote {out_test}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
