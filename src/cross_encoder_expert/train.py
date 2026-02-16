"""
Train cross-encoder expert. Same data format/order as other experts (uses Moritz load_dataset).
Run from repo root: python src/cross_encoder_expert/train.py --train-path ... --dev-path ... --save-path ...
Created by Adam Jen Khai Lo.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, set_seed

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cross_encoder_expert import config as default_config
from cross_encoder_expert.data_utils import load_and_process_data, tokenize_cross_encoder
from cross_encoder_expert.model import CrossEncoderRegressor
from cross_encoder_expert.collator import CrossEncoderCollator
from cross_encoder_expert.metrics import compute_metrics
from cross_encoder_expert.trainer import RegressionLossTrainer

try:
    from transformers import EarlyStoppingCallback
except ImportError:
    EarlyStoppingCallback = None


def main():
    p = argparse.ArgumentParser(description="Train cross-encoder regression expert")
    p.add_argument("--train-path", type=str, default="train.json")
    p.add_argument("--dev-path", type=str, default="dev.json")
    p.add_argument("--test-path", type=str, default="test.json")
    p.add_argument("--save-path", type=str, default=None, help="Save best model here (e.g. checkpoints/cross_encoder.pt)")
    p.add_argument("--output-dir", type=str, default=None, help="Trainer output dir (default: same dir as save-path or ./cross_encoder_results)")
    p.add_argument("--label-key", type=str, default=default_config.LABEL_KEY)
    p.add_argument("--epochs", type=int, default=default_config.EPOCHS)
    p.add_argument("--lr", type=float, default=default_config.LR)
    p.add_argument("--batch-size", type=int, default=default_config.BATCH_SIZE)
    p.add_argument("--seed", type=int, default=default_config.SEED)
    args = p.parse_args()

    set_seed(args.seed)
    output_dir = args.output_dir or (Path(args.save_path).parent if args.save_path else default_config.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    train_data, dev_data, test_data, frozen_dim = load_and_process_data(
        args.train_path,
        args.dev_path,
        args.test_path,
        use_frozen_embeddings=default_config.USE_FROZEN_EMBEDDINGS,
        frozen_model_name=default_config.FROZEN_MODEL_NAME,
        label_key=args.label_key,
    )
    tokenizer = AutoTokenizer.from_pretrained(default_config.MODEL_NAME)

    def tokenize_fn(batch):
        return tokenize_cross_encoder(
            batch,
            tokenizer,
            default_config.MAX_LENGTH,
            default_config.USE_FROZEN_EMBEDDINGS,
        )

    train_ds = Dataset.from_list(train_data).map(tokenize_fn, batched=True)
    dev_ds = Dataset.from_list(dev_data).map(tokenize_fn, batched=True)
    test_ds = Dataset.from_list(test_data).map(tokenize_fn, batched=True)
    keep_cols = ["input_ids", "attention_mask", "labels", "stdev"]
    if "token_type_ids" in train_ds.column_names:
        keep_cols.append("token_type_ids")
    if default_config.USE_FROZEN_EMBEDDINGS:
        keep_cols.extend(["frozen_a", "frozen_b"])
    for ds in (train_ds, dev_ds, test_ds):
        ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

    collator = CrossEncoderCollator(tokenizer)
    model = CrossEncoderRegressor(
        default_config.MODEL_NAME,
        frozen_dim=frozen_dim if default_config.USE_FROZEN_EMBEDDINGS else 0,
        use_dual_path=default_config.USE_DUAL_PATH,
        pooling=default_config.POOLING,
    )
    metrics_fn = lambda eval_pred: compute_metrics(eval_pred, default_config.LABEL_MIN, default_config.LABEL_MAX)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=default_config.WEIGHT_DECAY,
        warmup_ratio=default_config.WARMUP_RATIO,
        max_grad_norm=1.0,
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        disable_tqdm=False,
        dataloader_num_workers=0,
    )
    callbacks = []
    if getattr(default_config, "EARLY_STOPPING_PATIENCE", None) and EarlyStoppingCallback is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=default_config.EARLY_STOPPING_PATIENCE))

    trainer = RegressionLossTrainer(
        loss_type=default_config.LOSS_TYPE,
        loss_x0=default_config.LOSS_X0,
        loss_k=default_config.LOSS_K,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=metrics_fn,
        callbacks=callbacks,
    )
    trainer.train()

    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Training log saved to {log_path}")

    trainer.save_model(output_dir)
    trainer.save_state()
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and name.startswith("checkpoint-"):
            shutil.rmtree(path)
    print(f"Best model saved to {output_dir}")

    state_path = os.path.join(output_dir, "cross_encoder_state.pt")
    torch.save(model.state_dict(), state_path)
    print(f"State dict saved to {state_path}")
    if args.save_path and os.path.abspath(output_dir) != os.path.abspath(Path(args.save_path).parent):
        torch.save({"state_dict": model.state_dict(), "expert": "cross_encoder"}, args.save_path)
        print(f"Also saved to {args.save_path} for ensemble/scripts.")


if __name__ == "__main__":
    main()
