"""
Train cross-encoder only (no bi-encoder/frozen embeddings).
Run from repo root: python src/cross_encoder_only/train.py --train-path ... --dev-path ... --test-path ...
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, set_seed

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cross_encoder_only.data_utils import load_dataset, process_examples, tokenize_cross_encoder
from cross_encoder_only.model import CrossEncoderRegressor
from cross_encoder_only.collator import CrossEncoderCollator
from cross_encoder_only.metrics import compute_metrics
from cross_encoder_only.losses import SmoothK2Loss

# ===== CONFIG (inline) =====
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 256
LABEL_KEY = "average"
OUTPUT_DIR = "./cross_encoder_only_results"

# Training defaults
DEFAULT_SEED = 42
DEFAULT_LR = 8e-6
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 25
LOSS_TYPE = "smooth_k2"


class SimpleRegressionTrainer(Trainer):
    """Custom trainer with SmoothK2 loss."""
    def __init__(self, x0: float = 0.15, k: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use SmoothK2 loss function
        self.loss_fn = SmoothK2Loss(x0=x0, k=k)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute regression loss between predictions and labels."""
        # Don't pop from inputs - make copies to avoid modifying dict
        labels = inputs.get("labels").float()
        # Create filtered inputs without labels/stdev
        model_inputs = {k: v for k, v in inputs.items() if k not in ["labels", "stdev"]}
        outputs = model(**model_inputs)
        loss = self.loss_fn(outputs["logits"], labels)
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction_step to ensure proper output format for metrics."""
        labels = inputs.get("labels")
        # Forward pass
        with torch.no_grad():
            model_inputs = {k: v for k, v in inputs.items() if k not in ["labels", "stdev"]}
            outputs = model(**model_inputs)
            loss = self.loss_fn(outputs["logits"], labels.float()) if labels is not None else None
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Return predictions as tensors (not numpy) - Trainer will convert to numpy later
        logits = outputs["logits"].detach()
        
        return (loss, logits, labels)


def main():
    p = argparse.ArgumentParser(description="Train pure cross-encoder (no frozen embeddings)")
    p.add_argument("--train-path", type=str, required=True)
    p.add_argument("--dev-path", type=str, required=True)
    p.add_argument("--test-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--label-key", type=str, default=LABEL_KEY)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--loss", type=str, default="smooth_k2", choices=["smooth_k2"])
    p.add_argument("--x0", type=float, default=0.15, help="SmoothK2 loss x0 parameter")
    p.add_argument("--k", type=float, default=1.0, help="SmoothK2 loss k parameter")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and process raw JSON data into text pairs
    print(f"Loading data from {args.train_path}, {args.dev_path}, {args.test_path}...")
    train_data = process_examples(load_dataset(args.train_path), label_key=args.label_key)
    dev_data = process_examples(load_dataset(args.dev_path), label_key=args.label_key)
    test_data = process_examples(load_dataset(args.test_path), label_key=args.label_key)
    print(f"Loaded: {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenize_cross_encoder(batch, tokenizer, MAX_LENGTH)

    # Tokenize datasets
    train_ds = Dataset.from_list(train_data).map(tokenize_fn, batched=True)
    dev_ds = Dataset.from_list(dev_data).map(tokenize_fn, batched=True)
    test_ds = Dataset.from_list(test_data).map(tokenize_fn, batched=True)
    
    # Keep only required columns for training
    keep_cols = ["input_ids", "attention_mask", "labels", "stdev"]
    if "token_type_ids" in train_ds.column_names:
        keep_cols.append("token_type_ids")
    
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    dev_ds = dev_ds.remove_columns([c for c in dev_ds.column_names if c not in keep_cols])
    test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep_cols])

    # Initialize model and collator
    model = CrossEncoderRegressor(MODEL_NAME)
    collator = CrossEncoderCollator(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mae",
        greater_is_better=False,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.03,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )
    
    # Create trainer
    trainer = SimpleRegressionTrainer(
        x0=args.x0,
        k=args.k,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    print("Training configuration:")
    # Train with early stopping
    trainer.train()
    print("\nTraining complete")

    # Save trained model and state dict
    trainer.save_model(args.output_dir)
    state_path = os.path.join(args.output_dir, "cross_encoder_only_state.pt")
    torch.save(model.state_dict(), state_path)
    
    print(f"Model saved to {args.output_dir}")

    # Final evaluation on dev set
    print("\nFinal Evaluation:")
    for key, val in trainer.evaluate().items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()
