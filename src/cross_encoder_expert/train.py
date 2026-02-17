"""
Train cross-encoder expert.

Run from repo root:
python src/cross_encoder_expert/train.py --train-path ... --dev-path ... --test-path ...
Created by Adam Jen Khai Lo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

try:
    from transformers import EarlyStoppingCallback
except ImportError:
    EarlyStoppingCallback = None

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cross_encoder_expert.ce_data_utils import load_dataset, process_examples, tokenize_cross_encoder
from cross_encoder_expert.collator import CrossEncoderCollator
from cross_encoder_expert.losses import SmoothK2Loss
from cross_encoder_expert.metrics import compute_metrics
from cross_encoder_expert.model import CrossEncoderRegressor

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 256
LABEL_KEY = "average"
OUTPUT_DIR = "./cross_encoder_results"

DEFAULT_SEED = 42
DEFAULT_LR = 8e-6
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 25


class SimpleRegressionTrainer(Trainer):
    """Trainer that applies SmoothK2 regression loss."""

    def __init__(self, x0: float = 0.15, k: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = SmoothK2Loss(x0=x0, k=k)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels").float()
        model_inputs = {k: v for k, v in inputs.items() if k not in ("labels", "stdev")}
        outputs = model(**model_inputs)
        loss = self.loss_fn(outputs["logits"], labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.get("labels")
        with torch.no_grad():
            model_inputs = {k: v for k, v in inputs.items() if k not in ("labels", "stdev")}
            outputs = model(**model_inputs)
            loss = self.loss_fn(outputs["logits"], labels.float()) if labels is not None else None
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, outputs["logits"].detach(), labels)


def _build_callbacks():
    callbacks = []
    if EarlyStoppingCallback is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=5))
    return callbacks


def _cpu_state_dict(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cross-encoder regression expert")
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--dev-path", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, default=None, help="Optional compatibility checkpoint file.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--label-key", type=str, default=LABEL_KEY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--loss", type=str, default="smooth_k2", choices=["smooth_k2"])
    parser.add_argument("--x0", type=float, default=0.15, help="SmoothK2 loss x0 parameter.")
    parser.add_argument("--k", type=float, default=1.0, help="SmoothK2 loss k parameter.")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.train_path}, {args.dev_path}, {args.test_path}...")
    train_data = process_examples(load_dataset(args.train_path), label_key=args.label_key)
    dev_data = process_examples(load_dataset(args.dev_path), label_key=args.label_key)
    test_data = process_examples(load_dataset(args.test_path), label_key=args.label_key)
    print(f"Loaded: {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenize_cross_encoder(batch, tokenizer, MAX_LENGTH)

    train_ds = Dataset.from_list(train_data).map(tokenize_fn, batched=True)
    dev_ds = Dataset.from_list(dev_data).map(tokenize_fn, batched=True)

    keep_cols = ["input_ids", "attention_mask", "labels", "stdev"]
    if "token_type_ids" in train_ds.column_names:
        keep_cols.append("token_type_ids")
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    dev_ds = dev_ds.remove_columns([c for c in dev_ds.column_names if c not in keep_cols])

    model = CrossEncoderRegressor(MODEL_NAME)
    collator = CrossEncoderCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
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
        callbacks=_build_callbacks(),
    )

    trainer.train()
    print("Training complete")

    trainer.save_model(str(output_dir))
    state_dict = _cpu_state_dict(trainer.model)

    state_path = output_dir / "cross_encoder_state.pt"
    torch.save(state_dict, state_path)
    # Backward-compatible file name for prior runs.
    legacy_state_path = output_dir / "cross_encoder_only_state.pt"
    torch.save(state_dict, legacy_state_path)
    print(f"Model artifacts written to {output_dir}")

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state_dict, "expert": "cross_encoder"}, save_path)
        print(f"Compatibility checkpoint saved to {save_path}")

    print("Final evaluation on dev set:")
    for key, val in trainer.evaluate().items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()
