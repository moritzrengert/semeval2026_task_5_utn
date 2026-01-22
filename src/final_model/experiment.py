from __future__ import annotations

import argparse
import importlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from .data import (
    EnsembleDataset,
    Vocabulary,
    build_collate_fn,
    build_text_features,
    load_dataset,
    render_text,
)
from .losses import coral_expected_value
from .model import FinalEnsembleModel, LossConfig
from .modules import EnsembleModelConfig


@dataclass
class AdapterSpec:
    module_path: str
    class_name: str

    @classmethod
    def parse(cls, text: str) -> "AdapterSpec":
        if ":" not in text:
            raise ValueError("Adapter must be formatted as module_path:ClassName")
        module_path, class_name = text.split(":", 1)
        return cls(module_path=module_path, class_name=class_name)


class BaseModelAdapter:
    """Extend this to plug in arbitrary base models."""

    name: str = "adapter"

    def predict_text(self, record: Dict[str, Any]) -> Optional[str]:
        return None

    def predict_embedding(self, record: Dict[str, Any]) -> Optional[Sequence[float]]:
        return None


class EchoMeaningAdapter(BaseModelAdapter):
    """Example adapter that echoes the judged meaning as a pseudo-output."""

    name = "meaning_echo"

    def predict_text(self, record: Dict[str, Any]) -> Optional[str]:
        meaning = record.get("judged_meaning")
        return f"meaning={meaning}" if meaning else None


def load_named_outputs(entries: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Load key=value pairs where value is a JSON mapping sample_id->payload."""
    payload: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected name=path for output source, got: {entry}")
        name, path = entry.split("=", 1)
        with open(path, "r", encoding="utf-8") as handle:
            payload[name] = json.load(handle)
    return payload


def merge_nested_dicts(dicts: Iterable[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for source in dicts:
        for name, lookup in source.items():
            merged.setdefault(name, {})
            merged[name].update({str(k): v for k, v in lookup.items()})
    return merged


def load_adapters(specs: Sequence[AdapterSpec]) -> List[BaseModelAdapter]:
    adapters: List[BaseModelAdapter] = []
    for spec in specs:
        module = importlib.import_module(spec.module_path)
        cls = getattr(module, spec.class_name)
        instance = cls()
        if not isinstance(instance, BaseModelAdapter):
            raise TypeError(f"{spec.class_name} is not a BaseModelAdapter")
        adapters.append(instance)
    return adapters


def run_adapters(
    adapters: Sequence[BaseModelAdapter],
    records: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, Sequence[float]]]]:
    text_outputs: Dict[str, Dict[str, str]] = defaultdict(dict)
    dense_outputs: Dict[str, Dict[str, Sequence[float]]] = defaultdict(dict)
    for idx, record in enumerate(records):
        sample_id = str(record.get("sample_id", idx))
        for adapter in adapters:
            text = adapter.predict_text(record)
            if text is not None:
                text_outputs[adapter.name][sample_id] = text
            embedding = adapter.predict_embedding(record)
            if embedding is not None:
                dense_outputs[adapter.name][sample_id] = embedding
    return dict(text_outputs), dict(dense_outputs)


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    def _move(item):
        if isinstance(item, torch.Tensor):
            return item.to(device)
        if isinstance(item, dict):
            return {k: _move(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return type(item)(_move(v) for v in item)
        return item

    return {k: _move(v) for k, v in batch.items()}


def build_vocabulary(records: Sequence[Dict[str, Any]], base_outputs: Dict[str, Dict[str, str]], min_freq: int) -> Vocabulary:
    texts = []
    for record in records:
        sample_id = record.get("sample_id")
        sample_outputs = {name: outputs.get(str(sample_id)) for name, outputs in base_outputs.items()}
        texts.append(render_text(build_text_features(record, sample_outputs)))
    return Vocabulary.build(texts, min_freq=min_freq)


def evaluate(
    model: FinalEnsembleModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    acc_count = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                dense_features=batch["dense_features"],
            )
            pred_reg = outputs["regression"]
            mse_sum += torch.mean((pred_reg - batch["labels"]) ** 2).item() * batch["labels"].size(0)
            pred_ord = coral_expected_value(outputs["ordinal_logits"])
            # Convert ordinal expectation back to 1..K scale for MAE.
            mae_sum += torch.abs((pred_ord + 1) - batch["labels"]).sum().item()
            class_logits = outputs.get("class_logits")
            if class_logits is not None:
                acc_count += (class_logits.argmax(dim=-1) == batch["ordinal_labels"]).sum().item()
            total += batch["labels"].size(0)
    return {
        "mse": mse_sum / total if total else 0.0,
        "mae": mae_sum / total if total else 0.0,
        "acc": acc_count / total if total else 0.0,
    }


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_records = load_dataset(args.train_path)
    dev_records = load_dataset(args.dev_path) if args.dev_path else []

    adapter_specs = [AdapterSpec.parse(text) for text in args.adapter]
    adapters = load_adapters(adapter_specs) if adapter_specs else []
    adapter_text, adapter_dense = run_adapters(adapters, train_records + dev_records) if adapters else ({}, {})

    file_text = load_named_outputs(args.text_output) if args.text_output else {}
    file_dense = load_named_outputs(args.dense_output) if args.dense_output else {}

    base_outputs = merge_nested_dicts([file_text, adapter_text]) if not args.disable_textual_base else {}
    dense_lookup = merge_nested_dicts([file_dense, adapter_dense]) if not args.disable_dense_base else {}

    vocab = build_vocabulary(train_records, base_outputs, min_freq=args.min_freq)

    train_dataset = EnsembleDataset(
        records=train_records,
        vocabulary=vocab,
        base_outputs=base_outputs,
        dense_feature_lookup=dense_lookup,
        num_classes=args.num_classes,
    )
    dev_dataset = EnsembleDataset(
        records=dev_records,
        vocabulary=vocab,
        base_outputs=base_outputs,
        dense_feature_lookup=dense_lookup,
        num_classes=args.num_classes,
    ) if dev_records else None

    collate_fn = build_collate_fn(vocab, dense_feature_dims=train_dataset.dense_feature_dims)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if dev_dataset else None

    model_config = EnsembleModelConfig(
        vocab_size=vocab.vocab_size,
        num_classes=args.num_classes,
        pad_idx=vocab.pad_idx,
        text_dim=args.text_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ffn_dim=args.transformer_ffn_dim,
        dropout=args.dropout,
        dense_feature_dims=train_dataset.dense_feature_dims if not args.disable_dense_base else None,
        use_classification_head=args.cross_entropy_weight > 0,
    )
    loss_config = LossConfig(
        regression_weight=args.regression_weight,
        coral_weight=args.coral_weight,
        cross_entropy_weight=args.cross_entropy_weight,
        num_classes=args.num_classes,
    )
    model = FinalEnsembleModel(model_config, loss_config=loss_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            batch = move_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                dense_features=batch["dense_features"],
            )
            losses = model.compute_loss(batch, outputs)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            if step % args.log_every == 0:
                loss_str = ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
                print(f"epoch {epoch+1} step {step}: {loss_str}")
        if dev_loader:
            metrics = evaluate(model, dev_loader, device)
            print(f"[dev] epoch {epoch+1}: mse={metrics['mse']:.4f}, mae={metrics['mae']:.4f}, acc={metrics['acc']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final-stage ensemble model with ablations.")
    parser.add_argument("--train-path", type=Path, default=Path("semeval26-05-scripts/data/train.json"))
    parser.add_argument("--dev-path", type=Path, default=Path("semeval26-05-scripts/data/dev.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--text-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--coral-weight", type=float, default=0.0)
    parser.add_argument("--cross-entropy-weight", type=float, default=0.0)
    parser.add_argument("--adapter", action="append", default=[], help="Adapter in module_path:ClassName format.")
    parser.add_argument("--text-output", action="append", default=[], help="Textual outputs as name=path_to_json.")
    parser.add_argument("--dense-output", action="append", default=[], help="Dense outputs as name=path_to_json.")
    parser.add_argument("--disable-textual-base", action="store_true", help="Drop textual base outputs for ablations.")
    parser.add_argument("--disable-dense-base", action="store_true", help="Drop dense base outputs for ablations.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
