from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from final_model.losses import coral_expected_value, coral_loss
from models.nli_expert import NliExpertConfig, NliPlausibilityExpert, make_nli_collate_fn
from models.sbert_expert import SbertExpertConfig, SbertSemanticMatchingExpert, make_sbert_collate_fn
from models.expert_dataset import load_expert_dataset


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    def _move(item):
        if isinstance(item, torch.Tensor):
            return item.to(device)
        if isinstance(item, dict):
            return {k: _move(v) for k, v in item.items()}
        return item

    return {k: _move(v) for k, v in batch.items()}


def build_expert(
    args: argparse.Namespace,
) -> Tuple[torch.nn.Module, Any, str]:
    default_nli_model = NliExpertConfig().model_name
    default_sbert_model = SbertExpertConfig().model_name
    if args.expert == "nli":
        config = NliExpertConfig(
            model_name=args.model_name or default_nli_model,
            num_classes=args.num_classes,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            max_length=args.max_length,
        )
        model = NliPlausibilityExpert(config)
        collate = make_nli_collate_fn(model.tokenizer, max_length=config.max_length)
        head = "nli"
    elif args.expert == "sbert":
        config = SbertExpertConfig(
            model_name=args.model_name or default_sbert_model,
            num_classes=args.num_classes,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            max_length=args.max_length,
        )
        model = SbertSemanticMatchingExpert(config)
        collate = make_sbert_collate_fn(model.tokenizer, max_length=config.max_length)
        head = "sbert"
    else:
        raise ValueError(f"Unknown expert type '{args.expert}'")
    return model, collate, head


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Dict[str, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    mse_sum = 0.0
    mae_sum = 0.0
    acc_sum = 0.0
    for batch in loader:
        batch = move_to_device(batch, device)
        if hasattr(model, "forward") and "context_ids" in batch:
            logits = model(
                batch["context_ids"],
                batch["context_mask"],
                batch["gloss_ids"],
                batch["gloss_mask"],
            )["ordinal_logits"]
        else:
            logits = model(batch["input_ids"], batch["attention_mask"])["ordinal_logits"]
        loss = coral_loss(logits, batch["ordinal_labels"], num_classes=num_classes)
        preds = coral_expected_value(logits) + 1.0
        class_pred = (torch.sigmoid(logits) > 0.5).sum(dim=-1)
        mse_sum += torch.mean((preds - batch["scores"]) ** 2).item() * batch["scores"].size(0)
        mae_sum += torch.abs(preds - batch["scores"]).sum().item()
        loss_sum += loss.item() * batch["scores"].size(0)
        acc_sum += (class_pred == batch["ordinal_labels"]).sum().item()
        total += batch["scores"].size(0)
    if total == 0:
        return {"loss": 0.0, "mse": 0.0, "mae": 0.0, "acc": 0.0}
    return {
        "loss": loss_sum / total,
        "mse": mse_sum / total,
        "mae": mae_sum / total,
        "acc": acc_sum / total,
    }


def train(args: argparse.Namespace) -> None:
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    train_ds = load_expert_dataset(args.train_path, num_classes=args.num_classes, label_key=args.label_key)
    dev_ds = load_expert_dataset(args.dev_path, num_classes=args.num_classes, label_key=args.label_key) if args.dev_path else None

    model, collate_fn, expert_name = build_expert(args)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if dev_ds else None
    test_loader = None

    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            batch = move_to_device(batch, device)
            outputs = model.compute_loss(batch)
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.log_every == 0:
                print(f"[{expert_name}] epoch {epoch+1} step {step}: loss={loss.item():.4f}")
        if dev_loader:
            metrics = evaluate(model, dev_loader, device, num_classes=args.num_classes)
            print(
                f"[{expert_name}] dev epoch {epoch+1}: loss={metrics['loss']:.4f}, mse={metrics['mse']:.4f}, mae={metrics['mae']:.4f}, acc={metrics['acc']:.4f}"
            )

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "expert": expert_name}, save_path)
        print(f"Saved {expert_name} weights to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen experts with CORAL heads.")
    parser.add_argument("--expert", choices=["nli", "sbert"], default="nli")
    parser.add_argument("--train-path", type=str, default="semeval26-05-scripts/data/train.json")
    parser.add_argument("--dev-path", type=str, default="semeval26-05-scripts/data/dev.json")
    parser.add_argument("--label-key", type=str, default="average")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--model-name", type=str, default=None, help="HF model to use for the chosen expert.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-path", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
