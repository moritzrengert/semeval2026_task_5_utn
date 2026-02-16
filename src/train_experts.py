"""Training entry point for NLI and SBERT experts.
Created by Moritz Rengert.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from losses import coral_expected_value, coral_loss, compute_soft_kl_loss
from scipy.stats import spearmanr
from nli_expert import NliExpertConfig, NliPlausibilityExpert, make_nli_collate_fn
from sbert_expert import SbertExpertConfig, SbertSemanticMatchingExpert, make_sbert_collate_fn
from expert_dataset import SemevalExpertDataset
from data_utils import expand_annotator_samples, load_dataset


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Transfer all items in the batch to the device."""
    def _move(item):
        if isinstance(item, torch.Tensor):
            return item.to(device)
        if isinstance(item, dict):
            return {k: _move(v) for k, v in item.items()}
        return item

    return {k: _move(v) for k, v in batch.items()}


def resolve_device(requested_device: str | None) -> torch.device:
    """Select the available device."""
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def forward_batch(model: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """ Handle different input formats for NLI vs SBERT experts."""
    if "context_ids" in batch:
        return model(
            batch["context_ids"],
            batch["context_mask"],
            batch["gloss_ids"],
            batch["gloss_mask"],
        )
    return model(batch["input_ids"], batch["attention_mask"])


def build_expert(args: argparse.Namespace) -> Tuple[torch.nn.Module, Any, str]:
    if args.expert == "nli":
        config: NliExpertConfig = args.expert_config
        model = NliPlausibilityExpert(config)
        collate = make_nli_collate_fn(model.tokenizer, max_length=config.max_length, num_classes=config.num_classes)
        head = "nli"
    elif args.expert == "sbert":
        config: SbertExpertConfig = args.expert_config
        model = SbertSemanticMatchingExpert(config)
        collate = make_sbert_collate_fn(model.tokenizer, max_length=config.max_length, num_classes=config.num_classes)
        head = "sbert"
    else:
        raise ValueError(f"Unknown expert type '{args.expert}'")
    if args.train_encoder_layers > 0:
        model.unfreeze_last_layers(args.train_encoder_layers)
    return model, collate, head


def build_datasets(args: argparse.Namespace) -> Tuple[SemevalExpertDataset, SemevalExpertDataset | None]:
    train_records = load_dataset(args.train_path)
    dev_records = load_dataset(args.dev_path) if args.dev_path else []
    if args.expand_annotators:
        train_records = expand_annotator_samples(train_records, label_key=args.label_key, choices_key="choices")
        if dev_records:
            dev_records = expand_annotator_samples(dev_records, label_key=args.label_key, choices_key="choices")

    train_ds = SemevalExpertDataset(
        train_records,
        num_classes=args.num_classes,
        label_key=args.label_key,
        target_aware=args.target_aware,
    )
    dev_ds = (
        SemevalExpertDataset(
            dev_records,
            num_classes=args.num_classes,
            label_key=args.label_key,
            target_aware=args.target_aware,
        )
        if dev_records
        else None
    )
    return train_ds, dev_ds


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    train_loader: DataLoader,
    args: argparse.Namespace,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    # Split encoder vs head params for differential learning rates.
    enc_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            enc_params.append(param)
        else:
            head_params.append(param)
    param_groups = []
    if enc_params and args.encoder_lr > 0:
        param_groups.append({"params": enc_params, "lr": args.encoder_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    coral_weight: float,
    mse_weight: float,
    soft_weight: float,
) -> Dict[str, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    mse_sum = 0.0
    mae_sum = 0.0
    acc_sum = 0.0
    coral_sum = 0.0
    kl_sum = 0.0
    kl_count = 0
    preds_all = []
    labels_all = []
    within_sd_correct = 0
    within_sd_total = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = forward_batch(model, batch)
        logits = outputs["ordinal_logits"]
        class_logits = outputs.get("class_logits")
        coral = coral_loss(logits, batch["ordinal_labels"], num_classes=num_classes) * (num_classes - 1)
        preds = coral_expected_value(logits) + 1.0
        class_pred = (torch.sigmoid(logits) > 0.5).sum(dim=-1)
        mse_term = torch.mean((preds - batch["scores"]) ** 2)
        mse_sum += mse_term.item() * batch["scores"].size(0)
        mae_sum += torch.abs(preds - batch["scores"]).sum().item()
        total_loss = coral * coral_weight + mse_term * mse_weight
        # optional KL if soft labels exist
        soft_targets = batch.get("soft_targets")
        soft_mask = batch.get("soft_mask")
        kl, active_count = compute_soft_kl_loss(class_logits, soft_targets, soft_mask)
        if kl is not None:
            kl_sum += kl.item() * active_count
            kl_count += active_count
            if soft_weight > 0:
                total_loss = total_loss + soft_weight * kl
        coral_sum += coral.item() * batch["scores"].size(0)
        loss_sum += total_loss.item() * batch["scores"].size(0)
        acc_sum += (class_pred == batch["ordinal_labels"]).sum().item()
        total += batch["scores"].size(0)
        preds_all.append(preds.detach().cpu())
        labels_all.append(batch["scores"].detach().cpu())
        # within SD metric
        choices = batch.get("choices")
        choices_mask = batch.get("choices_mask")
        if choices is not None and choices_mask is not None:
            for i in range(choices.size(0)):
                if choices_mask[i].any():
                    vals = choices[i][choices_mask[i]].float()
                    mean = vals.mean().item()
                    sd = vals.std(unbiased=True).item() if vals.numel() > 1 else 0.0
                    within = (mean - sd) < preds[i].item() < (mean + sd)
                    within = within or abs(mean - preds[i].item()) < 1.0
                    within_sd_correct += 1 if within else 0
                    within_sd_total += 1
    if total == 0:
        return {"loss": 0.0, "mse": 0.0, "mae": 0.0, "acc": 0.0, "coral": 0.0, "kl": 0.0, "spearman": 0.0, "acc_within_sd": 0.0}
    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()
    spearman = spearmanr(preds_all, labels_all).correlation if len(preds_all) > 1 else 0.0
    acc_within_sd = within_sd_correct / within_sd_total if within_sd_total else 0.0
    return {
        "loss": loss_sum / total,
        "mse": mse_sum / total,
        "mae": mae_sum / total,
        "acc": acc_sum / total,
        "coral": coral_sum / total,
        "kl": kl_sum / kl_count if kl_count else 0.0,
        "spearman": spearman,
        "acc_within_sd": acc_within_sd,
    }


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    train_ds, dev_ds = build_datasets(args)

    model, collate_fn, expert_name = build_expert(args)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if dev_ds else None

    model.to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader, args)
    best_dev_loss = float("inf")
    epochs_without_improve = 0
    history = []
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()} if args.use_ema else None
    prev_train_loss = None
    overfit_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_to_device(batch, device)
            outputs = model.compute_loss(batch)
            coral_component = outputs["loss"] * (args.num_classes - 1)
            mse_component = torch.mean((coral_expected_value(outputs["ordinal_logits"]) + 1.0 - batch["scores"]) ** 2)
            loss = coral_component * args.coral_weight + mse_component * args.mse_weight
            # Optional soft-label loss
            soft_targets = batch.get("soft_targets")
            soft_mask = batch.get("soft_mask")
            class_logits = outputs.get("class_logits")
            soft_loss, _ = compute_soft_kl_loss(class_logits, soft_targets, soft_mask)
            if soft_loss is not None and args.soft_weight > 0:
                loss = loss + args.soft_weight * soft_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            if ema_state is not None:
                with torch.no_grad():
                    for name, param in model.state_dict().items():
                        ema_state[name].mul_(args.ema_decay).add_(param, alpha=1 - args.ema_decay)
            train_loss_sum += loss.item() * batch["scores"].size(0)
            train_count += batch["scores"].size(0)
            if step % args.log_every == 0:
                print(f"[{expert_name}] epoch {epoch+1} step {step}: loss={loss.item():.4f}")
        train_loss_epoch = train_loss_sum / max(1, train_count)
        if dev_loader:
            backup_state = None
            if ema_state is not None:
                backup_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_state, strict=False)
            metrics = evaluate(
                model,
                dev_loader,
                device,
                num_classes=args.num_classes,
                coral_weight=args.coral_weight,
                mse_weight=args.mse_weight,
                soft_weight=args.soft_weight,
            )
            print(
                f"[{expert_name}] dev epoch {epoch+1}: composite={metrics['loss']:.4f}, coral={metrics['coral']:.4f}, mse={metrics['mse']:.4f}, kl={metrics['kl']:.4f}, mae={metrics['mae']:.4f}, acc={metrics['acc']:.4f}, spearman={metrics['spearman']:.4f}, acc_sd={metrics['acc_within_sd']:.4f}"
            )
            if backup_state is not None:
                model.load_state_dict(backup_state, strict=False)
            history.append((epoch + 1, train_loss_epoch, metrics["loss"]))
            # Early stopping
            if metrics["loss"] + args.early_stop_delta < best_dev_loss:
                best_dev_loss = metrics["loss"]
                epochs_without_improve = 0
                if args.save_path:
                    save_path = Path(args.save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    state_to_save = ema_state if ema_state is not None else model.state_dict()
                    cpu_state = {k: v.detach().cpu().clone() for k, v in state_to_save.items()}
                    torch.save({"state_dict": cpu_state, "expert": expert_name}, save_path)
                    print(f"Saved {expert_name} weights to {save_path} (best dev loss).")
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= args.early_stop_patience:
                    print(f"[{expert_name}] Early stopping at epoch {epoch+1} (no improvement in {args.early_stop_patience} epochs).")
                    break
            # Overfitting detection: train improving while dev worse than best by margin
            if (
                prev_train_loss is not None
                and train_loss_epoch < prev_train_loss - args.overfit_train_delta
                and metrics["loss"] > best_dev_loss + args.overfit_dev_delta
            ):
                overfit_epochs += 1
                if overfit_epochs >= args.overfit_patience:
                    print(f"[{expert_name}] Stopping early due to overfitting detected ({overfit_epochs} epochs).")
                    break
            else:
                overfit_epochs = 0
        else:
            history.append((epoch + 1, train_loss_epoch, None))
        prev_train_loss = train_loss_epoch

    # Save final weights if not already saved best
    if args.save_path and not dev_loader:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state_to_save = ema_state if ema_state is not None else model.state_dict()
        cpu_state = {k: v.detach().cpu().clone() for k, v in state_to_save.items()}
        torch.save({"state_dict": cpu_state, "expert": expert_name}, save_path)
        print(f"Saved {expert_name} weights to {save_path}")

    # Plot loss curves
    if not args.no_plot and history and any(h[2] is not None for h in history):
        epochs, train_losses, dev_losses = zip(*[(e, tr, dv if dv is not None else tr) for e, tr, dv in history])
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_losses, label="train loss")
        plt.plot(epochs, dev_losses, label="dev loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{expert_name} training")
        plt.legend()
        plt.tight_layout()
        plot_path = Path(args.plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        print(f"Saved loss plot to {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", choices=["nli", "sbert"], default="nli")
    parser.add_argument("--train-path", type=str, default="semeval26-05-scripts/data/train.json")
    parser.add_argument("--dev-path", type=str, default="semeval26-05-scripts/data/dev.json")
    parser.add_argument("--label-key", type=str, default="average")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--plot-path", type=str, default="training_curve.png")
    parser.add_argument("--no-plot", action="store_true", help="Disable saving loss plot.")
    parser.add_argument("--target-aware", action="store_true", help="Mark the target homonym in text.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    config = NliExpertConfig() if args.expert == "nli" else SbertExpertConfig()
    args.expert_config = config
    for key, value in vars(config).items():
        setattr(args, key, value)
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)
