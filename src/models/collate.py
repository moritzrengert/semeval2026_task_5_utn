from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase


def make_nli_collate_fn(
    tokenizer: PreTrainedTokenizerBase, max_length: int, num_classes: int
):
    """Tokenize premise/hypothesis pairs for the NLI expert."""

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        premises = [sample["premise"] for sample in batch]
        hypotheses = [sample["hypothesis"] for sample in batch]
        enc = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.stack([sample["ordinal_label"] for sample in batch], dim=0)
        scores = torch.stack([sample["label"] for sample in batch], dim=0)
        softs = [sample.get("soft_targets") for sample in batch]
        soft_targets = torch.zeros((len(batch), num_classes), dtype=torch.float)
        soft_mask = torch.zeros((len(batch),), dtype=torch.bool)
        for i, s in enumerate(softs):
            if s is not None:
                soft_targets[i, : s.numel()] = s
                soft_mask[i] = True
        choices = torch.full((len(batch), num_classes), float("nan"))
        choices_mask = torch.zeros((len(batch), num_classes), dtype=torch.bool)
        for i, sample in enumerate(batch):
            ch = sample.get("choices")
            if ch is not None:
                choices[i, : ch.numel()] = ch
                choices_mask[i, : ch.numel()] = True
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "ordinal_labels": labels,
            "scores": scores,
            "soft_targets": soft_targets,
            "soft_mask": soft_mask,
            "choices": choices,
            "choices_mask": choices_mask,
            "sample_ids": [sample["sample_id"] for sample in batch],
        }

    return collate


def make_sbert_collate_fn(
    tokenizer: PreTrainedTokenizerBase, max_length: int, num_classes: int
):
    """Tokenize context/definition pairs for the SBERT expert."""

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        contexts = [sample["context"] for sample in batch]
        glosses = [sample["gloss"] for sample in batch]
        ctx_enc = tokenizer(
            contexts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        gloss_enc = tokenizer(
            glosses, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        labels = torch.stack([sample["ordinal_label"] for sample in batch], dim=0)
        scores = torch.stack([sample["label"] for sample in batch], dim=0)
        softs = [sample.get("soft_targets") for sample in batch]
        soft_targets = torch.zeros((len(batch), num_classes), dtype=torch.float)
        soft_mask = torch.zeros((len(batch),), dtype=torch.bool)
        for i, s in enumerate(softs):
            if s is not None:
                soft_targets[i, : s.numel()] = s
                soft_mask[i] = True
        choices = torch.full((len(batch), num_classes), float("nan"))
        choices_mask = torch.zeros((len(batch), num_classes), dtype=torch.bool)
        for i, sample in enumerate(batch):
            ch = sample.get("choices")
            if ch is not None:
                choices[i, : ch.numel()] = ch
                choices_mask[i, : ch.numel()] = True
        return {
            "context_ids": ctx_enc["input_ids"],
            "context_mask": ctx_enc["attention_mask"],
            "gloss_ids": gloss_enc["input_ids"],
            "gloss_mask": gloss_enc["attention_mask"],
            "ordinal_labels": labels,
            "scores": scores,
            "soft_targets": soft_targets,
            "soft_mask": soft_mask,
            "choices": choices,
            "choices_mask": choices_mask,
            "sample_ids": [sample["sample_id"] for sample in batch],
        }

    return collate
