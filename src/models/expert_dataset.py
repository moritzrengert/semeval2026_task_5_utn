"""Dataset wrappers for NLI and SBERT expert training.
Created by Moritz Rengert.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
from torch.utils.data import Dataset

from .data_utils import ordinal_class


def build_context(record: Dict[str, Any]) -> str:
    """Combine the contextual fields into a single sentence."""
    parts = [
        record.get("precontext"),
        record.get("sentence"),
        record.get("ending"),
        record.get("example_sentence"),
    ]
    return " ".join(str(p) for p in parts if p).strip()


def build_hypothesis(record: Dict[str, Any]) -> str:
    """Create an NLI-style hypothesis spelling out the judged meaning."""
    word = record.get("homonym", "the word")
    meaning = record.get("judged_meaning", "")
    return f"The word '{word}' here means: {meaning}"


class SemevalExpertDataset(Dataset):
    """Dataset that feeds individual SemEval samples to experts."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        num_classes: int = 5,
        label_key: str = "average",
        require_labels: bool = True,
        drop_invalid_labels: bool = False,
    ):
        self.require_labels = require_labels
        self.drop_invalid_labels = drop_invalid_labels
        self.records = self._filter_records(records, label_key) if require_labels else list(records)
        self.num_classes = num_classes
        self.label_key = label_key
        self.has_labels = require_labels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        sample_id = str(record.get("sample_id", idx))
        if self.require_labels:
            if self.label_key not in record:
                raise KeyError(f"Record {sample_id} is missing label '{self.label_key}'")
            label_val = float(record[self.label_key])
            ordinal_label = ordinal_class(label_val, self.num_classes)
            label_tensor = torch.tensor(label_val, dtype=torch.float)
            ordinal_tensor = torch.tensor(ordinal_label, dtype=torch.long)
        else:
            label_tensor = None
            ordinal_tensor = None
        context = build_context(record)
        meaning = str(record.get("judged_meaning", ""))
        hypothesis = build_hypothesis(record)
        choices = record.get("choices")
        soft_targets = None
        choices_tensor = None
        if choices and isinstance(choices, (list, tuple)) and len(choices) >= self.num_classes:
            probs = torch.tensor(choices[: self.num_classes], dtype=torch.float)
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = torch.full((self.num_classes,), 1.0 / self.num_classes, dtype=torch.float)
            soft_targets = probs
            choices_tensor = torch.tensor(choices[: self.num_classes], dtype=torch.float)
        return {
            "sample_id": sample_id,
            "context": context,
            "gloss": meaning,
            "premise": context,
            "hypothesis": hypothesis,
            "label": label_tensor,
            "ordinal_label": ordinal_tensor,
            "soft_targets": soft_targets,
            "choices": choices_tensor,
        }

    def _filter_records(self, records: Sequence[Dict[str, Any]], label_key: str) -> Sequence[Dict[str, Any]]:
        filtered = []
        for rec in records:
            if label_key not in rec:
                if self.drop_invalid_labels:
                    continue
                raise KeyError(f"Record is missing label '{label_key}'")
            try:
                float(rec[label_key])
            except (TypeError, ValueError):
                if self.drop_invalid_labels:
                    continue
                raise ValueError(f"Label '{label_key}' must be numeric, got {rec[label_key]!r}")
            filtered.append(rec)
        return filtered
