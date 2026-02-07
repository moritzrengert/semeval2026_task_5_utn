from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
from torch.utils.data import Dataset

from final_model.data import load_dataset, ordinal_class


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
    ):
        self.records = list(records)
        self.num_classes = num_classes
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        sample_id = str(record.get("sample_id", idx))
        if self.label_key not in record:
            raise KeyError(f"Record {sample_id} is missing label '{self.label_key}'")
        label_val = float(record[self.label_key])
        ordinal_label = ordinal_class(label_val, self.num_classes)
        context = build_context(record)
        meaning = str(record.get("judged_meaning", ""))
        hypothesis = build_hypothesis(record)
        return {
            "sample_id": sample_id,
            "context": context,
            "gloss": meaning,
            "premise": context,
            "hypothesis": hypothesis,
            "label": torch.tensor(label_val, dtype=torch.float),
            "ordinal_label": torch.tensor(ordinal_label, dtype=torch.long),
        }


def load_expert_dataset(path: str | None, num_classes: int = 5, label_key: str = "average") -> SemevalExpertDataset:
    records = load_dataset(path) if path else []
    return SemevalExpertDataset(records, num_classes=num_classes, label_key=label_key)
