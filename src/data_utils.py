"""Dataset loading and label utilities for expert models.
Created by Moritz Rengert.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union


def _numeric_sort_keys(keys: Sequence[str]) -> List[str]:
    try:
        return sorted(keys, key=lambda k: int(k))
    except Exception:
        return sorted(keys)


def load_dataset(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a SemEval JSON dataset file."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        keys = _numeric_sort_keys(list(payload.keys()))
        return [payload[k] for k in keys]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported dataset format at {path}")


def expand_annotator_samples(
    records: Sequence[Dict[str, Any]],
    label_key: str = "average",
    choices_key: str = "choices",
    include_average: bool = False,
) -> List[Dict[str, Any]]:
    """Treat each annotator score as its own training sample."""
    expanded: List[Dict[str, Any]] = []
    for record in records:
        choices = record.get(choices_key)
        if isinstance(choices, (list, tuple)) and choices:
            for annot_idx, choice in enumerate(choices):
                new_record = dict(record)
                new_record[label_key] = choice
                new_record["annotator_index"] = annot_idx
                expanded.append(new_record)
            if include_average and label_key in record:
                expanded.append(dict(record))
        else:
            expanded.append(record)
    return expanded


def ordinal_class(score: float, num_classes: int) -> int:
    """Map score in [1, num_classes] to a 0-based ordinal class."""
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")
    return int(max(0, min(num_classes - 1, (score - 0.5) // 1)))
