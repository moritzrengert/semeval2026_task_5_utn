from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

Tokenizer = Callable[[str], List[str]]
__all__ = [
    "Tokenizer",
    "simple_tokenizer",
    "load_dataset",
    "build_text_features",
    "render_text",
    "ordinal_class",
    "Vocabulary",
    "EnsembleSample",
    "EnsembleDataset",
    "build_collate_fn",
]


def simple_tokenizer(text: str) -> List[str]:
    """Whitespace tokenizer with lowercase."""
    return text.lower().split()


def load_dataset(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load the JSON dataset shipped with the SemEval task."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    # Files are stored as a mapping from id -> record.
    if isinstance(payload, dict):
        return list(payload.values())
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported dataset format at {path}")


def build_text_features(
    record: Dict[str, Any],
    base_outputs: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """Collect textual snippets to feed the final model."""
    pieces: List[Tuple[str, str]] = []
    precontext = record.get("precontext")
    if precontext:
        pieces.append(("precontext", str(precontext)))
    sentence = record.get("sentence")
    if sentence:
        pieces.append(("sentence", str(sentence)))
    ending = record.get("ending")
    if ending:
        pieces.append(("ending", str(ending)))
    meaning = record.get("judged_meaning")
    if meaning:
        pieces.append(("meaning", str(meaning)))
    example_sentence = record.get("example_sentence")
    if example_sentence:
        pieces.append(("example", str(example_sentence)))
    homonym = record.get("homonym")
    if homonym:
        pieces.append(("target_homonym", str(homonym)))
    if base_outputs:
        for model_name in sorted(base_outputs):
            output_text = base_outputs[model_name]
            if output_text:
                pieces.append((f"{model_name}_output", str(output_text)))
    return pieces


def render_text(pieces: Sequence[Tuple[str, str]], separator: str = " [SEP] ") -> str:
    """Concatenate named text features for the text encoder."""
    return separator.join(f"{name}: {text}" for name, text in pieces)


def ordinal_class(score: float, num_classes: int) -> int:
    """Map a floating score in [1, num_classes] to a 0-based ordinal class using midpoint thresholds."""
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")
    # Use 0.5 midpoints between classes to avoid banker’s rounding bias.
    idx = int(max(0, min(num_classes - 1, (score - 0.5) // 1)))
    return idx


class Vocabulary:
    """Minimal vocabulary for the lightweight Transformer encoder."""

    def __init__(
        self,
        token_to_idx: Optional[Dict[str, int]] = None,
        min_freq: int = 1,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
    ):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.min_freq = min_freq
        if token_to_idx is None:
            self.token_to_idx = {pad_token: 0, unk_token: 1}
        else:
            self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: tok for tok, idx in self.token_to_idx.items()}

    @property
    def pad_idx(self) -> int:
        return self.token_to_idx[self.pad_token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_idx)

    def encode_token(self, token: str) -> int:
        return self.token_to_idx.get(token, self.token_to_idx[self.unk_token])

    def encode(self, text: str, tokenizer: Tokenizer = simple_tokenizer) -> List[int]:
        return [self.encode_token(tok) for tok in tokenizer(text)]

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        tokenizer: Tokenizer = simple_tokenizer,
        min_freq: int = 1,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
    ) -> "Vocabulary":
        counter: Counter = Counter()
        for text in texts:
            counter.update(tokenizer(text))
        token_to_idx = {pad_token: 0, unk_token: 1}
        for token, freq in counter.items():
            if freq >= min_freq and token not in token_to_idx:
                token_to_idx[token] = len(token_to_idx)
        return cls(token_to_idx=token_to_idx, min_freq=min_freq, unk_token=unk_token, pad_token=pad_token)


@dataclass
class EnsembleSample:
    input_ids: torch.Tensor
    length: int
    dense_features: Optional[Dict[str, torch.Tensor]]
    label: torch.Tensor
    ordinal_label: torch.Tensor
    sample_id: str


class EnsembleDataset(Dataset):
    """Dataset that assembles raw SemEval data and arbitrary model outputs."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        vocabulary: Vocabulary,
        tokenizer: Tokenizer = simple_tokenizer,
        base_outputs: Optional[Dict[str, Dict[str, str]]] = None,
        dense_feature_lookup: Optional[Dict[str, Dict[str, Sequence[float]]]] = None,
        separator: str = " [SEP] ",
        num_classes: int = 5,
        label_key: str = "average",
    ):
        self.records = list(records)
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.base_outputs = base_outputs or {}
        self.separator = separator
        self.num_classes = num_classes
        self.label_key = label_key
        self.dense_feature_lookup = dense_feature_lookup or {}
        self.dense_feature_dims: Dict[str, int] = {}
        for name, lookup in self.dense_feature_lookup.items():
            if lookup:
                first = next(iter(lookup.values()))
                self.dense_feature_dims[name] = len(first)

    def _collect_base_outputs(self, sample_id: str) -> Dict[str, str]:
        bundle: Dict[str, str] = {}
        for name, lookup in self.base_outputs.items():
            if lookup is None:
                continue
            value = lookup.get(sample_id)
            if value is not None:
                bundle[name] = value
        return bundle

    def _collect_dense_features(self, sample_id: str) -> Optional[Dict[str, torch.Tensor]]:
        if not self.dense_feature_lookup:
            return None
        dense: Dict[str, torch.Tensor] = {}
        for name, lookup in self.dense_feature_lookup.items():
            value = lookup.get(sample_id)
            if value is None:
                continue
            tensor = torch.tensor(value, dtype=torch.float)
            expected_dim = self.dense_feature_dims.get(name)
            if expected_dim and tensor.numel() != expected_dim:
                raise ValueError(
                    f"Dense feature '{name}' for sample {sample_id} has {tensor.numel()} values; expected {expected_dim}"
                )
            dense[name] = tensor
        return dense if dense else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> EnsembleSample:
        record = self.records[idx]
        sample_id = str(record.get("sample_id", idx))
        text_parts = build_text_features(record, self._collect_base_outputs(sample_id))
        rendered = render_text(text_parts, separator=self.separator)
        token_ids = torch.tensor(self.vocabulary.encode(rendered, tokenizer=self.tokenizer), dtype=torch.long)
        if self.label_key not in record:
            raise KeyError(f"Record {sample_id} is missing label key '{self.label_key}'")
        try:
            label_value = float(record[self.label_key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Label '{self.label_key}' for sample {sample_id} must be numeric") from exc
        dense_features = self._collect_dense_features(sample_id)
        return EnsembleSample(
            input_ids=token_ids,
            length=len(token_ids),
            dense_features=dense_features,
            label=torch.tensor(label_value, dtype=torch.float),
            ordinal_label=torch.tensor(ordinal_class(label_value, self.num_classes), dtype=torch.long),
            sample_id=sample_id,
        )


def build_collate_fn(
    vocabulary: Vocabulary,
    dense_feature_dims: Optional[Dict[str, int]] = None,
) -> Callable[[List[EnsembleSample]], Dict[str, Any]]:
    """Create a collate_fn that pads text and stacks dense features."""

    def collate(batch: List[EnsembleSample]) -> Dict[str, Any]:
        batch_size = len(batch)
        max_len = max(sample.length for sample in batch)
        input_ids = torch.full((batch_size, max_len), vocabulary.pad_idx, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        for row, sample in enumerate(batch):
            input_ids[row, : sample.length] = sample.input_ids
            attention_mask[row, : sample.length] = 1
        dense_bundle: Optional[Dict[str, torch.Tensor]] = None
        if dense_feature_dims:
            dense_bundle = {}
            for name, dim in dense_feature_dims.items():
                stacked = []
                for sample in batch:
                    dense = None if sample.dense_features is None else sample.dense_features.get(name)
                    if dense is None:
                        stacked.append(torch.zeros(dim, dtype=torch.float))
                    else:
                        stacked.append(dense.float())
                dense_bundle[name] = torch.stack(stacked, dim=0)
        labels = torch.stack([sample.label for sample in batch], dim=0)
        ordinal_labels = torch.stack([sample.ordinal_label for sample in batch], dim=0)
        sample_ids = [sample.sample_id for sample in batch]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "dense_features": dense_bundle,
            "labels": labels,
            "ordinal_labels": ordinal_labels,
            "sample_ids": sample_ids,
        }

    return collate
