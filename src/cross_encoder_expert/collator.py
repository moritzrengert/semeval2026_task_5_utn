"""Collator for cross-encoder (same as cross_encoder_adam).
Created by Adam Jen Khai Lo."""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass
class CrossEncoderCollator:
    tokenizer: Any

    def __call__(self, features):
        if isinstance(features, dict):
            batch_size = len(features["labels"])
            input_list = []
            labels = [float(x) for x in features["labels"]]
            stdev = [float(x) for x in features.get("stdev", [0.0] * batch_size)]
            frozen_a_list = features.get("frozen_a", None)
            frozen_b_list = features.get("frozen_b", None)
            for i in range(batch_size):
                inp = {"input_ids": features["input_ids"][i], "attention_mask": features["attention_mask"][i]}
                if "token_type_ids" in features:
                    inp["token_type_ids"] = features["token_type_ids"][i]
                input_list.append(inp)
        else:
            labels = [float(f["labels"]) for f in features]
            stdev = [float(f.get("stdev", 0.0)) for f in features]
            frozen_a_list = [f["frozen_a"] for f in features] if "frozen_a" in features[0] else None
            frozen_b_list = [f["frozen_b"] for f in features] if "frozen_b" in features[0] else None
            input_list = []
            for f in features:
                inp = {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
                if "token_type_ids" in f:
                    inp["token_type_ids"] = f["token_type_ids"]
                input_list.append(inp)
        batch = self.tokenizer.pad(input_list, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        batch["stdev"] = torch.tensor(stdev, dtype=torch.float32)
        if frozen_a_list is not None:
            batch["frozen_a"] = torch.tensor(np.array(frozen_a_list), dtype=torch.float32)
            batch["frozen_b"] = torch.tensor(np.array(frozen_b_list), dtype=torch.float32)
        return batch
