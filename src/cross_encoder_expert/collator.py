"""Data collator for batching cross-encoder inputs.
Created by Adam Jen Khai Lo."""

import torch
from dataclasses import dataclass
from typing import Any


@dataclass
class CrossEncoderCollator:
    """Collates tokenized examples into padded batches."""
    tokenizer: Any

    def __call__(self, features):
        """Pad sequences to same length and create batch tensors."""
        # Handle both dict and list formats
        if isinstance(features, dict):
            batch_size = len(features["labels"])
            input_list = []
            labels = [float(x) for x in features["labels"]]
            stdev = [float(x) for x in features.get("stdev", [0.0] * batch_size)]
            
            for i in range(batch_size):
                inp = {"input_ids": features["input_ids"][i], "attention_mask": features["attention_mask"][i]}
                if "token_type_ids" in features:
                    inp["token_type_ids"] = features["token_type_ids"][i]
                input_list.append(inp)
        else:
            labels = [float(f["labels"]) for f in features]
            stdev = [float(f.get("stdev", 0.0)) for f in features]
            input_list = []
            
            for f in features:
                inp = {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
                if "token_type_ids" in f:
                    inp["token_type_ids"] = f["token_type_ids"]
                input_list.append(inp)
        
        # Pad and create batch
        batch = self.tokenizer.pad(input_list, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        batch["stdev"] = torch.tensor(stdev, dtype=torch.float32)
        
        return batch
