"""Text helpers specific to the NLI expert."""

from __future__ import annotations

from typing import Any, Dict


def build_hypothesis(record: Dict[str, Any], target_aware: bool = False) -> str:
    """Create an NLI-style hypothesis spelling out the judged meaning."""
    word = str(record.get("homonym", "the word"))
    if target_aware and word:
        word = f"[TGT] {word} [/TGT]"
    meaning = record.get("judged_meaning", "")
    return f"The word '{word}' here means: {meaning}"
