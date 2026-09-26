"""Plain-dataclass Question — inference-only shape passed to the tokenizer.

Kept deliberately minimal: no pydantic runtime, no label validation, no
option-count clamp. Validation happens once in ``_decide._coerce_question``
before we ever build this object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Question:
    qid: str
    type: Literal["choice", "score", "noul"]
    instructions: str
    options: list[str] | None
    # Sentinel label used only by the sequence-trim guard in _format.py
    # (fires if the token budget drops the model's ability to see any option
    # at all). Never read by the heads or the loss — inference has no gold.
    label: int | bool = 0
