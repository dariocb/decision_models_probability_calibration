from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

import torch


class QuestionType(StrEnum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


@dataclass(frozen=True)
class Option:
    """A proposition supplied at runtime rather than a fixed output neuron."""

    id: str
    description: str


@dataclass(frozen=True)
class Question:
    id: str
    type: QuestionType
    instructions: str
    options: Sequence[Option]

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("a question needs at least one option")
        if len({option.id for option in self.options}) != len(self.options):
            raise ValueError("option ids must be unique")


@dataclass
class DecisionResult:
    """A normalized result; `value` is type-specific."""

    probabilities: Mapping[str, torch.Tensor]
    value: str | torch.Tensor

