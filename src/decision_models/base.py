from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .types import Option


class DecisionModel(nn.Module, ABC):
    """Base class for models that score runtime propositions.

    Subclasses own text encoding. The rest of the decision stack can therefore
    be tested with a tiny encoder and later swapped for Qwen without changing
    probability semantics.
    """

    @property
    @abstractmethod
    def hidden_size(self) -> int: ...

    @abstractmethod
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """Return one `[hidden_size]` vector per input string."""

    def encode_state(self, state: str) -> torch.Tensor:
        return self.encode_texts([state])[0]

    def encode_options(self, options: Sequence[Option]) -> torch.Tensor:
        return self.encode_texts([option.description for option in options])


class QwenDecisionModel(DecisionModel):
    """Reusable Qwen3.5 encoder for custom decision heads.

    The default checkpoint is the backbone used by Clef-Flash. Loading is lazy:
    importing this package never downloads nine billion parameters.
    """

    DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: Any,
        *,
        max_length: int = 2048,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.max_length = max_length
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        torch_dtype: torch.dtype | str = "auto",
        device_map: str | dict[str, int | str] = "auto",
        freeze_backbone: bool = True,
        **kwargs: Any,
    ) -> "QwenDecisionModel":
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        backbone = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            **kwargs,
        )
        return cls(backbone, tokenizer, freeze_backbone=freeze_backbone)

    @property
    def hidden_size(self) -> int:
        config = self.backbone.config
        for name in ("hidden_size", "text_config"):
            value = getattr(config, name, None)
            if name == "text_config" and value is not None:
                value = getattr(value, "hidden_size", None)
            if value is not None:
                return int(value)
        raise AttributeError("could not infer hidden size from backbone config")

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        outputs = self.backbone(**batch, output_hidden_states=True, return_dict=True)
        hidden = outputs.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)


class CompatibilityScorer(nn.Module):
    """Learn a state/question/option compatibility utility."""

    def __init__(self, hidden_size: int, projection_size: int = 512) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(hidden_size * 2, projection_size),
            nn.GELU(),
            nn.LayerNorm(projection_size),
        )
        self.option = nn.Sequential(
            nn.Linear(hidden_size, projection_size),
            nn.LayerNorm(projection_size),
        )
        self.scale = projection_size**-0.5

    def forward(
        self,
        state: torch.Tensor,
        question: torch.Tensor,
        options: torch.Tensor,
    ) -> torch.Tensor:
        context = self.context(torch.cat((state, question), dim=-1))
        option_vectors = self.option(options)
        return torch.einsum("...d,...kd->...k", context, option_vectors) * self.scale

    @staticmethod
    def supervised_loss(
        utilities: torch.Tensor,
        target: torch.Tensor,
        *,
        label_smoothing: float = 0.0,
        brier_weight: float = 0.0,
    ) -> torch.Tensor:
        ce = F.cross_entropy(utilities, target, label_smoothing=label_smoothing)
        if brier_weight == 0:
            return ce
        probs = utilities.softmax(-1)
        one_hot = F.one_hot(target, utilities.shape[-1]).to(probs.dtype)
        brier = (probs - one_hot).square().sum(-1).mean()
        return ce + brier_weight * brier

