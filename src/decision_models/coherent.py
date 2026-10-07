from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from .base import CompatibilityScorer, DecisionModel
from .types import DecisionResult, Option


@dataclass(frozen=True)
class DecisionViews:
    """Multiple views of one exclusive-and-exhaustive latent variable."""

    option_ids: Sequence[str]
    ordinal_values: Sequence[float] | None = None

    def __post_init__(self) -> None:
        if len(self.option_ids) < 2:
            raise ValueError("at least two exhaustive propositions are required")
        if len(set(self.option_ids)) != len(self.option_ids):
            raise ValueError("option ids must be unique")
        if self.ordinal_values is not None and len(self.ordinal_values) != len(self.option_ids):
            raise ValueError("ordinal_values must align with option_ids")


class SharedUtilityHead(nn.Module):
    """Derive all query types from a single categorical distribution."""

    @staticmethod
    def probabilities(utilities: torch.Tensor) -> torch.Tensor:
        return utilities.softmax(dim=-1)

    @classmethod
    def choice(cls, utilities: torch.Tensor) -> torch.Tensor:
        return cls.probabilities(utilities)

    @classmethod
    def noul(cls, utilities: torch.Tensor, proposition_index: int) -> torch.Tensor:
        """Return `[P(false), P(true)]` for an exhaustive proposition.

        P(true) equals the corresponding choice marginal exactly. In utility
        form this is sigmoid(u_i - logsumexp(u_not_i)), not sigmoid(u_i).
        """
        probs = cls.probabilities(utilities)
        yes = probs[..., proposition_index]
        return torch.stack((1.0 - yes, yes), dim=-1)

    @classmethod
    def noul_from_log_odds(
        cls, utilities: torch.Tensor, proposition_index: int
    ) -> torch.Tensor:
        selected = utilities[..., proposition_index]
        mask = torch.ones_like(utilities, dtype=torch.bool)
        mask[..., proposition_index] = False
        alternatives = utilities.masked_fill(~mask, -torch.inf).logsumexp(-1)
        yes = torch.sigmoid(selected - alternatives)
        return torch.stack((1.0 - yes, yes), dim=-1)

    @classmethod
    def score(
        cls, utilities: torch.Tensor, ordinal_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = cls.probabilities(utilities)
        values = ordinal_values.to(device=probs.device, dtype=probs.dtype)
        return probs, (probs * values).sum(-1)

    @staticmethod
    def consistency_loss(
        canonical: torch.Tensor,
        *alternative_views: torch.Tensor,
    ) -> torch.Tensor:
        """Optional soft constraint for legacy independently-scored views.

        The shared-utility paths above make this loss identically zero. This
        helper is useful during migration when a model still exposes separate
        heads whose distributions should approach the canonical softmax.
        """
        if not alternative_views:
            return canonical.new_zeros(())
        losses = [(view - canonical).square().mean() for view in alternative_views]
        return torch.stack(losses).mean()


class CoherentDecisionModel(nn.Module):
    """Qwen-compatible model with structural cross-question coherence."""

    def __init__(
        self,
        encoder: DecisionModel,
        *,
        projection_size: int = 512,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.scorer = CompatibilityScorer(encoder.hidden_size, projection_size)
        self.views = SharedUtilityHead()

    def utilities(
        self,
        state: str,
        question: str,
        propositions: Sequence[Option],
    ) -> torch.Tensor:
        state_vector = self.encoder.encode_texts([state])
        question_vector = self.encoder.encode_texts([question])
        option_vectors = self.encoder.encode_options(propositions).unsqueeze(0)
        return self.scorer(state_vector, question_vector, option_vectors).squeeze(0)

    def decide_from_utilities(
        self,
        utilities: torch.Tensor,
        schema: DecisionViews,
    ) -> Mapping[str, DecisionResult]:
        if utilities.shape[-1] != len(schema.option_ids):
            raise ValueError("utilities must align with schema.option_ids")
        choice_probs = self.views.choice(utilities)
        named_choice = dict(zip(schema.option_ids, choice_probs.unbind(-1)))
        results: dict[str, DecisionResult] = {
            "choice": DecisionResult(
                probabilities=named_choice,
                value=schema.option_ids[int(choice_probs.argmax(-1))],
            )
        }
        for index, option_id in enumerate(schema.option_ids):
            binary = self.views.noul(utilities, index)
            results[f"noul:{option_id}"] = DecisionResult(
                probabilities={"false": binary[..., 0], "true": binary[..., 1]},
                value=binary[..., 1],
            )
        if schema.ordinal_values is not None:
            values = torch.as_tensor(schema.ordinal_values)
            score_probs, expectation = self.views.score(utilities, values)
            results["score"] = DecisionResult(
                probabilities=dict(zip(schema.option_ids, score_probs.unbind(-1))),
                value=expectation,
            )
        return results
