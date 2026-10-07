from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .base import CompatibilityScorer, DecisionModel


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    negative_log_likelihood: float
    brier: float
    expected_calibration_error: float


@torch.inference_mode()
def encode_in_batches(
    encoder: DecisionModel,
    texts: Sequence[str],
    *,
    batch_size: int = 8,
    output_device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Cache frozen backbone representations so head training is inexpensive."""
    chunks = []
    for start in range(0, len(texts), batch_size):
        encoded = encoder.encode_texts(texts[start : start + batch_size])
        chunks.append(encoded.to(device=output_device, dtype=torch.float32))
    return torch.cat(chunks, dim=0)


def decision_logits(
    scorer: CompatibilityScorer,
    states: torch.Tensor,
    question: torch.Tensor,
    options: torch.Tensor,
) -> torch.Tensor:
    """Score a batch against a shared runtime schema."""
    batch_size = states.shape[0]
    questions = question.expand(batch_size, -1)
    option_batch = options.expand(batch_size, -1, -1)
    return scorer(states, questions, option_batch)


def expected_calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    bins: int = 10,
) -> torch.Tensor:
    confidence, prediction = probabilities.max(-1)
    correctness = prediction.eq(targets).to(probabilities.dtype)
    boundaries = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    result = probabilities.new_zeros(())
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            result = result + mask.float().mean() * (
                correctness[mask].mean() - confidence[mask].mean()
            ).abs()
    return result


@torch.inference_mode()
def classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    bins: int = 10,
) -> ClassificationMetrics:
    probabilities = logits.softmax(-1)
    one_hot = nn.functional.one_hot(targets, logits.shape[-1]).to(probabilities.dtype)
    return ClassificationMetrics(
        accuracy=float(logits.argmax(-1).eq(targets).float().mean()),
        negative_log_likelihood=float(nn.functional.cross_entropy(logits, targets)),
        brier=float((probabilities - one_hot).square().sum(-1).mean()),
        expected_calibration_error=float(expected_calibration_error(probabilities, targets, bins=bins)),
    )


def fit_temperature(
    validation_logits: torch.Tensor,
    validation_targets: torch.Tensor,
    *,
    max_iter: int = 100,
) -> torch.Tensor:
    """Fit one positive temperature by validation-set NLL minimization."""
    logits = validation_logits.detach().clone().float()
    targets = validation_targets.detach().clone()
    log_temperature = nn.Parameter(logits.new_zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(logits / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temperature.detach().exp()


def train_head(
    scorer: CompatibilityScorer,
    train_states: torch.Tensor,
    train_targets: torch.Tensor,
    question: torch.Tensor,
    options: torch.Tensor,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.02,
    brier_weight: float = 0.1,
    seed: int = 42,
) -> list[float]:
    """Train only the compatibility head over cached Qwen representations."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    scorer.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(train_states), generator=generator)
        total = 0.0
        seen = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            states = train_states[indices].to(question.device)
            targets = train_targets[indices].to(question.device)
            optimizer.zero_grad(set_to_none=True)
            logits = decision_logits(scorer, states, question, options)
            loss = scorer.supervised_loss(
                logits,
                targets,
                label_smoothing=label_smoothing,
                brier_weight=brier_weight,
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            seen += len(indices)
        history.append(total / seen)
    scorer.eval()
    return history
