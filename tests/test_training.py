import torch

from decision_models.base import CompatibilityScorer
from decision_models.training import (
    classification_metrics,
    decision_logits,
    expected_calibration_error,
    fit_temperature,
    train_head,
)


def test_metrics_are_perfect_for_confident_correct_predictions() -> None:
    logits = torch.tensor([[20.0, -20.0], [-20.0, 20.0]])
    targets = torch.tensor([0, 1])
    metrics = classification_metrics(logits, targets)
    assert metrics.accuracy == 1.0
    assert metrics.brier < 1e-10
    assert metrics.expected_calibration_error < 1e-6


def test_training_reduces_loss_on_separable_cached_embeddings() -> None:
    torch.manual_seed(4)
    hidden = 4
    scorer = CompatibilityScorer(hidden, projection_size=8)
    states = torch.cat((torch.ones(20, hidden), -torch.ones(20, hidden)))
    targets = torch.cat((torch.zeros(20), torch.ones(20))).long()
    question = torch.zeros(1, hidden)
    options = torch.stack((torch.ones(hidden), -torch.ones(hidden))).unsqueeze(0)
    before = torch.nn.functional.cross_entropy(
        decision_logits(scorer, states, question, options), targets
    )
    train_head(
        scorer,
        states,
        targets,
        question,
        options,
        epochs=20,
        learning_rate=1e-2,
        label_smoothing=0.0,
        brier_weight=0.0,
    )
    after = torch.nn.functional.cross_entropy(
        decision_logits(scorer, states, question, options), targets
    )
    assert after < before


def test_temperature_scaling_reduces_validation_nll() -> None:
    with torch.inference_mode():
        logits = torch.tensor([[8.0, -8.0], [8.0, -8.0], [8.0, -8.0], [8.0, -8.0]])
    targets = torch.tensor([0, 0, 0, 1])
    before = torch.nn.functional.cross_entropy(logits, targets)
    temperature = fit_temperature(logits, targets)
    after = torch.nn.functional.cross_entropy(logits / temperature, targets)
    assert temperature > 1
    assert after < before
