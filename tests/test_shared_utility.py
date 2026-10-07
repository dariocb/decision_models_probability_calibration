import torch

from decision_models.coherent import DecisionViews, SharedUtilityHead


def test_noul_marginals_equal_choice_probabilities() -> None:
    utilities = torch.tensor([2.0, -0.5, 0.7])
    choice = SharedUtilityHead.choice(utilities)

    for index in range(utilities.numel()):
        binary = SharedUtilityHead.noul(utilities, index)
        torch.testing.assert_close(binary[1], choice[index])
        torch.testing.assert_close(binary.sum(), torch.tensor(1.0))


def test_log_odds_form_is_exactly_equivalent() -> None:
    utilities = torch.randn(8, 4, generator=torch.Generator().manual_seed(7))
    for index in range(utilities.shape[-1]):
        direct = SharedUtilityHead.noul(utilities, index)
        via_log_odds = SharedUtilityHead.noul_from_log_odds(utilities, index)
        torch.testing.assert_close(direct, via_log_odds, atol=1e-6, rtol=1e-6)


def test_score_uses_same_distribution_and_expected_value() -> None:
    utilities = torch.tensor([-1.0, 0.0, 1.0])
    values = torch.tensor([-1.0, 0.0, 1.0])
    choice = SharedUtilityHead.choice(utilities)
    score_probs, expected = SharedUtilityHead.score(utilities, values)

    torch.testing.assert_close(score_probs, choice)
    torch.testing.assert_close(expected, (choice * values).sum())


def test_optional_consistency_loss_detects_a_legacy_head_mismatch() -> None:
    canonical = torch.tensor([0.1, 0.2, 0.7])
    coherent = SharedUtilityHead.consistency_loss(canonical, canonical.clone())
    incoherent = SharedUtilityHead.consistency_loss(
        canonical, torch.tensor([0.4, 0.2, 0.4])
    )
    assert coherent.item() == 0.0
    assert incoherent.item() > 0.0


def test_schema_validates_alignment() -> None:
    schema = DecisionViews(("negative", "neutral", "positive"), (-1, 0, 1))
    assert schema.option_ids[1] == "neutral"
