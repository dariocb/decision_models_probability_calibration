import pytest

from decision_models.benchmark import coherence_metrics, make_sentiment_coherence_record


def test_record_contains_equivalent_views() -> None:
    record = make_sentiment_coherence_record("A surprisingly good result", 2)
    assert record["label_name"] == "positive"
    assert record["questions"]["sentiment_choice"]["type"] == "choice"
    assert record["questions"]["sentiment_score"]["type"] == "score"
    assert record["questions"]["is_positive"]["type"] == "noul"


def test_zero_metrics_for_coherent_answers() -> None:
    probs = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
    answers = {
        "sentiment_choice": {"probabilities": probs},
        "sentiment_score": {"probabilities": probs},
        "is_negative": {"noul": 0.1},
        "is_neutral": {"noul": 0.2},
        "is_positive": {"noul": 0.7},
    }
    metrics = coherence_metrics(answers)
    assert metrics.choice_score_l1 == pytest.approx(0.0)
    assert metrics.choice_noul_l1 == pytest.approx(0.0)
    assert metrics.noul_simplex_error == pytest.approx(0.0)


def test_metrics_detect_binary_incoherence() -> None:
    probs = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
    answers = {
        "sentiment_choice": {"probabilities": probs},
        "sentiment_score": {"probabilities": probs},
        "is_negative": {"noul": 0.4},
        "is_neutral": {"noul": 0.5},
        "is_positive": {"noul": 0.8},
    }
    metrics = coherence_metrics(answers)
    assert metrics.choice_noul_l1 > 0
    assert metrics.noul_simplex_error == pytest.approx(0.7)


def test_score_probabilities_may_use_ordinal_index_keys() -> None:
    choice = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
    answers = {
        "sentiment_choice": {"probabilities": choice},
        "sentiment_score": {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
        "is_negative": {"noul": 0.1},
        "is_neutral": {"noul": 0.2},
        "is_positive": {"noul": 0.7},
    }
    assert coherence_metrics(answers).choice_score_l1 == pytest.approx(0.0)
