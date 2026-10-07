from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


SENTIMENT_LABELS = ("negative", "neutral", "positive")
SENTIMENT_DESCRIPTIONS = {
    "negative": "The overall sentiment expressed by the text is negative.",
    "neutral": "The text is neutral or does not express a clear sentiment.",
    "positive": "The overall sentiment expressed by the text is positive.",
}


def make_sentiment_coherence_record(text: str, label: int | None = None) -> dict[str, Any]:
    """Express one sentiment variable through equivalent Clef question types."""
    questions: dict[str, Any] = {
        "sentiment_choice": {
            "type": "choice",
            "instructions": "What is the overall sentiment of this text?",
            "criteria": SENTIMENT_DESCRIPTIONS,
        },
        "sentiment_score": {
            "type": "score",
            "instructions": (
                "Rate the overall sentiment on the ordered scale from negative "
                "through neutral to positive."
            ),
            "criteria": [SENTIMENT_DESCRIPTIONS[name] for name in SENTIMENT_LABELS],
        },
    }
    for name in SENTIMENT_LABELS:
        questions[f"is_{name}"] = {
            "type": "noul",
            "instructions": f"Is the overall sentiment of this text {name}?",
            "criteria": {
                "true": SENTIMENT_DESCRIPTIONS[name],
                "false": f"The overall sentiment is not {name}.",
            },
        }
    record: dict[str, Any] = {"state": {"text": text}, "questions": questions}
    if label is not None:
        record["label"] = int(label)
        record["label_name"] = SENTIMENT_LABELS[int(label)]
    return record


def load_tweet_eval_records(split: str = "test", limit: int | None = 100) -> list[dict[str, Any]]:
    """Load TweetEval sentiment and construct cross-formulation records."""
    from datasets import load_dataset

    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [make_sentiment_coherence_record(row["text"], row["label"]) for row in dataset]


def _probability_vector(answer: Mapping[str, Any], labels: Sequence[str]) -> np.ndarray:
    probabilities = answer.get("probabilities")
    if probabilities is None:
        raise KeyError("answer has no probabilities")
    if isinstance(probabilities, Mapping):
        if all(label in probabilities for label in labels):
            return np.asarray([probabilities[label] for label in labels], dtype=float)
        if len(probabilities) == len(labels):
            try:
                ordered_keys = sorted(probabilities, key=lambda key: int(key))
            except (TypeError, ValueError):
                ordered_keys = list(probabilities)
            return np.asarray([probabilities[key] for key in ordered_keys], dtype=float)
        raise KeyError(f"probability keys do not align with labels: {list(probabilities)}")
    return np.asarray(probabilities, dtype=float)


def _noul_probability(answer: Any) -> float:
    if isinstance(answer, (int, float)):
        return float(answer)
    if "noul" in answer:
        return float(answer["noul"])
    if "probability" in answer:
        return float(answer["probability"])
    if "probabilities" in answer:
        probabilities = answer["probabilities"]
        if isinstance(probabilities, Mapping):
            return float(probabilities.get("true", probabilities.get("yes")))
    raise KeyError("cannot find true probability in noul answer")


@dataclass(frozen=True)
class CoherenceMetrics:
    choice_score_l1: float
    choice_noul_l1: float
    noul_simplex_error: float
    choice_noul_normalized_l1: float


def coherence_metrics(answers: Mapping[str, Any]) -> CoherenceMetrics:
    """Compare the probability vectors returned for equivalent questions."""
    choice = _probability_vector(answers["sentiment_choice"], SENTIMENT_LABELS)
    score = _probability_vector(answers["sentiment_score"], SENTIMENT_LABELS)
    noul = np.asarray(
        [_noul_probability(answers[f"is_{name}"]) for name in SENTIMENT_LABELS],
        dtype=float,
    )
    noul_sum = float(noul.sum())
    normalized = noul / noul_sum if noul_sum > 0 else np.full_like(noul, 1 / len(noul))
    return CoherenceMetrics(
        choice_score_l1=float(np.abs(choice - score).mean()),
        choice_noul_l1=float(np.abs(choice - noul).mean()),
        noul_simplex_error=abs(noul_sum - 1.0),
        choice_noul_normalized_l1=float(np.abs(choice - normalized).mean()),
    )
