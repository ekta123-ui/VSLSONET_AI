from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
REVIEW_DATA_PATH = BASE_DIR / "data" / "user_reviews.parquet"
SENTIMENT_LABELS = ("Negative", "Neutral", "Positive")
SCORE_MIN = 0
SCORE_MAX = 10


@lru_cache(maxsize=1)
def _get_valid_scores() -> pd.Series:
    if not REVIEW_DATA_PATH.exists():
        raise FileNotFoundError(f"Review dataset not found: {REVIEW_DATA_PATH}")

    frame = pd.read_parquet(REVIEW_DATA_PATH)
    score_column = next(
        (column for column in frame.columns if str(column).strip().casefold() == "score"),
        None,
    )
    if score_column is None:
        raise ValueError("Review dataset must contain a score column.")

    scores = pd.to_numeric(frame[score_column], errors="coerce").dropna()
    return scores[scores.between(SCORE_MIN, SCORE_MAX)]


def create_sentiment(score: float | str) -> str:
    numeric_score = float(score)
    if not SCORE_MIN <= numeric_score <= SCORE_MAX:
        raise ValueError(f"Score must be between {SCORE_MIN} and {SCORE_MAX}.")
    if numeric_score <= 4:
        return "Negative"
    if numeric_score <= 6:
        return "Neutral"
    return "Positive"


def get_sentiment_distribution() -> dict:
    scores = _get_valid_scores()
    labels = scores.map(create_sentiment)
    counts = labels.value_counts().reindex(SENTIMENT_LABELS, fill_value=0)
    total = int(counts.sum())

    return {
        "total_reviews": total,
        "label_source": "VALORANT Metacritic score thresholds; not human sentiment annotation",
        "labels": list(SENTIMENT_LABELS),
        "counts": {label: int(counts[label]) for label in SENTIMENT_LABELS},
        "percentages": {
            label: (round(float(counts[label]) / total * 100, 2) if total else 0.0)
            for label in SENTIMENT_LABELS
        },
        "topic_sentiment": None,
        "patch_sentiment": None,
    }


def get_score_distribution() -> dict:
    scores = _get_valid_scores()
    score_counts = scores.value_counts().sort_index()
    labels = scores.map(create_sentiment)
    total = int(len(scores))

    by_sentiment = {}
    for label in SENTIMENT_LABELS:
        label_scores = scores[labels.eq(label)]
        count = int(len(label_scores))
        by_sentiment[label] = {
            "count": count,
            "percentage": round(count / total * 100, 2) if total else 0.0,
            "average_score": round(float(label_scores.mean()), 2) if count else None,
        }

    return {
        "total_reviews": total,
        "score_column": "score",
        "score_distribution": [
            {"score": float(score), "count": int(score_counts[score])}
            for score in score_counts.index
        ],
        "sentiment_distribution": by_sentiment,
        "label_mapping": {
            "negative": f"{SCORE_MIN}-4",
            "neutral": "5-6",
            "positive": f"7-{SCORE_MAX}",
        },
    }