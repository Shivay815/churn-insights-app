"""Unit tests for the training pipeline on synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from train import MODEL_FEATURES, add_sentiment, build_pipeline, recall_at_decile, train_sentiment


def synthetic_frame(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "payment_type": rng.choice(["credit_card", "boleto"], n),
            "customer_state": rng.choice(["SP", "RJ", "MG"], n),
            "first_order_value": rng.uniform(10, 500, n),
            "freight_value": rng.uniform(0, 60, n),
            "item_count": rng.integers(1, 5, n),
            "distinct_products": rng.integers(1, 3, n),
            "installments": rng.integers(1, 10, n),
            "delivery_days": rng.integers(1, 40, n),
            "is_late_delivery": rng.integers(0, 2, n),
            "review_score": rng.integers(0, 6, n),
            "review_sentiment": rng.uniform(0, 1, n),
            "purchase_month": rng.integers(1, 13, n),
            "purchase_dow": rng.integers(0, 7, n),
            "review_text": ["otimo produto recomendo"] * (n // 2)
            + ["pessimo nao chegou"] * (n - n // 2),
        }
    )


def test_pipeline_fits_and_scores():
    df = synthetic_frame()
    y = (df.review_score >= 4).astype(int)
    pipe = build_pipeline()
    pipe.fit(df[MODEL_FEATURES], y)
    scores = pipe.predict_proba(df[MODEL_FEATURES])[:, 1]
    assert scores.shape == (len(df),)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_pipeline_handles_unknown_categories():
    df = synthetic_frame()
    y = (df.first_order_value > 200).astype(int)
    pipe = build_pipeline()
    pipe.fit(df[MODEL_FEATURES], y)
    unseen = df.head(1).copy()
    unseen["customer_state"] = "XX"  # never seen in training
    assert 0 <= pipe.predict_proba(unseen[MODEL_FEATURES])[0, 1] <= 1


def test_sentiment_scores_polarity():
    df = synthetic_frame(n=600)
    df["review_score"] = np.where(
        df.review_text.str.startswith("otimo"), 5, 1
    )
    model = train_sentiment(df)
    scored = add_sentiment(df, model)
    pos = scored[scored.review_text.str.startswith("otimo")].review_sentiment.mean()
    neg = scored[scored.review_text.str.startswith("pessimo")].review_sentiment.mean()
    assert pos > neg


def test_recall_at_decile_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    scores = np.linspace(1, 0, 100)
    assert recall_at_decile(y, scores) == 1.0
