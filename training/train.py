"""Train the repeat-purchase propensity model with a CI quality gate.

Two-stage design:
  1. NLP sub-model: TF-IDF + logistic regression over Portuguese review text,
     self-supervised from star ratings (>=4 positive, <=2 negative). Its
     probability becomes one dense feature: review_sentiment.
  2. Main model: HistGradientBoosting over first-order features + sentiment,
     class-weighted for the ~2.8% positive base rate.

Quality gate: if artifacts/metrics.json exists (the incumbent), the new model
must match or beat its ROC-AUC within a small tolerance — otherwise this
script exits non-zero and CI refuses to ship the regression.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
GATE_TOLERANCE = 0.005

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES = REPO_ROOT / "data" / "features.parquet"
ARTIFACTS = REPO_ROOT / "artifacts"

CATEGORICAL = ["payment_type", "customer_state"]
NUMERIC = [
    "first_order_value",
    "freight_value",
    "item_count",
    "distinct_products",
    "installments",
    "delivery_days",
    "is_late_delivery",
    "review_score",
    "review_sentiment",
    "purchase_month",
    "purchase_dow",
]
MODEL_FEATURES = CATEGORICAL + NUMERIC


def train_sentiment(train_df: pd.DataFrame) -> Pipeline:
    labeled = train_df[
        (train_df.review_text.str.len() > 3) & (train_df.review_score != 3)
        & (train_df.review_score > 0)
    ]
    y = (labeled.review_score >= 4).astype(int)
    model = make_pipeline(
        TfidfVectorizer(max_features=3000, min_df=5, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000, C=1.0),
    )
    model.fit(labeled.review_text, y)
    return model


def add_sentiment(df: pd.DataFrame, sentiment: Pipeline) -> pd.DataFrame:
    df = df.copy()
    has_text = df.review_text.str.len() > 3
    df["review_sentiment"] = 0.5  # neutral prior when no text
    if has_text.any():
        df.loc[has_text, "review_sentiment"] = sentiment.predict_proba(
            df.loc[has_text, "review_text"]
        )[:, 1]
    return df


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            ("num", "passthrough", NUMERIC),
        ]
    )
    # Unweighted, shallow: variant comparison showed class_weight="balanced"
    # wrecked calibration (Brier 0.235 vs 0.027) and cost AUC. With honest
    # probabilities the app can show real propensities, not distorted ones.
    clf = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=400,
        early_stopping=True,
        random_state=SEED,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", clf)])


def recall_at_decile(y_true: np.ndarray, scores: np.ndarray) -> float:
    cutoff = np.quantile(scores, 0.9)
    top = scores >= cutoff
    return float(y_true[top].sum() / max(y_true.sum(), 1))


def main() -> None:
    started = time.perf_counter()
    # sort_values: belt-and-braces reproducibility on top of the SQL ORDER BY
    df = pd.read_parquet(FEATURES).sort_values("customer_unique_id").reset_index(drop=True)
    df["is_late_delivery"] = df.is_late_delivery.astype(int)
    y = df.repeated_within_window.astype(int)

    # The sentiment sub-model is supervised by REVIEW STARS, not the repeat
    # label, so fitting it on the full population leaks nothing about the
    # target into the CV evaluation below.
    sentiment = train_sentiment(df)
    df = add_sentiment(df, sentiment)

    # Out-of-fold evaluation: every customer is scored by a model that never
    # saw them. One honest number instead of a single lucky/unlucky split —
    # the original single-split design swung ±0.03 AUC purely on row order,
    # which the quality gate (correctly) refused.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_scores = cross_val_predict(
        build_pipeline(), df[MODEL_FEATURES], y, cv=cv, method="predict_proba"
    )[:, 1]

    metrics = {
        "roc_auc": round(float(roc_auc_score(y, oof_scores)), 4),
        "pr_auc": round(float(average_precision_score(y, oof_scores)), 4),
        "recall_at_top_decile": round(recall_at_decile(y.values, oof_scores), 4),
        "lift_at_top_decile": round(
            recall_at_decile(y.values, oof_scores) / 0.10, 2
        ),
        "brier": round(float(brier_score_loss(y, oof_scores)), 5),
        "base_rate": round(float(y.mean()), 4),
        "evaluation": "5-fold stratified out-of-fold",
        "n_customers": int(len(df)),
        "train_seconds": None,  # filled below
    }

    # ── Quality gate ────────────────────────────────────────────────
    metrics_path = ARTIFACTS / "metrics.json"
    if metrics_path.exists():
        incumbent = json.loads(metrics_path.read_text())
        floor = incumbent["roc_auc"] - GATE_TOLERANCE
        if metrics["roc_auc"] < floor:
            sys.exit(
                f"✗ QUALITY GATE FAILED: new ROC-AUC {metrics['roc_auc']} < "
                f"incumbent floor {floor:.4f} — refusing to ship a regression"
            )
        print(f"✓ gate passed: {metrics['roc_auc']} vs incumbent {incumbent['roc_auc']}")

    # ── Final model: fit on ALL data (evaluation already done via OOF) ──
    pipe = build_pipeline()
    pipe.fit(df[MODEL_FEATURES], y)

    # ── Artifacts ───────────────────────────────────────────────────
    ARTIFACTS.mkdir(exist_ok=True)
    metrics["train_seconds"] = round(time.perf_counter() - started, 1)
    joblib.dump(pipe, ARTIFACTS / "model.joblib", compress=3)
    joblib.dump(sentiment, ARTIFACTS / "sentiment.joblib", compress=3)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    perm = permutation_importance(
        pipe, df[MODEL_FEATURES], y, scoring="roc_auc",
        n_repeats=5, random_state=SEED,
    )
    importances = sorted(
        zip(MODEL_FEATURES, perm.importances_mean.round(5)),
        key=lambda kv: -abs(kv[1]),
    )
    (ARTIFACTS / "feature_importances.json").write_text(
        json.dumps(dict(importances), indent=2)
    )

    # Scored sample for the app's ranking table: out-of-fold scores, so every
    # row was scored by a model that never trained on it (no raw text kept).
    sample = df[MODEL_FEATURES + ["customer_unique_id"]].copy()
    sample["customer_unique_id"] = sample.customer_unique_id.str[:8] + "…"
    sample["propensity"] = oof_scores.round(4)
    sample["actual_repeat"] = y.values
    sample.nlargest(500, "propensity").to_parquet(
        ARTIFACTS / "scored_holdout.parquet"
    )

    print(json.dumps(metrics, indent=2))
    print(f"✓ artifacts written to {ARTIFACTS}")


if __name__ == "__main__":
    main()
