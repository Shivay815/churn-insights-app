"""Contract tests on the built feature table."""

from pathlib import Path

import pandas as pd
import pytest

FEATURES = Path(__file__).resolve().parents[1] / "data" / "features.parquet"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    if not FEATURES.exists():
        pytest.skip("features.parquet not built — run `make features`")
    return pd.read_parquet(FEATURES)


def test_one_row_per_customer(df):
    assert df.customer_unique_id.is_unique


def test_base_rate_sane(df):
    rate = df.repeated_within_window.mean()
    assert 0.01 < rate < 0.10, f"base rate {rate:.4f} outside expected band"


def test_no_nulls_in_model_features(df):
    cols = [
        "first_order_value", "item_count", "installments", "payment_type",
        "delivery_days", "review_score", "customer_state",
        "purchase_month", "purchase_dow",
    ]
    assert df[cols].notna().all().all()


def test_review_score_domain(df):
    assert df.review_score.between(0, 5).all()  # 0 = no review sentinel


def test_value_ranges(df):
    assert (df.first_order_value >= 0).all()
    assert (df.item_count >= 1).all()
