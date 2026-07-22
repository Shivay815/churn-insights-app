"""Build the training table from Project A's governed warehouse.

Train/serve parity by construction: features come from the same dbt marts
(fct_orders, stg_reviews) that power the analytics dashboard — there is no
separate, drift-prone feature pipeline.

Label design (censoring-aware):
  target = customer places a second order within LABEL_WINDOW_DAYS of their
  first. Customers whose first order falls inside the final window of the
  dataset are EXCLUDED — they haven't had time to repeat, and labeling them
  0 would poison the training signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAREHOUSE = (
    REPO_ROOT.parent / "ecom-analytics-platform" / "warehouse" / "olist.duckdb"
)
OUT_PATH = REPO_ROOT / "data" / "features.parquet"

LABEL_WINDOW_DAYS = 180

FEATURE_SQL = f"""
with orders as (
    select * from main_marts.fct_orders
    where order_status = 'delivered'
),

first_orders as (
    select *
    from (
        select
            *,
            row_number() over (
                partition by customer_unique_id order by purchased_at
            ) as order_rank
        from orders
    )
    where order_rank = 1
),

max_date as (
    select max(purchased_at) as dataset_end from orders
),

repeats as (
    select
        f.order_id as first_order_id,
        min(o.purchased_at) as second_purchase_at
    from first_orders as f
    join orders as o
        on f.customer_unique_id = o.customer_unique_id
        and o.purchased_at > f.purchased_at
    group by f.order_id
),

reviews as (
    select order_id, comment_message
    from main_staging.stg_reviews
)

select
    f.customer_unique_id,
    -- features: everything known at/shortly after the first order
    cast(f.gross_revenue as double) as first_order_value,
    cast(f.freight_revenue as double) as freight_value,
    f.item_count,
    f.distinct_products,
    coalesce(f.max_installments, 1) as installments,
    coalesce(f.primary_payment_type, 'unknown') as payment_type,
    coalesce(f.delivery_days, -1) as delivery_days,
    coalesce(f.is_late_delivery, false) as is_late_delivery,
    coalesce(f.review_score, 0) as review_score,  -- 0 = no review left
    coalesce(reviews.comment_message, '') as review_text,
    f.customer_state,
    extract(month from f.purchased_at) as purchase_month,
    extract(dow from f.purchased_at) as purchase_dow,
    -- label
    (
        repeats.second_purchase_at is not null
        and repeats.second_purchase_at
            <= f.purchased_at + interval {LABEL_WINDOW_DAYS} days
    ) as repeated_within_window
from first_orders as f
cross join max_date
left join repeats on f.order_id = repeats.first_order_id
left join reviews on f.order_id = reviews.order_id
-- censoring guard: only customers with a full observation window
where f.purchased_at <= max_date.dataset_end - interval {LABEL_WINDOW_DAYS} days
-- deterministic output order: parallel scans otherwise shuffle rows between
-- machines, which changes CV splits and makes metrics irreproducible
order by f.customer_unique_id
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    args = parser.parse_args()
    if not args.warehouse.exists():
        raise SystemExit(
            f"warehouse not found at {args.warehouse} — build Project A first "
            "(see Makefile `data` target)"
        )
    OUT_PATH.parent.mkdir(exist_ok=True)
    con = duckdb.connect(str(args.warehouse), read_only=True)
    con.execute(f"copy ({FEATURE_SQL}) to '{OUT_PATH}' (format parquet)")
    n, pos = con.execute(
        f"select count(*), sum(case when repeated_within_window then 1 else 0 end) "
        f"from ({FEATURE_SQL})"
    ).fetchone()
    print(f"✓ {OUT_PATH.name}: {n:,} customers, {pos:,} repeaters ({100 * pos / n:.2f}% base rate)")
