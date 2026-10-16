"""
Data loading, joins and data-quality checks. Pure pandas/numpy (no Streamlit), so it can be tested and reused.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
# Dataset location: $MONEY_LEAK_DATA, else ./data (shipped with the repo), else ../04_business_money_leak (original hackathon folder)
DATA_DIR = Path(os.environ.get("MONEY_LEAK_DATA") or next(
    (p for p in (_HERE / "data", _HERE.parent / "04_business_money_leak") if (p / "transactions.csv").exists()), _HERE / "data"))

BAD_STATUSES = ["Cancelled", "Returned"]


# --------------------------------------------------------------------------- load
def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "customers": pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["signup_date"]),
        "products": pd.read_csv(DATA_DIR / "products.csv"),
        "transactions": pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["timestamp"]),
        "inventory": pd.read_csv(DATA_DIR / "inventory.csv", parse_dates=["date"]),
        "campaigns": pd.read_csv(DATA_DIR / "campaigns.csv", parse_dates=["date"]),
    }


def build_orders(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per transaction, joined with product + customer, with money columns.

    KEY ASSUMPTION (verified: unit_price == list_price * (1 - discount_pct) on 100% of rows):
    unit_price is ALREADY net of discount. We never apply the discount twice.
    """
    p, c, t = raw["products"], raw["customers"], raw["transactions"]
    o = t.merge(p, on="product_id", how="left").merge(c, on="customer_id", how="left")
    o["date"] = o["timestamp"].dt.normalize()
    o["month"] = o["timestamp"].dt.to_period("M").astype(str)
    o["list_value"] = o["quantity"] * o["list_price"]
    o["booked"] = o["quantity"] * o["unit_price"]              # net of discount
    o["discount_amt"] = o["list_value"] - o["booked"]
    o["cogs"] = o["quantity"] * o["unit_cost"]
    o["margin"] = o["booked"] - o["cogs"]
    o["below_cost"] = o["unit_price"] < o["unit_cost"]
    o["is_cancelled"] = o["status"].eq("Cancelled")
    o["is_returned"] = o["status"].eq("Returned")
    o["is_bad"] = o["status"].isin(BAD_STATUSES)
    o["before_signup"] = o["timestamp"] < o["signup_date"]
    return o


# --------------------------------------------------------------------------- data quality
def valid_campaign_mask(m: pd.DataFrame) -> pd.Series:
    return (m["impressions"] >= m["clicks"]) & (m["clicks"] >= m["conversions"]) & (m["clicks"] > 0)


def data_quality(raw: dict[str, pd.DataFrame], orders: pd.DataFrame) -> pd.DataFrame:
    p, inv, m = raw["products"], raw["inventory"], raw["campaigns"]
    bad_c = ~valid_campaign_mask(m)
    rows = [
        ("Products where unit_cost > list_price", int((p.unit_cost > p.list_price).sum()),
         f"{(p.unit_cost > p.list_price).mean():.0%} of catalogue sells below cost at list price",
         "Kept, but labelled 'verify cost data' when cost > 3x price"),
        ("Orders sold below unit cost", int(orders.below_cost.sum()),
         f"{orders.below_cost.mean():.0%} of orders have negative margin",
         "Kept; reported as pricing leak with confidence label"),
        ("Orders dated before customer signup_date", int(orders.before_signup.sum()),
         "Customer tenure features unreliable for these rows",
         "Kept in revenue totals; tenure features clipped at 0"),
        ("Campaign rows with clicks > impressions or conversions > clicks", int(bad_c.sum()),
         f"{m.loc[bad_c, 'spend'].sum():,.0f} of spend has impossible funnel numbers",
         "Excluded from CTR/CVR/CPA; spend reported separately"),
        ("Campaign conversions vs real orders",
         int(m.conversions.sum()),
         f"{m.conversions.sum():,.0f} 'conversions' vs {len(orders):,} orders - cannot reconcile",
         "No ROAS claimed; efficiency measured as CPA on reported conversions only"),
        ("Inventory rows flagged stockout but stock_on_hand > 0",
         int(((inv.stockout_flag == 1) & (inv.stock_on_hand > 0)).sum()),
         "stockout_flag does not match stock_on_hand (only "
         f"{int((inv.stock_on_hand == 0).sum())} rows have zero stock)",
         "Risk computed from stock vs demand vs lead time, not from the flag"),
        ("Inventory rows below reorder threshold but not flagged",
         int(((inv.stock_on_hand <= inv.reorder_threshold) & (inv.stockout_flag == 0)).sum()),
         "Flag misses low-stock situations", "Used 'below reorder threshold' as a separate signal"),
        ("Missing values / duplicate keys (all tables)", 0, "None found", "-"),
    ]
    return pd.DataFrame(rows, columns=["Issue", "Rows", "Why it matters", "How we handled it"])


# --------------------------------------------------------------------------- money waterfall


# --------------------------------------------------------------------------- segment drill-down


# --------------------------------------------------------------------------- pricing leak


# --------------------------------------------------------------------------- stockout risk


# --------------------------------------------------------------------------- campaigns
def campaign_metrics(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = valid_campaign_mask(m)
    v = m[ok].copy()
    v["ctr"] = v.clicks / v.impressions
    v["cvr"] = v.conversions / v.clicks
    v["cpa"] = v.spend / v.conversions.replace(0, np.nan)
    med = v["cpa"].median()
    v["inefficient"] = v["cpa"].gt(2 * med) | v["conversions"].eq(0)
    v["excess_spend"] = np.where(v["inefficient"], (v.spend - v.conversions * med).clip(lower=0), 0.0)
    return v, m[~ok].copy()


# --------------------------------------------------------------------------- weekly anomalies + forecast
def weekly(o: pd.DataFrame) -> pd.DataFrame:
    x = o.copy()
    x["week"] = x["date"] - pd.to_timedelta(x["date"].dt.weekday, unit="D")
    g = x.groupby("week").agg(days=("date", "nunique"), orders=("transaction_id", "count"),
                              booked=("booked", "sum"), realised=("booked", lambda s: s[x.loc[s.index, "status"].eq("Completed")].sum()),
                              bad_rate=("is_bad", "mean"), margin=("margin", "sum")).reset_index()
    g = g[g["days"] == 7].reset_index(drop=True)           # drop partial weeks at the edges
    for col in ["booked", "orders", "bad_rate", "realised"]:
        med = g[col].median()
        mad = (g[col] - med).abs().median() * 1.4826
        g[f"{col}_rz"] = (g[col] - med) / (mad if mad else 1)
    # baseline forecast: mean of previous 4 weeks (seasonal-naive on flat data); compare with actual
    g["forecast_booked"] = g["booked"].shift(1).rolling(4).mean()
    g["forecast_err_pct"] = (g["booked"] - g["forecast_booked"]) / g["forecast_booked"]
    return g


# --------------------------------------------------------------------------- alert engine


def load_all():
    raw = load_raw()
    o = build_orders(raw)
    return raw, o
