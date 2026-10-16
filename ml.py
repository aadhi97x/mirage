"""
Three small, local scikit-learn models. No GPU, no external service, fixed random seeds.

1. baseline_model  - gradient boosting learns weekly orders from spend/discount/season on the first half of the year and is tested on the second.
2. anomaly_model   - Isolation Forest scores products by how unusual their behaviour is (no labels needed).
3. uplift_model    - T-learner (two gradient-boosting models) estimates the extra units per order associated with a deep discount.

Every result is reported as measured, including "the model is no better than a simple average". There are no labels, so none of
these is a verified leak detector; they are evidence layers next to the statistical tests in traps.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

SEED = 7


def _ci(x: np.ndarray, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    means = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------- 1. baseline
def baseline_model(raw: dict, o: pd.DataFrame) -> dict:
    t = o[o.status == "Completed"].copy()
    t["week"] = pd.to_datetime(t.timestamp).dt.to_period("W-SUN").dt.start_time
    wk = t.groupby("week").agg(orders=("transaction_id", "size"), avg_disc=("discount_pct", "mean")).reset_index()
    c = raw["campaigns"].copy()
    c["week"] = pd.to_datetime(c.date).dt.to_period("W-SUN").dt.start_time
    wk = wk.merge(c.groupby("week").spend.sum().reset_index(), on="week", how="left").fillna({"spend": 0})
    wk = wk.iloc[1:-1].reset_index(drop=True)  # drop partial edge weeks
    wk["woy"] = wk.week.dt.isocalendar().week.astype(int)
    X, y = wk[["spend", "avg_disc", "woy"]], wk.orders
    cut = len(wk) // 2
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=80, learning_rate=0.05, min_samples_leaf=3, random_state=SEED).fit(Xtr, ytr)
    pred = m.predict(Xte)
    naive = np.full(len(yte), ytr.mean())
    mae_m, mae_n = mean_absolute_error(yte, pred), mean_absolute_error(yte, naive)
    imp = permutation_importance(m, Xte, yte, n_repeats=200, random_state=SEED, scoring="neg_mean_absolute_error")
    resid = (yte.values - pred)
    return dict(weeks=len(wk), train_weeks=cut, test_weeks=len(wk) - cut, mae_model=float(mae_m), mae_naive=float(mae_n),
                r2=float(r2_score(yte, pred)), beats_naive=bool(mae_m < mae_n),
                spend_importance=float(imp.importances_mean[0]), spend_importance_sd=float(imp.importances_std[0]),
                resid_mean=float(resid.mean()), resid_ci=_ci(resid), table=wk.assign(set=["train"] * cut + ["test"] * (len(wk) - cut)))


# ---------------------------------------------------------------- 2. anomalies
def anomaly_model(o: pd.DataFrame, min_orders: int = 20, top: int = 10) -> dict:
    g = o.groupby("product_id").agg(orders=("transaction_id", "size"), net_sales=("booked", "sum"), avg_discount=("discount_pct", "mean"),
                                    bad_rate=("is_bad", "mean"), below_cost=("below_cost", "mean"), margin=("margin", "sum")).reset_index()
    g = g[g.orders >= min_orders].reset_index(drop=True)
    feats = ["orders", "net_sales", "avg_discount", "bad_rate", "below_cost", "margin"]
    Z = StandardScaler().fit_transform(g[feats])
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED).fit(Z)
    g["anomaly_score"] = -iso.score_samples(Z)  # higher = more unusual
    zdf = pd.DataFrame(Z, columns=feats)
    g["most_unusual_on"] = zdf.abs().idxmax(axis=1).values
    g["direction"] = [("high" if zdf.loc[i, f] > 0 else "low") for i, f in enumerate(g.most_unusual_on)]
    g = g.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    return dict(n_products=len(g), table=g.head(top), all=g)


# ---------------------------------------------------------------- 3. uplift (T-learner)
def uplift_model(o: pd.DataFrame, deep: float = 0.15) -> dict:
    d = o[(o.status == "Completed") & ((o.discount_pct == 0) | (o.discount_pct >= deep))].copy()
    d["treated"] = (d.discount_pct >= deep).astype(int)
    d["month_n"] = pd.to_datetime(d.timestamp).dt.month
    feats = ["category", "list_price", "region", "customer_segment", "channel", "payment_method", "month_n"]
    X = pd.get_dummies(d[feats], columns=["category", "region", "customer_segment", "channel", "payment_method"]).astype(float)
    y = d.quantity.values
    rng = np.random.default_rng(SEED)
    test = rng.random(len(d)) < 0.5
    tr, te = ~test, test
    kw = dict(max_depth=4, max_iter=120, learning_rate=0.05, random_state=SEED)
    m1 = HistGradientBoostingRegressor(**kw).fit(X[tr & (d.treated == 1).values], y[tr & (d.treated == 1).values])
    m0 = HistGradientBoostingRegressor(**kw).fit(X[tr & (d.treated == 0).values], y[tr & (d.treated == 0).values])
    uplift = m1.predict(X[te]) - m0.predict(X[te])
    lo, hi = _ci(uplift)
    naive = float(y[(d.treated == 1).values].mean() - y[(d.treated == 0).values].mean())
    return dict(n_treated=int(d.treated.sum()), n_control=int((1 - d.treated).sum()), mean_uplift=float(uplift.mean()), ci=(lo, hi),
                naive_diff=naive, sd=float(uplift.std()), control_mean=float(y[(d.treated == 0).values].mean()))


def run_all(raw: dict, o: pd.DataFrame) -> dict:
    return dict(baseline=baseline_model(raw, o), anomalies=anomaly_model(o), uplift=uplift_model(o))
