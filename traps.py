"""
Profit Mirage - Growth Trap engine.

Every calculation is deterministic pandas/numpy. A "Growth Trap" = a positive headline metric sitting next to a
concerning supporting metric. Each trap is TESTED on the data; if the data does not support it we say so.

Financial honesty
  * "Net sales" = booked value of Completed orders (unit_price is already net of discount; verified on 100% of rows).
  * "Margin proxy" = unit_price - unit_cost on Completed orders. README warns costs were generated independently, so this
    is a PROXY, never called profit.
  * Every money figure carries a label: Observed / Proxy / Estimate.
Ranking (transparent):  score = 0.4*Impact + 0.3*Urgency + 0.3*Evidence, each in 0..1 (see rank_breakdown on each card).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import analysis as A

W_IMPACT, W_URGENCY, W_EVIDENCE = 0.4, 0.3, 0.3
SEED = 7


def inr(x: float) -> str:
    a = abs(x)
    s = f"{a/1e6:.2f}M" if a >= 1e6 else (f"{a/1e3:.0f}K" if a >= 1e4 else f"{a:,.0f}")
    return f"{'-' if x < 0 else ''}₹{s}"


# ------------------------------------------------------------------ small stats helpers
def perm_corr_p(x, y, n=5000, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = np.corrcoef(x, y)[0, 1]
    rng = np.random.default_rng(seed)
    cnt = sum(abs(np.corrcoef(rng.permutation(x), y)[0, 1]) >= abs(r) for _ in range(n))
    return float(r), float((cnt + 1) / (n + 1))


def boot_corr_ci(x, y, n=3000, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        rs.append(np.corrcoef(x[s], y[s])[0, 1])
    return np.percentile(rs, [2.5, 97.5])


def boot_mean_diff_ci(a, b, n=3000, seed=SEED):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), np.percentile(d, [2.5, 97.5])


def _score(impact_value, net_sales, persistence, checks):
    impact = float(min(1.0, max(0.0, impact_value) / (0.25 * net_sales))) if net_sales else 0.0
    evid = float(np.mean([c["score"] for c in checks])) if checks else 0.0
    total = W_IMPACT * impact + W_URGENCY * persistence + W_EVIDENCE * evid
    return {"impact": impact, "urgency": float(persistence), "evidence": evid, "total": float(total),
            "formula": f"{W_IMPACT}×Impact + {W_URGENCY}×Urgency + {W_EVIDENCE}×Evidence"}


def _check(name, score, note):
    return {"name": name, "score": score, "note": note}


# ------------------------------------------------------------------ context
def context(raw, o):
    comp = o[o.status == "Completed"]
    return {"net_sales": float(comp.booked.sum()), "margin_proxy": float(comp.margin.sum()),
            "orders": int(len(o)), "completed": int(len(comp)),
            "bad_rate": float(o.is_bad.mean()), "disc_intensity": float(o.discount_amt.sum() / o.list_value.sum()),
            "campaign_spend": float(raw["campaigns"].spend.sum()),
            "conversions": int(raw["campaigns"].conversions.sum())}


# ------------------------------------------------------------------ A. ADVERTISING TRAP
def advertising_trap(raw, o) -> dict:
    m = raw["campaigns"].copy()
    comp = o[o.status == "Completed"]
    ctx = context(raw, o)
    spend, net, margin = ctx["campaign_spend"], ctx["net_sales"], ctx["margin_proxy"]

    full_weeks = A.weekly(o)["week"]
    ow = o.assign(week=o.date - pd.to_timedelta(o.date.dt.weekday, unit="D")).groupby("week").agg(
        orders=("transaction_id", "count"), net=("booked", "sum"))
    mw = m.assign(week=m.date - pd.to_timedelta(m.date.dt.weekday, unit="D")).groupby("week").agg(
        spend=("spend", "sum"), conversions=("conversions", "sum"), clicks=("clicks", "sum"))
    W = ow.join(mw).loc[full_weeks].reset_index()
    r, p = perm_corr_p(W.spend, W.orders)
    lo, hi = boot_corr_ci(W.spend, W.orders)
    q_hi, q_lo = W[W.spend >= W.spend.quantile(.75)], W[W.spend <= W.spend.quantile(.25)]
    diff, (dlo, dhi) = boot_mean_diff_ci(q_hi.orders, q_lo.orders)

    # persistence: quarters where campaign spend exceeded the margin proxy of completed orders
    qs = comp.assign(q=comp.timestamp.dt.to_period("Q")).groupby("q").agg(net=("booked", "sum"), margin=("margin", "sum"))
    qs["spend"] = m.assign(q=m.date.dt.to_period("Q")).groupby("q").spend.sum()
    qs["over"] = qs.spend > qs.margin
    persistence = float(qs.over.mean())

    # channel mismatch: campaign channels that do not exist as an order channel
    order_ch = set(o.channel.unique())
    unmatched = m[~m.channel.isin(order_ch)]
    v, bad = A.campaign_metrics(m)

    ev = [
        f"Campaigns report {ctx['conversions']:,} conversions, but the transaction table has only {ctx['orders']:,} orders "
        f"({ctx['conversions']/ctx['orders']:.0f}× more). Campaign rows carry no order ID, so conversions cannot be tied to sales.",
        f"Campaign spend {inr(spend)} = {spend/net:.0%} of completed net sales ({inr(net)}) and {spend/margin:.2f}× the margin proxy ({inr(margin)}).",
        f"Spend exceeded the margin proxy in {int(qs.over.sum())} of {len(qs)} quarters.",
        f"Across {len(W)} full weeks, weekly orders vs weekly spend: r = {r:.2f} (95% CI {lo:.2f} to {hi:.2f}), permutation p = {p:.2f} → no association detected.",
        f"Top-spend-quartile weeks averaged {q_hi.spend.mean():,.0f} spend vs {q_lo.spend.mean():,.0f} in the lowest quartile "
        f"({q_hi.spend.mean()/q_lo.spend.mean():.1f}×), yet orders were {q_hi.orders.mean():.0f} vs {q_lo.orders.mean():.0f} "
        f"(difference {diff:+.1f}, 95% CI {dlo:+.1f} to {dhi:+.1f}).",
        f"{len(unmatched)} campaigns ({inr(unmatched.spend.sum())}) run on channels ({', '.join(sorted(unmatched.channel.unique()))}) "
        f"that do not exist as an order channel in transactions.",
        f"{len(bad)} campaign rows have impossible funnels (clicks > impressions or conversions > clicks), {inr(bad.spend.sum())} of spend.",
    ]
    checks = [
        _check("Fields support the calculation", 0.5, "Spend, conversions and margin proxy exist, but no campaign→order key: only date/category/channel-level comparison."),
        _check("Pattern persists over time", persistence, f"Spend > margin proxy in {int(qs.over.sum())}/{len(qs)} quarters."),
        _check("Statistical support", 0.5, f"No association found (p={p:.2f}), but with {len(W)} weeks the test cannot rule out a modest effect (CI up to {hi:.2f}); absence of evidence is not proof."),
    ]
    exposure = max(0.0, spend - margin)
    card = dict(
        id="advertising", name="Advertising trap", status="Flagged",
        headline_label="What looks successful?",
        headline=f"{ctx['conversions']:,} conversions reported by campaigns",
        headline_sub=f"{v.conversions.sum()/v.clicks.sum():.0%} of clicks convert on valid campaign rows",
        concern_label="What may be wrong?",
        concern=f"Only {ctx['orders']:,} orders exist; spend is {spend/margin:.2f}× the margin proxy",
        concern_sub="Weekly orders do not move with weekly spend",
        evidence=ev,
        impact_value=exposure, impact_label="Proxy",
        impact_text=f"{inr(exposure)} - campaign spend above the margin proxy ({inr(spend)} vs {inr(margin)}). A comparison, not a proven loss: "
                    f"some orders may be marketing-driven but cannot be attributed.",
        assumptions=["Margin proxy = unit_price − unit_cost on Completed orders; costs were generated independently (README).",
                     "All campaign spend is compared with all completed-order margin; no attribution is possible with the supplied fields.",
                     "Orders are the outcome used; brand effects or offline sales outside this dataset are not captured."],
        confidence="Medium",
        uncertainty=["Observational data: correlation (or its absence) does not prove cause.",
                     "No campaign→order identifier, so the link is only by week/category/channel.",
                     "Synthetic data: relationships may be random by construction."],
        next_step="Add order IDs / UTM attribution to campaign tracking, then compare cost per attributed completed order with the margin per order "
                  f"(≈ {inr(comp.margin.mean())}). Pause or cap the lowest-efficiency campaigns for 2–4 weeks as a holdout test.",
        persistence=persistence, checks=checks,
        rank_breakdown=_score(exposure, net, persistence, checks),
        data={"weekly": W, "quarters": qs.reset_index().assign(q=lambda d: d.q.astype(str)),
              "valid_campaigns": v, "unmatched_spend": float(unmatched.spend.sum())},
    )
    return card


# ------------------------------------------------------------------ B. DISCOUNT TRAP
def discount_trap(raw, o) -> dict:
    comp = o[o.status == "Completed"]
    ctx = context(raw, o)
    net = ctx["net_sales"]
    d = o.groupby("discount_pct").quantity.agg(["mean", "std", "count"]).reset_index()
    d["ci"] = 1.96 * d["std"] / np.sqrt(d["count"])
    high = o[o.discount_pct >= 0.15].quantity
    none = o[o.discount_pct == 0].quantity
    diff, (lo, hi) = boot_mean_diff_ci(high, none)
    pp = o.groupby("product_id").agg(n=("quantity", "size"), disc=("discount_pct", "mean"))
    r, p = perm_corr_p(pp.disc, pp.n)

    oq = o.assign(q=o.timestamp.dt.to_period("Q").astype(str))
    qd = pd.DataFrame([{"q": q, "lift": g[g.discount_pct >= .15].quantity.mean() - g[g.discount_pct == 0].quantity.mean()} for q, g in oq.groupby("q")])
    persistence = float((qd.lift < 0.15).mean())          # "no meaningful lift" (<0.15 units, i.e. <5%) in each quarter

    given_all = float(comp.discount_amt.sum())
    given_deep = float(comp[comp.discount_pct >= 0.15].discount_amt.sum())
    below_cost_disc = int((comp.below_cost & (comp.discount_pct > 0)).sum())
    lift_upper = hi / none.mean()

    ev = [
        f"Discounts given on completed orders: {inr(given_all)} ({given_all/comp.list_value.sum():.1%} of list value); "
        f"{inr(given_deep)} of it at 15–25% depth.",
        f"Units per order: {none.mean():.2f} at 0% discount vs {high.mean():.2f} at 15–25% (difference {diff:+.2f}, 95% CI {lo:+.2f} to {hi:+.2f}). "
        f"Even the upper bound is only +{lift_upper:.1%} volume per order.",
        f"Across {len(pp)} products, orders vs average discount depth: r = {r:.2f}, permutation p = {p:.2f} → no association.",
        f"'No meaningful volume lift' held in {int((qd.lift < 0.15).sum())} of {len(qd)} quarters.",
        f"{below_cost_disc:,} discounted completed orders were already priced below unit cost.",
    ]
    checks = [
        _check("Fields support the calculation", 1.0, "discount_pct, quantity, unit_price, list_price present on every order; unit_price = list_price × (1 − discount) on all rows."),
        _check("Pattern persists over time", persistence, f"No meaningful lift in {int((qd.lift < 0.15).sum())}/{len(qd)} quarters."),
        _check("Statistical support", 1.0 if hi < 0.15 else 0.5, f"CI for lift ({lo:+.2f}, {hi:+.2f}) excludes a ≥0.15-unit lift."),
    ]
    card = dict(
        id="discount", name="Discount trap", status="Flagged",
        headline_label="What looks successful?",
        headline=f"{o.quantity.sum():,} units sold; {ctx['completed']:,} completed orders",
        headline_sub=f"Discounts are offered on {(o.discount_pct > 0).mean():.0%} of orders",
        concern_label="What may be wrong?",
        concern=f"Deeper discounts did not bring bigger baskets ({diff:+.2f} units/order)",
        concern_sub=f"{inr(given_deep)} given away at 15–25% depth",
        evidence=ev,
        impact_value=given_deep, impact_label="Observed",
        impact_text=f"{inr(given_deep)} of discount value on completed orders at 15–25% depth (observed). Whether it is a loss depends on "
                    f"whether those customers would have bought anyway, which this data cannot prove.",
        assumptions=["Discount value = (list_price − unit_price) × quantity.",
                     "Volume lift measured as units per order at deep vs no discount; order counts and customer traffic are not experimentally controlled.",
                     "Retention or brand benefits of discounting are not measured in the data."],
        confidence="High on the arithmetic, Medium on the interpretation",
        uncertainty=["Observational, not an A/B test: customers were not randomly assigned to discounts.",
                     "Basket size is only one possible payoff; repeat purchase is not tested here."],
        next_step="Run a 2-week A/B test in one category: cap discounts at 10% for half of the traffic and compare orders, basket size and net sales. "
                  "Stop discounting products that already sell below cost.",
        persistence=persistence, checks=checks,
        rank_breakdown=_score(given_deep, net, persistence, checks),
        data={"by_depth": d, "quarter_lift": qd},
    )
    return card


# ------------------------------------------------------------------ C. RETURNS TRAP (tested)
def returns_trap(raw, o) -> dict:
    from statistics import NormalDist
    h = np.where(o.timestamp.dt.month <= 6, "H1", "H2")
    g = o.assign(h=h).groupby(["product_id", "h"]).agg(n=("is_bad", "size"), bad=("is_bad", "mean"), net=("booked", "sum")).unstack()
    g.columns = ["_".join(c) for c in g.columns]
    p0 = o.is_bad.mean()
    g["delta"] = g.bad_H2 - g.bad_H1
    g["z"] = g.delta / np.sqrt(p0 * (1 - p0) * (1 / g.n_H1 + 1 / g.n_H2))
    g["net_growth"] = g.net_H2 / g.net_H1 - 1
    thr = NormalDist().inv_cdf(1 - 0.05 / len(g) / 2)       # Bonferroni across all products (two-sided)
    cand = g[g.net_growth > 0].sort_values("z", ascending=False)
    n_sig = int((g.z.abs() >= thr).sum())
    mo = o.groupby("month").is_bad.mean()
    top = cand.head(5)
    ev = [f"Compared H1 vs H2 cancel+return rate for {len(g)} products; the threshold for 'beyond chance' (Bonferroni) is |z| ≥ {thr:.2f}.",
          f"Products beyond chance: {n_sig}. Largest rise among products with growing sales: {top.index[0]} (z = {top.z.iloc[0]:.2f}, "
          f"{top.bad_H1.iloc[0]:.0%} → {top.bad_H2.iloc[0]:.0%}), which is what chance alone would produce across {len(g)} products.",
          f"Overall cancel+return rate is {p0:.1%}; month to month it ranges {mo.min():.1%} to {mo.max():.1%}."]
    return dict(id="returns", name="Returns trap", status="Tested - not supported",
                headline_label="What looks successful?", headline="Sales growing on some products",
                headline_sub="", concern_label="What may be wrong?", concern="Rising returns/cancellations on those products",
                concern_sub="No product's change is beyond chance",
                evidence=ev, impact_value=0.0, impact_label="Not estimated", impact_text="No exposure estimated: the pattern is not distinguishable from noise.",
                assumptions=["Cancelled and Returned are both counted as 'bad'; Pending is kept as not-bad."],
                confidence="Medium", uncertainty=["~70 orders per product per year gives wide noise bands; small real effects could be missed."],
                next_step="Keep monitoring monthly; start recording return reasons so the next review can separate quality, delivery and fit issues.",
                persistence=0.0, checks=[], rank_breakdown=None, data={"products": g.reset_index(), "threshold": thr, "n_sig": n_sig})


# ------------------------------------------------------------------ D. BESTSELLER TRAP (tested)
def bestseller_trap(raw, o) -> dict:
    comp = o[o.status == "Completed"]
    pr = comp.groupby("product_id").agg(net=("booked", "sum"), margin=("margin", "sum")).join(raw["products"].set_index("product_id"))
    pr = pr.sort_values("net", ascending=False)
    top = pr.head(50)
    neg_total = int((pr.margin < 0).sum())
    neg_top = int((top.margin < 0).sum())
    ev = [f"Top 50 products by completed net sales ({top.net.sum()/pr.net.sum():.0%} of net sales): {neg_top} have a negative margin proxy.",
          f"{neg_total} of {len(pr)} sold products have a negative margin proxy, but they are low sellers: only {int((pr.head(100).margin < 0).sum())} sit in the top 100 by sales.",
          f"The top-50 margin proxy is {inr(top.margin.sum())} ({top.margin.sum()/top.net.sum():.0%} of their net sales).",
          f"Caution: {int((raw['products'].unit_cost > raw['products'].list_price).sum())} of {len(raw['products'])} products have unit_cost above list price, which looks like a cost-data error (README warns about this), so margin-based claims are weak either way."]
    return dict(id="bestseller", name="Bestseller trap", status="Tested - not supported",
                headline_label="What looks successful?", headline="Highest-revenue products", headline_sub="",
                concern_label="What may be wrong?", concern="They may lose money after discounts and costs",
                concern_sub="None of the top 50 do on this data",
                evidence=ev, impact_value=0.0, impact_label="Not estimated", impact_text="Not supported: bestsellers keep a positive margin proxy.",
                assumptions=["Margin proxy uses unit_cost as supplied."], confidence="Low-Medium (cost data unreliable)",
                uncertainty=["If unit_cost is wrong, this conclusion could change."],
                next_step="Ask finance to validate unit_cost for the 85 negative-margin products; this is a data check, not a bestseller problem.",
                persistence=0.0, checks=[], rank_breakdown=None, data={"products": pr.reset_index()})


# ------------------------------------------------------------------ E. STOCKOUT TRAP (tested)
def stockout_trap(raw, o) -> dict:
    inv = raw["inventory"].copy()
    inv["month"] = inv.date.dt.to_period("M").astype(str)
    f = inv.groupby(["product_id", "month"]).stockout_flag.max().reset_index()
    x = o.groupby(["product_id", "month"]).size().rename("n").reset_index()
    y = f.merge(x, how="left", on=["product_id", "month"]).fillna({"n": 0})
    a, b = y[y.stockout_flag == 1].n, y[y.stockout_flag == 0].n
    diff, (lo, hi) = boot_mean_diff_ci(a, b)
    mism = int(((inv.stockout_flag == 1) & (inv.stock_on_hand > 0)).sum())
    ev = [f"Product-months with a stockout flag: {a.mean():.2f} orders vs {b.mean():.2f} without (difference {diff:+.2f}, 95% CI {lo:+.2f} to {hi:+.2f}) → no visible sales dip.",
          f"The stockout_flag is unreliable: {mism} of {int(inv.stockout_flag.sum())} flagged rows still show stock on hand > 0.",
          "Inventory is ~20 snapshots per product across the year, so exact stockout timing cannot be reconstructed; no missed-sales estimate is attempted."]
    return dict(id="stockout", name="Stockout trap", status="Tested - not supported",
                headline_label="What looks successful?", headline="Strong demand on some products", headline_sub="",
                concern_label="What may be wrong?", concern="Limited availability may be capping sales",
                concern_sub="Flag does not match stock levels",
                evidence=ev, impact_value=0.0, impact_label="Not estimated",
                impact_text="Not estimated: timing and flag quality do not allow a defensible missed-sales figure.",
                assumptions=["A flagged month is treated as 'possible stockout', not a proven lost sale."], confidence="Low",
                uncertainty=["Sparse snapshots; flag contradicts stock_on_hand."],
                next_step="Fix stockout logging (daily stock + zero-stock events) before attempting any lost-sales estimate.",
                persistence=0.0, checks=[], rank_breakdown=None, data={"by_flag": y.groupby("stockout_flag").n.mean().reset_index()})


# ------------------------------------------------------------------ assemble
def build_cards(raw, o) -> list[dict]:
    cards = [advertising_trap(raw, o), discount_trap(raw, o), returns_trap(raw, o), bestseller_trap(raw, o), stockout_trap(raw, o)]
    flagged = sorted([c for c in cards if c["status"] == "Flagged"], key=lambda c: -c["rank_breakdown"]["total"])
    rest = [c for c in cards if c["status"] != "Flagged"]
    for i, c in enumerate(flagged, 1):
        c["rank"] = i
    for c in rest:
        c["rank"] = None
    return flagged + rest


# ------------------------------------------------------------------ rule-based unusual product-quarters (no ML)
def rule_flags(o: pd.DataFrame, min_orders: int = 8, top: int = 15):
    """Transparent rules, thresholds learned only from Q1-Q2 and applied to Q3-Q4 product-quarter windows."""
    x = o.assign(q=o.timestamp.dt.to_period("Q").astype(str))
    rows = []
    for (pid, q), g in x.groupby(["product_id", "q"]):
        c = g[g.status == "Completed"]
        net = c.booked.sum()
        rows.append(dict(product_id=pid, q=q, orders=len(g), bad_rate=g.is_bad.mean(),
                         disc_intensity=g.discount_amt.sum() / g.list_value.sum(),
                         margin_pct=(c.margin.sum() / net) if net > 0 else np.nan))
    w = pd.DataFrame(rows)
    w = w[w.orders >= min_orders]
    train, test = w[w.q.isin(["2025Q1", "2025Q2"])], w[w.q.isin(["2025Q3", "2025Q4"])].copy()
    thr = {"bad_rate": float(train.bad_rate.quantile(.90)), "disc_intensity": float(train.disc_intensity.quantile(.90))}
    test["rule_bad"] = test.bad_rate >= thr["bad_rate"]
    test["rule_disc"] = test.disc_intensity >= thr["disc_intensity"]
    test["rule_margin"] = test.margin_pct < 0
    test["rules_triggered"] = test[["rule_bad", "rule_disc", "rule_margin"]].sum(axis=1)
    def why(r):
        parts = []
        if r.rule_bad: parts.append(f"cancel+return {r.bad_rate:.0%} (≥{thr['bad_rate']:.0%})")
        if r.rule_disc: parts.append(f"discount {r.disc_intensity:.1%} (≥{thr['disc_intensity']:.1%})")
        if r.rule_margin: parts.append(f"margin proxy {r.margin_pct:.0%} (<0)")
        return "; ".join(parts)
    test["why"] = test.apply(why, axis=1)
    summary = {"windows_train": int(len(train)), "windows_test": int(len(test)),
               "flagged": int((test.rules_triggered > 0).sum()), "thresholds": thr}
    out = test[test.rules_triggered > 0].sort_values(["rules_triggered", "orders"], ascending=False).head(top)
    return out, summary
