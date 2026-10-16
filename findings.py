"""
MIRAGE content layer: turns trap cards into investigations with explicit Observed / Interpretation / Unknown / Recommendation
sections, a structured evidence ledger (source, fields, comparison, rule, limit, records), and data-health checks.
No new statistics here - every number is read from the computed cards or recomputed directly from the CSVs.
"""
from __future__ import annotations
import pandas as pd
import analysis as A
import traps as T

PERIOD = "Jan–Dec 2025 (full dataset)"
SOURCES = {"advertising": ["campaigns.csv", "transactions.csv", "products.csv"], "discount": ["transactions.csv", "products.csv"],
           "returns": ["transactions.csv"], "bestseller": ["transactions.csv", "products.csv"], "stockout": ["inventory.csv", "transactions.csv"]}


def _li(label, source, fields, comparison, value, rule, supports, limit, records=None):
    return dict(label=label, source=source, fields=fields, comparison=comparison, value=value, rule=rule, supports=supports, limit=limit, records=records)


def enrich(cards: list[dict], raw: dict, o: pd.DataFrame) -> list[dict]:
    ctx = T.context(raw, o)
    comp = o[o.status == "Completed"]
    m = raw["campaigns"]
    prod = raw["products"]
    n_bad_cost = int((prod.unit_cost > prod.list_price).sum())
    by = {c["id"]: c for c in cards}

    # ---------------- advertising
    c = by["advertising"]; d = c["data"]; W = d["weekly"]; q = d["quarters"]
    r, p = T.perm_corr_p(W.spend, W.orders); lo, hi = T.boot_corr_ci(W.spend, W.orders)
    unm = m[~m.channel.isin(set(o.channel.unique()))]
    inv_c = m[~A.valid_campaign_mask(m)]
    spend, margin = ctx["campaign_spend"], ctx["margin_proxy"]
    c.update(
        category="Campaigns", sources=SOURCES["advertising"], period=PERIOD,
        finding_title="Campaigns report strong conversion, yet orders have not moved with ad spend",
        summary=f"Campaigns report {ctx['conversions']:,} conversions against {ctx['orders']:,} orders, and spend is {spend/margin:.2f}× the margin proxy.",
        why_it_matters="If campaign spend cannot be connected to completed orders, a business may keep paying for activity that looks productive in the campaign tool but does not show up in sales. "
                       "It also changes how headline sales should be read: a large share of the margin proxy is absorbed by marketing spend.",
        interpretation="Possible explanations include missing attribution (campaign rows carry no order ID), campaigns running on channels with no matching sales channel, "
                       "conversions that are not purchases, or genuinely low-impact spend. The data cannot tell these apart.",
        unknown=c["uncertainty"] + ["Whether a campaign 'conversion' means a purchase, a sign-up or another event.", "Whether sales happen outside this dataset (for example offline)."],
        confirm=["Confirm what a 'conversion' means in the campaign tool and whether it can be joined to an order ID.",
                 "Repeat the weekly spend-versus-orders comparison using attributed orders only.",
                 f"Check unit_cost for the {n_bad_cost} products priced below cost before relying on the margin proxy."],
        ledger=[
            _li("Reported conversions vs recorded orders", "campaigns.csv · transactions.csv", "conversions; transaction_id (count)", "Whole period, no baseline",
                f"{ctx['conversions']:,} vs {ctx['orders']:,} ({ctx['conversions']/ctx['orders']:.0f}×)", "Σ conversions compared with the number of orders",
                "Campaign conversions cannot be reconciled with the orders table.", "A conversion may legitimately be something other than an order.", "campaigns_all"),
            _li("Campaign spend vs margin proxy", "campaigns.csv · transactions.csv · products.csv", "spend; unit_price, unit_cost, quantity, status", "Each quarter of 2025",
                f"{T.inr(spend)} vs {T.inr(margin)} ({spend/margin:.2f}×); above the proxy in {int(q.over.sum())} of {len(q)} quarters",
                "Σ spend vs Σ (unit_price − unit_cost) × quantity on Completed orders", "Spend is large relative to the margin proxy.",
                "The proxy depends on unit_cost, which the dataset README says was generated independently. It is not profit.", "quarterly"),
            _li("Weekly spend vs weekly orders", "campaigns.csv · transactions.csv", "date, spend; timestamp, transaction_id", f"{len(W)} full weeks (partial edge weeks excluded)",
                f"r = {r:.2f} (95% CI {lo:.2f} to {hi:.2f}); permutation p = {p:.2f}", "Pearson correlation, 5,000-shuffle permutation test, bootstrap interval",
                "No association between spend and orders was detected.", f"With {len(W)} weeks a modest effect cannot be ruled out; absence of evidence is not proof.", "weekly"),
            _li("Campaigns on channels with no sales channel", "campaigns.csv · transactions.csv", "campaigns.channel; transactions.channel", "Whole period",
                f"{len(unm):,} campaigns · {T.inr(unm.spend.sum())} on {', '.join(sorted(unm.channel.unique()))}", "Campaign channels not present in transactions.channel",
                "Part of the spend cannot be linked to any order channel.", "Channel names may simply be recorded differently.", "unmatched_campaigns"),
            _li("Campaign rows with impossible funnels", "campaigns.csv", "impressions, clicks, conversions", "Whole period",
                f"{len(inv_c)} rows · {T.inr(inv_c.spend.sum())} spend", "clicks > impressions or conversions > clicks",
                "Part of the campaign table cannot be trusted as performance data.", "May reflect logging errors rather than real performance.", "invalid_campaigns"),
        ])

    # ---------------- discount
    c = by["discount"]
    high = o[o.discount_pct >= .15].quantity; none = o[o.discount_pct == 0].quantity
    diff, (dlo, dhi) = T.boot_mean_diff_ci(high, none)
    given_all = float(comp.discount_amt.sum()); given_deep = float(comp[comp.discount_pct >= .15].discount_amt.sum())
    pp = o.groupby("product_id").agg(n=("quantity", "size"), disc=("discount_pct", "mean")); pr_, pp_ = T.perm_corr_p(pp.disc, pp.n)
    bc = comp[comp.below_cost & (comp.discount_pct > 0)]
    c.update(
        category="Discounting", sources=SOURCES["discount"], period=PERIOD,
        finding_title="Deeper discounts have not produced larger baskets",
        summary=f"Orders at 15–25% discount average {high.mean():.2f} units against {none.mean():.2f} at no discount, and {T.inr(given_deep)} of discount value was given at that depth.",
        why_it_matters="Discounts lower revenue per unit. If they do not lift volume, they reduce net sales without a visible offsetting benefit. "
                       "If they build loyalty or repeat purchase, that benefit is not captured in this data.",
        interpretation="The pattern is consistent with discounts being applied regardless of demand, for example in blanket promotions. "
                       "It does not show that discounting reduced margin: cost data is unreliable and no experiment was run.",
        unknown=c["uncertainty"] + ["Repeat purchase and retention effects are not measured.", "Whether discount depth is set by a promotion calendar, by product, or ad hoc."],
        confirm=["Run a controlled test in one category (for example cap discounts at 10% for half of traffic) and compare orders and net sales.",
                 "Validate unit_cost for discounted products before estimating any margin impact.",
                 "Find out how discount depth is assigned today."],
        ledger=[
            _li("Discount value given", "transactions.csv · products.csv", "list_price, unit_price, quantity, discount_pct, status", "Completed orders, whole period",
                f"{T.inr(given_all)} ({given_all/comp.list_value.sum():.1%} of list value); {T.inr(given_deep)} at 15–25% depth", "Σ (list_price − unit_price) × quantity",
                "Shows how much revenue was forgone through discounts.", "Forgone revenue is not a proven loss: customers might have bought anyway or not at all.", "deep_discount_orders"),
            _li("Units per order by discount depth", "transactions.csv", "quantity, discount_pct", "15–25% discount vs no discount",
                f"{high.mean():.2f} vs {none.mean():.2f} units (difference {diff:+.2f}, 95% CI {dlo:+.2f} to {dhi:+.2f})", "Bootstrap interval for the difference in mean units per order",
                "No visible volume lift at deeper discounts.", "Observational: customers were not randomly assigned to discount levels.", "by_depth"),
            _li("Products with deeper average discounts", "transactions.csv", "product_id, discount_pct", f"{len(pp)} products",
                f"r = {pr_:.2f}; permutation p = {pp_:.2f}", "Correlation of orders per product with average discount depth",
                "Products discounted more heavily do not sell more orders.", "Product popularity and discount policy may be related in ways the data does not show.", "product_discount"),
            _li("Discounted orders already below cost", "transactions.csv · products.csv", "unit_price, unit_cost, discount_pct", "Completed orders",
                f"{len(bc):,} orders", "unit_price < unit_cost and discount_pct > 0", "Some discounts were applied to items that already had no margin proxy.",
                "unit_cost may be wrong for these products.", "below_cost_discounted"),
        ])

    # ---------------- returns
    c = by["returns"]; g = c["data"]["products"]
    c.update(
        category="Returns", sources=SOURCES["returns"], period="H1 vs H2 2025",
        finding_title="Returns and cancellations: no product stands out beyond chance",
        summary=c["evidence"][1],
        why_it_matters="A product whose returns rise while its sales grow can make growth less valuable. It is worth checking, but only if the rise is larger than normal variation.",
        interpretation="The largest increases are about the size random variation would produce across this many products. This is a review with no material signal, not proof that nothing is wrong.",
        unknown=c["uncertainty"] + ["Reasons for returns and cancellations are not recorded."],
        confirm=["Start recording a return reason on each returned or cancelled order.", "Repeat this check monthly with the same threshold."],
        ledger=[_li("Change in return+cancel rate per product", "transactions.csv", "product_id, status, timestamp", "H1 vs H2 2025",
                    f"{c['data']['n_sig']} of {len(g)} products beyond chance (|z| ≥ {c['data']['threshold']:.2f})", "Two-period z-test per product with Bonferroni correction",
                    "No product shows a change larger than chance.", "About 70 orders per product per year gives wide noise bands.", "returns_products")])

    # ---------------- bestseller
    c = by["bestseller"]; pr = c["data"]["products"]; top = pr.head(50)
    c.update(
        category="Products", sources=SOURCES["bestseller"], period=PERIOD,
        finding_title="Top-selling products keep a positive margin proxy",
        summary=c["evidence"][0],
        why_it_matters="A best seller that loses money after discounts and costs would make headline revenue misleading.",
        interpretation="On this data the loss-making products are low sellers. Because unit_cost looks unreliable for some products, this is a weak all-clear rather than a clean bill of health.",
        unknown=c["uncertainty"] + ["True product costs."],
        confirm=[f"Ask finance to validate unit_cost for the {int((pr.margin < 0).sum())} products with a negative margin proxy."],
        ledger=[_li("Margin proxy of top 50 products by net sales", "transactions.csv · products.csv", "unit_price, unit_cost, quantity, status", "Completed orders",
                    f"{int((top.margin < 0).sum())} of 50 negative; top-50 margin proxy {T.inr(top.margin.sum())}", "Σ (unit_price − unit_cost) × quantity per product",
                    "Best sellers are not losing money on this data.", "unit_cost is unreliable for some products.", "top_products")])

    # ---------------- stockout
    c = by["stockout"]
    c.update(
        category="Inventory", sources=SOURCES["stockout"], period=PERIOD,
        finding_title="Stockout flags do not line up with stock levels or sales",
        summary=c["evidence"][0],
        why_it_matters="If stock availability caps sales, observed sales understate real demand. That could only be estimated with reliable stock timing.",
        interpretation="The stockout flag contradicts the stock-on-hand column, so no lost-sales estimate is defensible.",
        unknown=c["uncertainty"],
        confirm=["Log daily stock and zero-stock events before attempting a lost-sales estimate."],
        ledger=[_li("Orders in flagged vs unflagged product-months", "inventory.csv · transactions.csv", "stockout_flag, product_id, date; transaction_id", "Product-months",
                    c["evidence"][0].split("→")[0].strip(), "Bootstrap interval for the difference in average orders", "No visible dip in orders in flagged months.",
                    "Snapshots are sparse (about 20 per product per year).", "stock_by_flag"),
                _li("Flag vs stock on hand", "inventory.csv", "stockout_flag, stock_on_hand", "Whole period",
                    c["evidence"][1].split(":")[1].strip().rstrip("."), "stockout_flag = 1 while stock_on_hand > 0", "The flag cannot be trusted as a stockout signal.", "Flag may mean something other than zero stock.", "stock_flag_rows")])
    return cards


def add_ml(cards: list[dict], mlres: dict) -> list[dict]:
    """Append the local ML models' measured results to the matching investigations' evidence ledgers."""
    by = {c["id"]: c for c in cards}
    b, u = mlres["baseline"], mlres["uplift"]
    adv, dis = by["advertising"], by["discount"]
    adv["data"]["ml_weekly"] = b["table"]
    verdict = ("The model is no better than a plain average, so spend, discount and season do not predict weekly orders here."
               if not b["beats_naive"] else "The model beats a plain average, so spend/discount/season carry some predictive signal.")
    adv["ledger"].append(_li(
        "Gradient-boosting baseline: weekly orders from spend, discount and season", "campaigns.csv · transactions.csv",
        "spend, average discount, week of year → completed orders", f"Trained on first {b['train_weeks']} weeks, tested on last {b['test_weeks']}",
        f"error {b['mae_model']:.1f} orders/week vs {b['mae_naive']:.1f} for a plain average; R² {b['r2']:.2f}; spend importance {b['spend_importance']:+.2f} ± {b['spend_importance_sd']:.2f}",
        "HistGradientBoostingRegressor, permutation importance (200 repeats), fixed seed", verdict,
        "Only 51 weeks, so the model is noisy; a weak result is not proof that spend has zero effect.", "ml_weekly"))
    lo, hi = u["ci"]
    pct = u["mean_uplift"] / u["control_mean"]
    dis["ledger"].append(_li(
        "Uplift model (T-learner): extra units per order with a 15–25% discount", "transactions.csv · products.csv",
        "category, list price, region, segment, channel, payment, month → quantity", f"{u['n_treated']:,} discounted vs {u['n_control']:,} undiscounted completed orders",
        f"{u['mean_uplift']:+.3f} units per order ({pct:+.1%} of the {u['control_mean']:.2f} baseline); 95% interval {lo:+.3f} to {hi:+.3f}; raw difference {u['naive_diff']:+.3f}",
        "Two gradient-boosting models (discounted / not), predicted on a held-out half of orders, bootstrap interval",
        "Estimated extra units from a deep discount are essentially zero.", "Observational data: discounts were not randomly assigned, so this is association, not proven cause.", None))
    return cards


# ---------------------------------------------------------------- records behind evidence items
LABELS = {"transaction_id": "Order", "timestamp": "Time", "product_id": "Product", "quantity": "Quantity", "list_price": "List price", "unit_price": "Unit price",
          "unit_cost": "Unit cost", "discount_pct": "Discount", "status": "Status", "campaign_id": "Campaign", "date": "Date", "channel": "Channel", "category": "Category",
          "spend": "Spend", "impressions": "Impressions", "clicks": "Clicks", "conversions": "Conversions", "orders": "Orders", "net": "Net sales", "week": "Week week-start",
          "q": "Quarter", "margin": "Margin proxy", "over": "Spend above margin proxy", "inventory_record_id": "Record", "stock_on_hand": "Stock on hand",
          "stockout_flag": "Stockout flag", "reorder_threshold": "Reorder threshold", "avg_discount": "Average discount", "mean": "Mean units per order", "count": "Orders"}


def get_records(key: str, raw: dict, o: pd.DataFrame, cards: list[dict]):
    """Returns (DataFrame, description). Customer identifiers are never included."""
    by = {c["id"]: c for c in cards}
    m, inv = raw["campaigns"], raw["inventory"]
    comp = o[o.status == "Completed"]
    tcols = ["transaction_id", "timestamp", "product_id", "quantity", "list_price", "unit_price", "discount_pct", "status"]
    if key == "campaigns_all":
        return m[["campaign_id", "date", "channel", "category", "spend", "impressions", "clicks", "conversions"]], "All campaign rows (campaigns.csv)"
    if key == "quarterly":
        return by["advertising"]["data"]["quarters"], "Quarterly spend vs margin proxy (aggregated)"
    if key == "ml_weekly":
        return by["advertising"]["data"]["ml_weekly"], "Weekly table used by the baseline model (train/test split)"
    if key == "weekly":
        return by["advertising"]["data"]["weekly"][["week", "orders", "net", "spend", "conversions", "clicks"]], "Full weeks used in the correlation test"
    if key == "unmatched_campaigns":
        return m[~m.channel.isin(set(o.channel.unique()))][["campaign_id", "date", "channel", "category", "spend", "conversions"]], "Campaigns on channels not present in transactions"
    if key == "invalid_campaigns":
        return m[~A.valid_campaign_mask(m)][["campaign_id", "date", "channel", "category", "spend", "impressions", "clicks", "conversions"]], "Rows where clicks > impressions or conversions > clicks"
    if key == "deep_discount_orders":
        return comp[comp.discount_pct >= .15][tcols], "Completed orders with a 15–25% discount"
    if key == "by_depth":
        return by["discount"]["data"]["by_depth"][["discount_pct", "mean", "std", "count"]], "Units per order by discount depth (all orders)"
    if key == "product_discount":
        return o.groupby("product_id").agg(orders=("quantity", "size"), avg_discount=("discount_pct", "mean")).reset_index(), "Orders and average discount per product"
    if key == "below_cost_discounted":
        return comp[comp.below_cost & (comp.discount_pct > 0)][tcols + ["unit_cost"]], "Completed discounted orders priced below unit cost"
    if key == "returns_products":
        g = by["returns"]["data"]["products"]
        return g[["product_id", "n_H1", "n_H2", "bad_H1", "bad_H2", "z"]], "Return+cancel rate per product, H1 vs H2 (z-score)"
    if key == "top_products":
        return by["bestseller"]["data"]["products"].head(50)[["product_id", "category", "net", "margin", "unit_cost", "list_price"]], "Top 50 products by completed net sales"
    if key == "stock_flag_rows":
        return inv[(inv.stockout_flag == 1) & (inv.stock_on_hand > 0)][["inventory_record_id", "product_id", "date", "stock_on_hand", "reorder_threshold", "stockout_flag"]], "Rows flagged as stockout while stock on hand > 0"
    if key == "stock_by_flag":
        return by["stockout"]["data"]["by_flag"], "Average orders per product-month by stockout flag"
    return pd.DataFrame(), "No records"


# ---------------------------------------------------------------- data health (defined checks, honest results)
def source_health(raw: dict, o: pd.DataFrame) -> pd.DataFrame:
    c, p, t, i, m = raw["customers"], raw["products"], raw["transactions"], raw["inventory"], raw["campaigns"]
    def miss(d): return int(d.isna().sum().sum())
    def rng(s): return f"{pd.to_datetime(s).min():%d %b %Y} – {pd.to_datetime(s).max():%d %b %Y}"
    rows = [
        ("customers.csv", len(c), c.shape[1], rng(c.signup_date) + " (signup)", [f"missing cells {miss(c)}", f"duplicate customer_id {int(c.customer_id.duplicated().sum())}"], 0),
        ("products.csv", len(p), p.shape[1], "n/a", [f"missing cells {miss(p)}", f"duplicate product_id {int(p.product_id.duplicated().sum())}",
                                                   f"{int((p.unit_cost > p.list_price).sum())} products with unit_cost > list_price"], int((p.unit_cost > p.list_price).sum() > 0)),
        ("transactions.csv", len(t), t.shape[1], rng(t.timestamp), [f"missing cells {miss(t)}", f"duplicate transaction_id {int(t.transaction_id.duplicated().sum())}",
                                                                   f"orphan customer_id {int((~t.customer_id.isin(c.customer_id)).sum())}", f"orphan product_id {int((~t.product_id.isin(p.product_id)).sum())}",
                                                                   f"invalid quantity/price {int(((t.quantity <= 0) | (t.unit_price <= 0)).sum())}",
                                                                   f"{int(o.before_signup.sum())} orders dated before the customer's signup"], int(o.before_signup.sum() > 0)),
        ("inventory.csv", len(i), i.shape[1], rng(i.date), [f"missing cells {miss(i)}", f"duplicate record id {int(i.inventory_record_id.duplicated().sum())}",
                                                           f"orphan product_id {int((~i.product_id.isin(p.product_id)).sum())}",
                                                           f"{int(((i.stockout_flag == 1) & (i.stock_on_hand > 0)).sum())} rows flagged stockout while stock > 0"], int(((i.stockout_flag == 1) & (i.stock_on_hand > 0)).any())),
        ("campaigns.csv", len(m), m.shape[1], rng(m.date), [f"missing cells {miss(m)}", f"duplicate campaign_id {int(m.campaign_id.duplicated().sum())}",
                                                           f"{int((~A.valid_campaign_mask(m)).sum())} rows with impossible funnels", "no order or campaign key to join with transactions"], 1),
    ]
    out = pd.DataFrame([dict(File=f, Rows=r, Columns=cc, Coverage=cov, Checks="; ".join(chk), Status="Review" if flag else "No issues in checks run") for f, r, cc, cov, chk, flag in rows])
    return out


def join_checks(raw: dict, o: pd.DataFrame) -> pd.DataFrame:
    c, p, t, i = raw["customers"], raw["products"], raw["transactions"], raw["inventory"]
    pct = lambda a, b: f"{a:.1%}"
    return pd.DataFrame([
        ("transactions → customers (customer_id)", pct(t.customer_id.isin(c.customer_id).mean(), 1), "All orders match a customer"),
        ("transactions → products (product_id)", pct(t.product_id.isin(p.product_id).mean(), 1), "All orders match a product"),
        ("inventory → products (product_id)", pct(i.product_id.isin(p.product_id).mean(), 1), "All inventory rows match a product"),
        ("campaigns → transactions", "No key", "No campaign or order identifier exists; only date, channel and category overlap, and Email/Social have no matching order channel"),
        ("Cost inputs for margin", f"{(p.unit_cost <= p.list_price).mean():.1%} plausible", "unit_cost exists for every product, but costs and prices were generated independently (README)"),
    ], columns=["Join / input", "Match / availability", "Note"])
