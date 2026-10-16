"""MIRAGE - evidence-led review of where reported performance and underlying evidence disagree."""
from __future__ import annotations
import datetime as dt, json, pathlib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import analysis as A, traps as T, findings as F, ml, llm, theme

st.set_page_config(page_title="mirage.", page_icon="◦", layout="wide", initial_sidebar_state="collapsed")
theme.plotly_template()
st.markdown(theme.CSS, unsafe_allow_html=True)
C = theme.C
ACT = pathlib.Path(__file__).parent / "outputs" / "actions.json"
STATUSES = ["Not started", "Planned", "Investigated", "Resolved"]
NAV = ["Overview", "Investigations", "Data"]


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Reading the five source files and fitting the local models…")
def load():
    raw, o = A.load_all()
    mlres = ml.run_all(raw, o)
    cards = F.add_ml(F.enrich(T.build_cards(raw, o), raw, o), mlres)
    return raw, o, cards, mlres

def load_actions() -> dict:
    try: return json.loads(ACT.read_text())
    except Exception: return {}

def on_status(tid):
    d = load_actions()
    d[tid] = {"status": st.session_state[f"status_{tid}"], "updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
    ACT.parent.mkdir(exist_ok=True); ACT.write_text(json.dumps(d, indent=2))

def status_of(tid): return load_actions().get(tid, {}).get("status", "Not started")

ss = st.session_state
ss.setdefault("nav", "Overview"); ss.setdefault("view", "list"); ss.setdefault("inv", None)

def go_nav(): ss.view = "list"
def open_inv(tid): ss.nav = "Investigations"; ss.view = "detail"; ss.inv = tid
def set_view(v): ss.view = v
def back(): ss.view = "list"

try:
    raw, o, cards, mlres = load(); ERR = None
except Exception as e:  # shown honestly, no fake data
    raw = o = cards = mlres = None; ERR = f"{type(e).__name__}: {e}"

# ------------------------------------------------------------------ top bar
a, b, c = st.columns([1.2, 2.6, 2.2], vertical_alignment="center")
a.markdown('<div class="wordmark">mirage<span>.</span></div>', unsafe_allow_html=True)
b.radio("Navigation", NAV, key="nav", horizontal=True, label_visibility="collapsed", on_change=go_nav)
if ERR: c.markdown('<div class="status"><span class="dot err"></span>Data could not be loaded</div>', unsafe_allow_html=True)
else:
    llm_ok, llm_name = llm.status()
    c.markdown(f'<div class="status"><span class="dot"></span>5 source files loaded · {len(raw["transactions"]):,} orders · {F.PERIOD}<br>local LLM: {llm_name if llm_ok else "off (template text)"}</div>', unsafe_allow_html=True)
st.divider()

if ERR:
    st.markdown('<div class="eyebrow">Data unavailable</div><div class="hero">The source files could not be read.</div>', unsafe_allow_html=True)
    st.error(ERR); st.caption(f"Expected the dataset folder at {A.DATA_DIR} (override with the MONEY_LEAK_DATA environment variable). Nothing is shown until real data loads.")
    st.stop()

by = {x["id"]: x for x in cards}
flagged = [x for x in cards if x["status"] == "Flagged"]
others = [x for x in cards if x["status"] != "Flagged"]

# ------------------------------------------------------------------ helpers
def fmt_pct(x): return f"{x:+.1%}"
def tag(txt, cls=""): return f'<span class="tag {cls}">{txt}</span>'

def kpis():
    comp = o[o.status == "Completed"]; mo = pd.to_datetime(comp.timestamp).dt.month; h1 = comp[mo <= 6]; h2 = comp[mo > 6]
    m = raw["campaigns"]; md = pd.to_datetime(m.date).dt.month
    items = [("Net sales", T.inr(comp.booked.sum()), h1.booked.sum(), h2.booked.sum(), "Observed"),
             ("Completed orders", f"{len(comp):,}", len(h1), len(h2), "Observed"),
             ("Margin proxy", T.inr(comp.margin.sum()), h1.margin.sum(), h2.margin.sum(), "Proxy"),
             ("Campaign spend", T.inr(m.spend.sum()), m[md <= 6].spend.sum(), m[md > 6].spend.sum(), "Observed")]
    cols = st.columns(4)
    for col, (l, v, p, n, kind) in zip(cols, items):
        col.markdown(f'<div class="card"><div class="kpi-l">{l} · {kind}</div><div class="kpi-v">{v}</div><div class="kpi-s">H2 vs H1: {fmt_pct(n/p-1)}</div></div>', unsafe_allow_html=True)

def chart(card):
    fig = go.Figure(); d = card["data"]; i = card["id"]
    if i == "advertising":
        w = d["weekly"]; fig.add_scatter(x=w.spend, y=w.orders, mode="markers", marker=dict(color=C["deep"], size=8), name="Week")
        fig.update_layout(xaxis_title="Weekly campaign spend", yaxis_title="Weekly orders")
        cap = "Each dot is one full week. If spend drove orders, dots would climb to the right."
    elif i == "discount":
        b_ = d["by_depth"]; fig.add_bar(x=[f"{x:.0%}" for x in b_.discount_pct], y=b_["mean"], marker_color=C["olive"])
        fig.update_layout(xaxis_title="Discount depth", yaxis_title="Average units per order")
        cap = "If deeper discounts built bigger baskets, bars would rise from left to right."
    elif i == "returns":
        z = d["products"].z; fig.add_histogram(x=z, nbinsx=30, marker_color=C["cool"])
        fig.add_vline(x=d["threshold"], line_dash="dot", line_color=C["red"]); fig.add_vline(x=-d["threshold"], line_dash="dot", line_color=C["red"])
        fig.update_layout(xaxis_title="Change in return+cancel rate, H1→H2 (z-score)", yaxis_title="Products")
        cap = "Dotted lines mark the Bonferroni-corrected threshold. No product crosses it."
    elif i == "bestseller":
        p = d["products"].head(50); fig.add_scatter(x=p.net, y=p.margin, mode="markers", marker=dict(color=C["green"], size=8))
        fig.add_hline(y=0, line_color=C["line"]); fig.update_layout(xaxis_title="Net sales (top 50 products)", yaxis_title="Margin proxy")
        cap = "Each dot is one of the 50 top-selling products. Below the line would mean negative margin proxy."
    else:
        s = d["by_flag"]; fig.add_bar(x=["Not flagged", "Flagged"], y=s.n, marker_color=C["cool"])
        fig.update_layout(yaxis_title="Product-months")
        cap = "Number of product-months in each group. The flag does not match stock on hand (see the evidence ledger)."
    fig.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig, width="stretch", key=f"chart_{i}", config={"displayModeBar": False}); st.caption(cap)

@st.dialog("Records behind this evidence", width="large")
def show_records(key):
    df, desc = F.get_records(key, raw, o, cards)
    st.caption(f"{desc} · {len(df):,} rows. Customer identifiers are excluded.")
    show = df.head(500).rename(columns=lambda x: F.LABELS.get(x, x))
    st.dataframe(show, width="stretch", hide_index=True)
    if len(df) > 500: st.caption("Showing the first 500 rows; the download contains all of them.")
    st.download_button("Download CSV", df.to_csv(index=False), file_name=f"{key}.csv", mime="text/csv")

RECO = {"advertising": ("Hold the next increase in campaign spend until conversions can be tied to orders.",
                        "Spend is large relative to the margin proxy and weekly orders show no detected link to it. Pausing growth in spend is low-regret while the tracking gap is closed.",
                        "Orders attributed to campaigns move with spend once attribution exists; or the campaign tool shows that conversions are not purchases."),
        "discount": ("Test a 10% discount cap on one category before changing policy.",
                     "Deeper discounts forgo revenue and show no visible lift in units per order. A controlled test separates cause from coincidence.",
                     "Net sales and orders in the capped group stay level with the control group.")}

def summary_md(card):
    r = RECO.get(card["id"])
    lines = [f"# {card['finding_title']}", f"*{card['category']} · {card['period']} · status: {status_of(card['id'])}*", "",
             "## What we observed (Observed)", *[f"- {e}" for e in card["evidence"]], "", "## Why it matters (Interpretation)", card["why_it_matters"], "",
             "## What the data does not tell us (Data limitation)", *[f"- {u}" for u in card["unknown"]], "", "## What to verify next", *[f"- {u}" for u in card["confirm"]]]
    if r: lines += ["", "## Recommended action", r[0]]
    lines += ["", "*Figures are Observed unless marked Proxy. Margin proxy = unit price − unit cost; not profit.*"]
    return "\n".join(lines)

def ai_box(card):
    k = f"ai_{card['id']}"
    if st.button("Plain-language summary (local LLM)", key=f"btn_{k}"):
        with st.spinner("Asking the local model…"): ss[k] = llm.explain(card)
    if k in ss:
        txt, src = ss[k]; st.markdown(f'<div class="card">{txt}</div>', unsafe_allow_html=True); st.caption(f"Source: {src}. Wording only; every figure comes from the data.")

# ------------------------------------------------------------------ OVERVIEW
def overview():
    st.markdown('<div class="eyebrow">Business overview</div><div class="hero">Looks good on paper.<br><em>Does it hold up?</em></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="lede">Reported performance for {F.PERIOD}, set against what the underlying records actually show. Comparisons are second half versus first half of 2025.</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Surface</div>', unsafe_allow_html=True); kpis()
    lead = flagged[0]
    st.markdown('<div class="eyebrow">Discrepancy · lead investigation</div>', unsafe_allow_html=True)
    s, u = st.columns(2)
    with s: st.markdown(f'<div class="card"><div class="col-h">Surface signal {tag("Observed","obs")}</div><div class="big">{lead["headline"]}</div><div class="small">{lead["headline_sub"]}</div></div>', unsafe_allow_html=True)
    with u: st.markdown(f'<div class="card"><div class="col-h">Underlying evidence {tag("Observed","obs")}</div><div class="big">{lead["concern"]}</div><div class="small">{lead["concern_sub"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f'**{lead["finding_title"]}**'); 
    for e in lead["evidence"][:3]: st.markdown(f"- {e}")
    st.button("Open investigation →", key="open_lead", type="primary", on_click=open_inv, args=(lead["id"],))
    st.markdown('<div class="eyebrow">Other signals</div>', unsafe_allow_html=True)
    for x in flagged[1:] + others:
        col1, col2, col3 = st.columns([5, 2, 1.2], vertical_alignment="center")
        col1.markdown(f'**{x["finding_title"]}**<br><span class="small">{x["category"]}</span>', unsafe_allow_html=True)
        col2.markdown(tag("Flagged", "flag") if x["status"] == "Flagged" else tag("Reviewed · no signal"), unsafe_allow_html=True)
        col3.button("Open →", key=f"ov_{x['id']}", on_click=open_inv, args=(x["id"],))
    st.markdown('<div class="eyebrow">Optional · local LLM</div>', unsafe_allow_html=True)
    if st.button("Write a short brief across all investigations", key="brief_btn"):
        with st.spinner("Asking the local model…"): ss["brief"] = llm.brief(cards)
    if "brief" in ss:
        t, src = ss["brief"]; st.markdown(f'<div class="card">{t}</div>', unsafe_allow_html=True); st.caption(f"Source: {src}. Wording only; numbers come from the analysis.")

# ------------------------------------------------------------------ INVESTIGATIONS
def listing():
    st.markdown('<div class="eyebrow">Investigations</div><div class="hero">What was examined.</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    cat = f1.selectbox("Category", ["All"] + sorted({x["category"] for x in cards}), key="f_cat")
    stt = f2.selectbox("Status", ["All", "Flagged", "Reviewed · no signal"] + STATUSES, key="f_stat")
    srt = f3.selectbox("Sort by", ["Priority score", "Category"], key="f_sort")
    rows = list(cards)
    if cat != "All": rows = [x for x in rows if x["category"] == cat]
    if stt == "Flagged": rows = [x for x in rows if x["status"] == "Flagged"]
    elif stt == "Reviewed · no signal": rows = [x for x in rows if x["status"] != "Flagged"]
    elif stt in STATUSES: rows = [x for x in rows if status_of(x["id"]) == stt]
    rows.sort(key=(lambda x: -(x["rank_breakdown"]["total"] if x["rank_breakdown"] else -1)) if srt == "Priority score" else (lambda x: x["category"]))
    st.caption("Priority score = 0.4 × impact + 0.3 × urgency + 0.3 × evidence strength (documented under Data). Only flagged findings are scored.")
    if not rows: st.info("No investigations match these filters.")
    for x in rows:
        c1, c2, c3, c4 = st.columns([5, 1.6, 1.6, 1], vertical_alignment="center")
        sc = f' · score {x["rank_breakdown"]["total"]:.2f}' if x["rank_breakdown"] else ""
        c1.markdown(f'**{x["finding_title"]}**<br><span class="small">{x["category"]}{sc}</span>', unsafe_allow_html=True)
        c2.markdown(tag("Flagged", "flag") if x["status"] == "Flagged" else tag("No signal"), unsafe_allow_html=True)
        c3.markdown(f'<span class="small">{status_of(x["id"])}</span>', unsafe_allow_html=True)
        c4.button("Open →", key=f"li_{x['id']}", on_click=open_inv, args=(x["id"],))
        st.markdown("<hr style='margin:.4rem 0'>", unsafe_allow_html=True)

def detail():
    x = by[ss.inv]
    st.markdown(f'<div class="crumb">Investigations / {x["category"]} / {x["finding_title"][:60]}</div>', unsafe_allow_html=True)
    st.button("← All investigations", key="back1", on_click=back)
    st.markdown(f'<div class="hero" style="font-size:2.3rem">{x["finding_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="small">{x["category"]} · {x["period"]} · Sources: {", ".join(x["sources"])}</div>', unsafe_allow_html=True)
    main, side = st.columns([2.2, 1])
    with main:
        st.markdown(f'<div class="sec">What we observed {tag("Observed","obs")}</div>', unsafe_allow_html=True)
        st.markdown(x["summary"])
        for e in x["evidence"]: st.markdown(f"- {e}")
        st.markdown(f'<div class="sec">Why it matters {tag("Interpretation","int")}</div>', unsafe_allow_html=True)
        st.markdown(x["why_it_matters"]); st.markdown(x["interpretation"])
        st.markdown(f'<div class="sec">What the data does not tell us {tag("Data limitation","lim")}</div>', unsafe_allow_html=True)
        for u in x["unknown"]: st.markdown(f"- {u}")
        st.markdown('<div class="sec">The picture</div>', unsafe_allow_html=True); chart(x)
        st.markdown('<div class="sec">Evidence ledger</div>', unsafe_allow_html=True)
        for n, it in enumerate(x["ledger"]):
            with st.expander(f'{it["label"]} — {it["value"]}'):
                st.markdown(f'**Source:** {it["source"]}  \n**Fields:** {it["fields"]}  \n**Comparison:** {it["comparison"]}  \n**Rule:** {it["rule"]}  \n'
                            f'**What it supports:** {it["supports"]}  \n**Limitation:** {it["limit"]}')
                if it["records"]: st.button("View records", key=f"rec_{x['id']}_{n}", on_click=show_records, args=(it["records"],))
    with side:
        st.markdown(f'<div class="card"><div class="col-h">Impact · {x["impact_label"]}</div><div>{x["impact_text"]}</div>'
                    f'<hr><div class="col-h">Confidence</div><div>{x["confidence"]}</div></div>', unsafe_allow_html=True)
        if x["assumptions"]:
            st.markdown('<div class="col-h">Assumptions</div>', unsafe_allow_html=True)
            for a_ in x["assumptions"]: st.markdown(f'<div class="small">• {a_}</div>', unsafe_allow_html=True)
        st.markdown('<div class="col-h" style="margin-top:1rem">What to verify next</div>', unsafe_allow_html=True)
        for v in x["confirm"]: st.markdown(f'<div class="small">• {v}</div>', unsafe_allow_html=True)
        st.selectbox("Status", STATUSES, index=STATUSES.index(status_of(x["id"])), key=f"status_{x['id']}", on_change=on_status, args=(x["id"],))
        if x["id"] in RECO: st.button("Decision brief →", key="to_brief", type="primary", on_click=set_view, args=("brief",))
        st.download_button("Export summary", summary_md(x), file_name=f"mirage_{x['id']}.md", mime="text/markdown", key="exp1")
        ai_box(x)

def brief():
    x = by[ss.inv]; rec, why, confirm = RECO[x["id"]]
    st.button("← Back to investigation", key="back2", on_click=set_view, args=("detail",))
    st.markdown(f'<div class="eyebrow">Decision brief · {x["category"]}</div><div class="hero" style="font-size:2.3rem">{rec}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec">Why this action</div>{why}', unsafe_allow_html=True)
    st.markdown('<div class="sec">What would confirm it</div>' + confirm, unsafe_allow_html=True)
    st.markdown('<div class="sec">Assumptions and limits</div>', unsafe_allow_html=True)
    for a_ in x["assumptions"] + x["unknown"]: st.markdown(f"- {a_}")
    st.markdown('<div class="small">Impact figures are a ' + x["impact_label"].lower() + ' using unreliable cost data; they are not a forecast and not profit.</div>', unsafe_allow_html=True)
    st.download_button("Export decision brief", summary_md(x), file_name=f"mirage_brief_{x['id']}.md", mime="text/markdown", key="exp2")

# ------------------------------------------------------------------ DATA
def models_section():
    b, an, u = mlres["baseline"], mlres["anomalies"], mlres["uplift"]
    st.markdown('<div class="sec">Local ML models</div>', unsafe_allow_html=True)
    st.caption("scikit-learn, fixed seeds. Reported as measured; with no labels none of these is a verified leak detector.")
    k1, k2, k3 = st.columns(3)
    k1.markdown(f'<div class="card"><div class="kpi-l">Baseline · weekly orders</div><div class="kpi-v">{b["mae_model"]:.1f} vs {b["mae_naive"]:.1f}</div>'
                f'<div class="kpi-s">error per week: model vs plain average. {"Model is better." if b["beats_naive"] else "No better than a plain average."}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="card"><div class="kpi-l">Uplift · deep discount</div><div class="kpi-v">{u["mean_uplift"]:+.3f} units</div>'
                f'<div class="kpi-s">per order, 95% interval {u["ci"][0]:+.3f} to {u["ci"][1]:+.3f} (baseline {u["control_mean"]:.2f})</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="card"><div class="kpi-l">Isolation Forest · products</div><div class="kpi-v">{an["n_products"]}</div>'
                f'<div class="kpi-s">products scored (20+ orders). Unusual does not mean leaking; the table shows what made each one unusual.</div></div>', unsafe_allow_html=True)
    t = an["table"].rename(columns=lambda x: F.LABELS.get(x, x.replace("_", " ").capitalize()))
    st.markdown("**Most unusual products**")
    st.dataframe(t, width="stretch", hide_index=True)

def data_page():
    st.markdown('<div class="eyebrow">Data</div><div class="hero">What this review rests on.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">Source files</div>', unsafe_allow_html=True)
    st.dataframe(F.source_health(raw, o), width="stretch", hide_index=True)
    st.markdown('<div class="sec">Joins and inputs</div>', unsafe_allow_html=True)
    st.dataframe(F.join_checks(raw, o), width="stretch", hide_index=True)
    st.markdown('<div class="card"><b>Limitation.</b> Costs and prices appear to have been generated independently, so the margin proxy (unit price − unit cost) is not profit. '
                'Campaigns cannot be joined to orders. Currency is shown as ₹ by assumption. Three small local models (below) add evidence but, with no labels, none is a verified leak detector.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">Data quality issues handled</div>', unsafe_allow_html=True)
    st.dataframe(A.data_quality(raw, o), width="stretch", hide_index=True)
    st.markdown('<div class="sec">Method</div>', unsafe_allow_html=True)
    st.markdown("- **Tests:** permutation tests, bootstrap 95% intervals, Bonferroni correction across products.\n"
                "- **Rules:** thresholds are learned on Q1–Q2 and applied to Q3–Q4; see the rule-based flags below.\n"
                "- **Local ML (scikit-learn):** gradient-boosting baseline, Isolation Forest, T-learner uplift. Fixed seeds, no GPU, results reported as measured.\n"
                "- **Local LLM (Ollama):** only phrases the numbers; it never sees raw rows and never computes a figure. Without Ollama the app shows plain template text.\n"
                "- **Priority score:** 0.4 × impact (figure ÷ 25% of net sales, capped at 1) + 0.3 × urgency (share of quarters the pattern holds) + 0.3 × evidence (mean of three checks).")
    models_section()
    fl, summ = T.rule_flags(o)
    st.caption(f"Rule-based flags: {summ}" if isinstance(summ, str) else "Rule-based flags (top 15 products)")
    st.dataframe(fl, width="stretch", hide_index=True)

if ss.nav == "Overview": overview()
elif ss.nav == "Investigations":
    if ss.view == "detail" and ss.inv: detail()
    elif ss.view == "brief" and ss.inv in RECO: brief()
    else: listing()
else: data_page()
