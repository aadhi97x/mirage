# mirage. — Round 1 slide content (Team LARP)
Points and diagrams, not paragraphs. Numbers come from the app. Label the UI as a **prototype on synthetic data**.

## 01 · Title & Team
- Team **LARP** · Adhidev Jayaraj, Cavan J Paul, Emlin
- Track: **The Money Leak** · Product: **mirage.** — *"Looks good on paper. Does it hold up?"*

## 02 · Problem & Target User
- **User:** small/medium retail owners and category managers with no analytics team
- **Pain:** sales, returns, discounts, stock and campaigns sit in separate tables; reports say *what happened*, not *which contradiction matters first*
- **Evidence from the dataset:** 18,000 orders but campaigns report 895,411 conversions · ad spend ₹4.81M vs margin proxy ₹2.57M
- Visual: Overview screenshot (surface signal vs underlying evidence)

## 03 · Proposed Solution & USP (MANDATORY)
- **Solution:** Surface → Discrepancy → Decision. Tests whether apparent success is real, shows the evidence, says what to verify next
- **USP:** every claim has an evidence ledger (source, fields, rule, limitation, the records behind it); shows hypotheses that did *not* hold up
- vs dashboard: connects metrics · vs chatbot: numbers come from code, not a model · vs anomaly score: says what triggered, what is unknown, what to do
- Visual: Investigation detail screenshot

## 04 · Dataset Usage & Insights
- Tables: transactions ↔ products ↔ customers (no orphans/duplicates); inventory ↔ products; campaigns linked only by date/channel/category (no order ID)
- Assumptions: Completed orders = net sales; margin **proxy** = price − cost (costs unreliable); 171 impossible campaign rows flagged
- Insights: spend > margin proxy in 4/4 quarters; weekly orders vs spend r = 0.13 (p = 0.37); deep discounts show no volume lift; returns, bestsellers, stockouts tested and *not* supported

## 05 · ML Approach
- **Built from scratch, fully local — no external API.** scikit-learn on CPU, fixed seeds
- **Gradient boosting** baseline: weekly orders from spend, discount, season; trained H1, tested H2 → error 13.0 vs 11.8 for a plain average (no better)
- **Isolation Forest:** ranks 250 products by how unusual they are (unsupervised, no labels needed)
- **T-learner uplift:** deep discount (15–25%) changes units per order by −0.016 (95% interval −0.027 to −0.005) against a baseline of 3.01 → effectively zero
- Plus permutation tests, bootstrap intervals, Bonferroni correction for the verdicts
- Optional **local LLM (Ollama)** only phrases computed facts; never computes numbers
- Metric honesty: no labels → no precision/accuracy claimed; null results reported

## 06 · Base Prototype & MVP (MANDATORY)
- **Built:** Streamlit prototype — Overview, Investigations (detail, evidence ledger, record view, decision brief, status tracker), Data (source health, models)
- **MVP for Round 2:** CSV upload + validation · order-level campaign attribution · reviewed labels for Precision@K · what-if for discount cap · export brief

## 07 · Technical Approach
- Stack: Python · pandas/numpy · scikit-learn · Plotly · Streamlit · Ollama (optional local LLM)
- Flow: `Data → Validate & join → Tests + local ML → Investigations + ranking → UI → Decision brief`
- Ranking: 0.4×impact + 0.3×urgency + 0.3×evidence strength

## 08 · Feasibility, Impact & Next Steps
- **Feasibility:** runs on CSVs, CPU only, no API key, no cloud dependency
- **Impact:** ₹4.81M of campaign spend cannot be tied to orders; ₹479K of deep discounts show no volume payoff
- **Limitations:** synthetic data; observational; no attribution key; costs unreliable; models unvalidated (no labels)
- **Next:** order-level attribution, A/B test discounts, return reasons, daily stock logs
