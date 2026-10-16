# mirage.

**An evidence-led review of where a business looks healthy on paper but the underlying records disagree.**
Built for DATATHON 2026 · track *The Money Leak* · Team LARP (Adhidev Jayaraj, Cavan J Paul, Emlin).

Dashboards say *what happened*. mirage. sets the headline numbers against the raw records, tests whether a suspicious pattern is real, and says what to verify next. It also shows the hypotheses that did **not** hold up.

## What it finds on the sample data
| Finding | Evidence |
|---|---|
| Campaigns vs orders | 895,411 reported conversions vs 18,000 orders; spend is 1.87× the margin proxy |
| Ad spend vs sales | weekly orders do not move with weekly spend (r = 0.13, permutation p = 0.37) |
| Deep discounts | 15–25% off shows no lift in units per order; the uplift model estimates about 0 extra units |
| Tested, not supported | returns, bestsellers, stockouts |

## How it works
- **Statistics (the verdicts):** permutation tests, bootstrap confidence intervals, Bonferroni correction, rule thresholds learned on Q1–Q2 and applied to Q3–Q4 (`traps.py`).
- **Local ML (`ml.py`, scikit-learn, fixed seeds, CPU only):**
  1. gradient-boosting baseline for weekly orders (trained on H1, tested on H2),
  2. Isolation Forest to rank unusual products,
  3. T-learner uplift model for deep discounts.
  Results are reported as measured, including "no better than a plain average". There are no labels, so none of this is a verified leak detector.
- **Local LLM (`llm.py`, optional):** an open-source model served by [Ollama](https://ollama.com) turns the computed facts into plain language. It never sees raw rows and never computes a number. No API key and no data leaves your machine (the page only fetches two fonts from Google Fonts, with system-font fallbacks). Without Ollama the app shows plain template text and says so.
- **Honest money:** "margin proxy" = unit price − unit cost, never "profit" (costs in the dataset look unreliable). Figures are labelled Observed / Proxy / Estimate.

## Screens
Overview (surface vs underlying evidence) → Investigations (list, detail with evidence ledger and the records behind each claim, decision brief) → Data (source health, joins, method, local models).

## Run it
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Optional local LLM:
```bash
ollama pull qwen2.5:7b      # fits a 6 GB GPU; any small instruct model works
```
The app detects Ollama automatically (top-right status shows the model or "off").

## Configuration (all optional)
| Variable | Meaning | Default |
|---|---|---|
| `MONEY_LEAK_DATA` | folder with the five CSVs | `./data`, then `../04_business_money_leak` |
| `OLLAMA_HOST` | Ollama server | `http://localhost:11434` |
| `MONEY_LEAK_LLM` | model name | first installed of qwen2.5:7b, llama3.1:8b, llama3.2:3b … |

No secrets are needed. `.env` files and runtime state are git-ignored.

## Hosting
- **Streamlit Community Cloud:** push this folder as the repo root and set the main file to `app.py`. The statistics and ML models run fine; the LLM text falls back to the template because Ollama is not available there. The status tracker file is not persistent on free hosting.
- **Your own machine/server with Ollama:** everything works, including the LLM wording.

## Project layout
```
app.py        Streamlit UI (Overview / Investigations / Data)
traps.py      statistical tests and the five investigations
ml.py         the three local scikit-learn models
findings.py   evidence ledger, records views, data-health checks
analysis.py   loading, joins, data-quality
llm.py        local LLM wording (Ollama) with template fallback
theme.py      design tokens and CSS
data/         the five CSVs (synthetic hackathon dataset)
docs/         slide content for the idea PPT
```

## Limitations
Synthetic data; observational (association, not cause); campaigns have no order key; costs unreliable so margin is a proxy; rule thresholds and models are unvalidated because no reviewed labels exist. The tool supports decisions; it does not make them.
