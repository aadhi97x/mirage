# mirage.

**Looks good on paper. Does it hold up?** An evidence-led review that sets a business's reported numbers against its raw records, tests whether a suspicious pattern is real, and says what to verify next.

DATATHON 2026 · Track: **The Money Leak** · Team **LARP** (Adhidev Jayaraj, Cavan J Paul)

![Overview](docs/screenshots/01-overview.png)

**Three models trained from scratch on the track dataset** (scikit-learn, no pretrained weights, no external API): a gradient-boosting baseline for weekly orders, an Isolation Forest for unusual products, and a T-learner uplift model for discounts. [Details and results below](#our-models-trained-from-scratch-in-this-repo) · train them with `python train.py`.

## The problem
Businesses rarely lose money in one obvious hole. They lose it in places that *look* healthy: ad campaigns that report thousands of conversions, discounts that feel like marketing, products that sell well but quietly lose margin. Sales, returns, discounts, stock and campaign data sit in separate tables, so nobody checks whether the headline is backed by the records.

The track asks us to find revenue and operational problems in this data, investigate what drives them, and decide what to do.

## Target user
Small and mid-size retail owners and category/marketing managers who have no analytics team. They need to know **which contradiction matters first, how sure we are, and what to check before acting**, not another dashboard.

## What it finds on the track dataset
| Finding | Evidence |
|---|---|
| Campaigns vs orders | 895,411 reported conversions vs 18,000 orders; campaign spend is 1.87× the margin proxy |
| Ad spend vs sales | weekly orders do not move with weekly spend (r = 0.13, permutation p = 0.37) |
| Deep discounts | 15–25% discounts show no lift in units per order; the uplift model estimates about 0 extra units |
| Tested and **not** supported | returns, bestsellers, stockouts (shown openly, not hidden) |

"Margin proxy" = unit price − unit cost. It is never called profit, because costs in this dataset look unreliable. Every figure is labelled Observed, Proxy or Estimate.

## Screenshots
| Investigations | Investigation detail with evidence ledger |
|---|---|
| ![Investigations](docs/screenshots/02-investigations.png) | ![Detail](docs/screenshots/03-investigation-detail.png) |

Data health, join checks and the trained models:

![Data and models](docs/screenshots/04-data-and-models.png)

## Our models (trained from scratch, in this repo)
All three are trained on the track's own files (`data/*.csv`) with fixed seeds (`SEED = 7`), on CPU only. Code: [`ml.py`](ml.py). No pretrained weights and no external API are used for any result.

### 1. Gradient boosting: "what should weekly orders look like?"
- **Task:** regression. Predict weekly completed orders.
- **Model:** `HistGradientBoostingRegressor` (depth 3, 80 iterations, learning rate 0.05).
- **Features:** weekly campaign spend, average discount depth, week of year.
- **Training / test:** first 25 full weeks train, last 26 weeks test (time-ordered, no shuffling). Partial edge weeks are dropped.
- **Compared against:** a plain "predict the training average" baseline. Spend importance is measured with permutation importance (200 repeats).
- **Result:** error 13.0 orders/week vs 11.8 for the plain average, R² −0.21. The model does **not** beat the average, so spend, discount and season do not predict orders here. That supports the finding that ad spend is not visibly tied to sales.

### 2. Isolation Forest: "which products behave unusually?"
- **Task:** unsupervised anomaly detection (the dataset has no labels).
- **Model:** `IsolationForest` (300 trees) on standardised product features.
- **Features:** orders, net sales, average discount, cancel+return rate, share of orders below cost, margin proxy (250 products, each with 20+ orders).
- **Output:** an anomaly score per product plus the feature that made it stand out most (`product_anomaly_scores.csv`).
- **Reading it:** "unusual" is not "leaking". It is a list of products worth a closer look; the app shows the top 10 on the Data page.

### 3. T-learner uplift: "do deep discounts actually sell more?"
- **Task:** estimate the extra units per order caused by a 15–25% discount.
- **Model:** two `HistGradientBoostingRegressor`s, one fitted on discounted orders and one on undiscounted orders. The uplift for an order is the difference between their predictions.
- **Features:** category, list price, region, customer segment, channel, payment method, month.
- **Training / test:** random 50/50 split of completed orders; uplift is estimated on the held-out half, with a bootstrap 95% interval.
- **Result:** −0.016 units per order (95% interval −0.027 to −0.005) against a baseline of 3.01 units, which is effectively zero. Orders were not randomly assigned discounts, so this is association, not proven cause.

### How the models are used
They sit next to the statistical tests as extra evidence: the baseline and uplift results appear in the evidence ledger of the Campaigns and Discounting investigations, and all three are summarised on the **Data** page. The verdicts also use permutation tests, bootstrap intervals, Bonferroni correction and rule thresholds learned on Q1–Q2 and applied to Q3–Q4 (`traps.py`). There are no labels in the dataset, so we claim no accuracy or precision: these are evidence layers, not verified leak detectors, and weak results are reported as weak.

**Reproduce the training:**
```bash
python train.py
```
This loads `data/*.csv`, trains the three models and prints metrics. It also writes sample outputs, which are committed:
- [`docs/sample_outputs/model_report.json`](docs/sample_outputs/model_report.json): all metrics
- [`docs/sample_outputs/product_anomaly_scores.csv`](docs/sample_outputs/product_anomaly_scores.csv): Isolation Forest score for every product
- [`docs/sample_outputs/weekly_baseline_table.csv`](docs/sample_outputs/weekly_baseline_table.csv): weekly table with the train/test split

## Run the app
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate       macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Open http://localhost:8501. Python 3.10+.

Screens: **Overview** (surface vs underlying evidence) → **Investigations** (list, detail with evidence ledger and the records behind every claim, decision brief, status tracker, export) → **Data** (source health, joins, method, models).

## Data and loading
The five track files are in [`data/`](data/) (`customers`, `products`, `transactions`, `inventory`, `campaigns`; see `data/README.txt`). `analysis.py` loads and joins them (`load_all`), builds order-level fields and runs the data-quality checks. To use another copy set `MONEY_LEAK_DATA=/path/to/folder`.

## Declarations: APIs, pretrained models, extra data
| Item | Declaration |
|---|---|
| External APIs | **None.** The app makes no API calls to any hosted AI or data service. |
| Pretrained models | **None are used for any result.** One *optional* pretrained open-source LLM (default `qwen2.5:7b`, run locally through [Ollama](https://ollama.com)) can phrase the already-computed facts as plain-language text. It receives only the already computed results. Without Ollama the app works the same way but the results are shown in plain template text. |
| Extra datasets | **None.** Only the five track files. |
| Fonts | The page loads Inter and Instrument Serif from Google Fonts (CSS only), with system-font fallbacks. |
| Libraries | streamlit, pandas, numpy, plotly, jinja2, scikit-learn |

Optional local LLM:
```bash
ollama pull qwen2.5:7b      # fits a 6 GB GPU; any small instruct model works
```
The app detects Ollama automatically; the top-right status shows the model name or "off".

## Configuration (all optional)
| Variable | Meaning | Default |
|---|---|---|
| `MONEY_LEAK_DATA` | folder with the five CSVs | `./data` |
| `OLLAMA_HOST` | Ollama server | `http://localhost:11434` |
| `MONEY_LEAK_LLM` | LLM model name | first installed of qwen2.5:7b, llama3.1:8b, llama3.2:3b … |

No secrets are needed. `.env` files and runtime state are git-ignored.

## Hosting
Streamlit Community Cloud: use this folder as the repo root and `app.py` as the main file. Statistics and models run as usual; LLM wording falls back to template text (no Ollama there), and the status-tracker file does not persist on free hosting.

## Project layout
```
app.py        Streamlit UI (Overview / Investigations / Data)
train.py      trains the three models from the CSVs and writes sample outputs
ml.py         the three scikit-learn models
traps.py      statistical tests and the five investigations
findings.py   evidence ledger, records views, data-health checks
analysis.py   data loading, joins, data-quality checks
llm.py        optional local LLM wording (Ollama) with template fallback
theme.py      design tokens and CSS
data/         the track's five CSV files
docs/         screenshots, sample outputs, slide content
```

## Limitations
### Limitations

- **Synthetic data:** The current dataset is synthetic and may not fully represent real-world business conditions.
- **Correlation, not causation:** The analysis is observational. Identified relationships should not be interpreted as evidence of cause and effect.
- **Campaign attribution:** Campaigns lack a reliable order-level attribution key, limiting the ability to connect individual campaigns to specific purchases.
- **Profitability estimates:** Cost data may be incomplete or unreliable, so estimated margins should be treated as proxies rather than definitive financial measurements.
- **Model validation:** Models are relatively small and have not been validated against reviewed ground-truth labels, as such labels are not currently available.
- **Decision support:** Mirage provides analytical insights to inform human decisions; it does not replace human judgment or make decisions autonomously.
