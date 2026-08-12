# Intelligent Customer Signal Detector — POC

Agentic pipeline (LangGraph) that ingests support chat logs, billing/usage
records, and satisfaction scores, correlates them, scores customer risk
(XGBoost + SHAP), and escalates high-risk customers to an LLM-generated
retention action.

## Quick start — the only thing you need to change

```bash
cp .env.example .env
# open .env and paste your key:
#   GEMINI_API_KEY=your-real-key
```

Then:

```bash
pip install -r requirements.txt
python -m data.generate_raw_sources   # sample siloed CSVs + data/customers.json (already included, re-run to regenerate)
python -m agents.train_model          # trains the risk model (already included, re-run to retrain)
streamlit run app/streamlit_app.py
```

Without a key, the app runs in **offline demo mode** automatically
(rule-based sentiment/action nodes) — fully functional, no crashes, no
external calls. This is also useful for CI or if you want a deterministic
demo run.

## About the Gemini rate limit (important)

Gemini's free tier caps `flash-lite` models at **15 requests/minute**.
Running several customers concurrently through the graph (`BATCH_SIZE`)
without protection causes `429 RESOURCE_EXHAUSTED` errors, since multiple
threads can call the API in the same second.

`agents/rate_limiter.py` implements a shared thread-safe token-bucket
limiter (`GEMINI_RPM_LIMIT`, default 15). Every live LLM call — sentiment
extraction and action recommendation, across every concurrent customer —
goes through `_llm_rate_limiter.acquire()` first, so total throughput
never exceeds the configured limit regardless of `BATCH_SIZE`. Raising
`BATCH_SIZE` just controls how many customers are in flight waiting on
the shared limiter, not how fast requests actually leave. If you're on a
paid tier with a higher quota, raise `GEMINI_RPM_LIMIT` in `.env` to match.

## Architecture

```
extract_sentiment  →  correlate_signals  →  score_risk  ──▶ route_on_risk
   (LLM, rate-limited)  (pulls CSAT/NPS)    (XGBoost+SHAP)        │
                                                    ┌──────────────┴──────────────┐
                                             risk < 65                   risk >= 65
                                                    │                             │
                                                   END              recommend_action (LLM, rate-limited)
                                                                              │
                                                                             END
```

- **`extract_sentiment`** — Gemini call turns the transcript into
  structured `{sentiment, sentiment_score, intent, key_phrase}` JSON,
  validated against a Pydantic schema (`SentimentResult`) with a lenient
  JSON/Python-literal parser as fallback for slightly malformed model
  output.
- **`correlate_signals`** — pulls `csat_score`/`nps_score` off the joined
  customer record into their own `satisfaction` state key.
- **`score_risk`** — XGBoost model (6 features: tenure, billing, late
  payments, usage trend, ticket count, sentiment score) produces a base
  risk score; CSAT/NPS are applied as a rule-based *adjustment* on top
  (not model inputs) so a customer with no survey response isn't
  penalized for missing data. SHAP explains the base model's contribution.
- **`route_on_risk`** — conditional edge. Only customers scoring ≥
  `RISK_ESCALATION_THRESHOLD` (default 65) proceed to `recommend_action`.
  This is the actual agentic branching, not a UI filter — low-risk
  customers' pipelines terminate immediately.
- **`recommend_action`** — Gemini call drafts a one-sentence retention
  action, using risk score, sentiment, and satisfaction context.

## Batch execution

`app/streamlit_app.py`'s `run_all()` calls the compiled graph's native
`.batch()`:

```python
results = compiled.batch(
    inputs,
    config={"max_concurrency": BATCH_SIZE},
    return_exceptions=True,
)
```

`max_concurrency` caps how many full customer pipelines run at once;
`return_exceptions=True` means one customer's failure (bad data, a
transient API error) doesn't abort the whole batch — that entry comes
back as an `Exception` and is shown with `Risk Band = "error"` in the UI
instead of crashing the page.

## Sample input files (siloed sources)

`data/raw_inputs/` — deliberately not pre-joined, matching how a real ops
team would actually have this data sitting in three separate systems:

| File | Simulates | Notes |
|---|---|---|
| `support_chat_logs.csv` | Helpdesk export | 0–4 tickets per customer |
| `billing_usage_records.csv` | Billing/usage system | One row per customer |
| `satisfaction_scores.csv` | Survey tool | Only 24/40 customers responded — partial coverage on purpose |
| `_ground_truth_risk.csv` | — | Demo/model-training only, never read by the live pipeline |

`data/loader.py` joins these by `customer_id`. `data/customers.json` is a
pre-joined convenience export used as the Streamlit app's default dataset
before any CSV is uploaded — the app also supports uploading your own
billing/chat/satisfaction CSVs directly in the sidebar.

## Project structure

```
signal-detector/
├── .env.example                  # copy to .env, add your GEMINI_API_KEY
├── data/
│   ├── raw_inputs/                # siloed sample CSVs
│   ├── customers.json             # pre-joined convenience export
│   ├── generate_raw_sources.py    # regenerates the sample CSVs + customers.json
│   └── loader.py                  # joins the siloed sources by customer_id
├── agents/
│   ├── config.py                  # all settings, only GEMINI_API_KEY is required
│   ├── rate_limiter.py            # thread-safe RPM limiter shared across .batch()
│   ├── graph.py                   # LangGraph pipeline
│   ├── scoring.py                 # XGBoost + SHAP
│   ├── train_model.py             # trains agents/risk_model.json
│   └── risk_model.json            # trained model artifact
├── app/
│   └── streamlit_app.py           # demo UI (upload CSVs or use sample data)
└── requirements.txt
```

## Known limitations

- 40 synthetic customers — the XGBoost model is illustrative, not
  production-grade. The point of the POC is the agentic architecture
  (branching, rate-limited concurrency, explainability), not model accuracy.
- `RISK_ESCALATION_THRESHOLD` is a config constant; a production version
  would tune it against a labeled churn outcome.
- Satisfaction adjustment in `scoring.py` is a simple rule (±5–10 points),
  not learned — kept explicit/auditable on purpose for a POC.
