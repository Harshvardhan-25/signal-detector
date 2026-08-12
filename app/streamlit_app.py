import json
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.config import BATCH_SIZE, GEMINI_RPM_LIMIT
from agents.graph import (
    LIVE_LLM_SUPPORTED,
    LLM_IMPORT_ERROR,
    USE_LIVE_LLM,
    build_graph,
)

st.set_page_config(page_title="Customer Signal Detector", layout="wide")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "customers.json"


@st.cache_resource
def get_graph():
    return build_graph()


@st.cache_data
def load_customers():
    return json.loads(DATA_PATH.read_text())


def _read_csv(file):
    try:
        return pd.read_csv(file, dtype=str, keep_default_na=False, na_values=[""], engine="python")
    except Exception:
        return pd.read_csv(file, dtype=str, keep_default_na=False, na_values=[""])


def _safe_int(value, default=0):
    try:
        if pd.isna(value) or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _normalize_billing(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "customer_name": "name",
        "plan": "plan",
        "monthly_bill_usd": "monthly_bill_usd",
        "tenure_months": "tenure_months",
        "late_payments_90d": "late_payments_90d",
        "usage_trend_pct_90d": "usage_trend_pct_90d",
        "support_tickets_90d": "support_tickets_90d",
        "customer_id": "customer_id",
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    if "customer_id" not in df.columns:
        return pd.DataFrame([])
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    for col in ["monthly_bill_usd", "tenure_months", "late_payments_90d", "usage_trend_pct_90d", "support_tickets_90d"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _safe_int(v, default=0))
    if "support_tickets_90d" not in df.columns:
        df["support_tickets_90d"] = 1
    return df


def _normalize_chat(df: pd.DataFrame) -> dict:
    if "customer_id" not in df.columns or "transcript" not in df.columns:
        return {}
    df = df.copy()
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["transcript"] = df["transcript"].astype(str).str.strip()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["customer_id", "timestamp"], ascending=[True, False])
    grouped = df.groupby("customer_id", sort=True)
    transcripts = {}
    for cid, group in grouped:
        texts = group["transcript"].dropna().astype(str).tolist()
        transcripts[cid] = " ".join(texts).strip()
    return transcripts


def _normalize_satisfaction(df: pd.DataFrame) -> dict:
    if "customer_id" not in df.columns:
        return {}
    df = df.copy()
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    if "survey_date" in df.columns:
        df["survey_date"] = pd.to_datetime(df["survey_date"], errors="coerce")
        df = df.sort_values(["customer_id", "survey_date"], ascending=[True, False])
    grouped = df.groupby("customer_id", sort=True)
    satisfaction = {}
    for cid, group in grouped:
        row = group.iloc[0]
        satisfaction[cid] = {
            "csat_score": _safe_int(row.get("csat_score", ""), default=None) if row.get("csat_score", "") != "" else None,
            "nps_score": _safe_int(row.get("nps_score", ""), default=None) if row.get("nps_score", "") != "" else None,
        }
    return satisfaction


def _synthesize_transcript_from_satisfaction(scores: dict) -> str:
    csat = scores.get("csat_score") or 0
    nps = scores.get("nps_score") or 0
    if csat >= 4 and nps >= 0:
        return f"Customer reported strong satisfaction in the survey: CSAT {csat}/5, NPS {nps}."
    if csat >= 3 and nps >= 0:
        return f"Customer survey is mixed: CSAT {csat}/5, NPS {nps}."
    return f"Customer reported low satisfaction: CSAT {csat}/5, NPS {nps}."


def _build_customers(billing_df=None, chat_map=None, satisfaction_map=None):
    baseline = {c["customer_id"]: c.copy() for c in load_customers()}

    source_ids = set()
    if billing_df is not None and not billing_df.empty:
        source_ids.update(billing_df["customer_id"].tolist())
    if chat_map is not None:
        source_ids.update(chat_map.keys())
    if satisfaction_map is not None:
        source_ids.update(satisfaction_map.keys())

    if not source_ids:
        return list(baseline.values())

    customers = []
    for cid in sorted(source_ids):
        customer = baseline.get(cid, {
            "customer_id": cid,
            "name": f"Customer {cid}",
            "plan": "Basic",
            "tenure_months": 12,
            "monthly_bill_usd": 99,
            "late_payments_90d": 0,
            "usage_trend_pct_90d": 0,
            "support_tickets_90d": 1,
            "transcript": "No transcript provided.",
            "csat_score": None,
            "nps_score": None,
        }).copy()

        if billing_df is not None and cid in billing_df["customer_id"].tolist():
            row = billing_df[billing_df["customer_id"] == cid].iloc[-1]
            customer["name"] = row.get("name", customer.get("name", customer["customer_id"]))
            customer["plan"] = row.get("plan", customer.get("plan", "Basic"))
            customer["monthly_bill_usd"] = _safe_int(row.get("monthly_bill_usd", ""), default=customer.get("monthly_bill_usd", 99))
            customer["tenure_months"] = _safe_int(row.get("tenure_months", ""), default=customer.get("tenure_months", 12))
            customer["late_payments_90d"] = _safe_int(row.get("late_payments_90d", ""), default=customer.get("late_payments_90d", 0))
            customer["usage_trend_pct_90d"] = _safe_int(row.get("usage_trend_pct_90d", ""), default=customer.get("usage_trend_pct_90d", 0))
            customer["support_tickets_90d"] = _safe_int(row.get("support_tickets_90d", ""), default=customer.get("support_tickets_90d", 1))

        if chat_map is not None and cid in chat_map:
            transcript = chat_map[cid]
            if transcript:
                customer["transcript"] = transcript

        if satisfaction_map is not None and cid in satisfaction_map:
            scores = satisfaction_map[cid]
            customer["csat_score"] = scores.get("csat_score")
            customer["nps_score"] = scores.get("nps_score")
            if not customer.get("transcript") or customer["transcript"] == "No transcript provided.":
                customer["transcript"] = _synthesize_transcript_from_satisfaction(scores)

        customer["monthly_bill_usd"] = _safe_int(customer.get("monthly_bill_usd", 99), default=99)
        customer["tenure_months"] = _safe_int(customer.get("tenure_months", 12), default=12)
        customer["late_payments_90d"] = _safe_int(customer.get("late_payments_90d", 0), default=0)
        customer["usage_trend_pct_90d"] = _safe_int(customer.get("usage_trend_pct_90d", 0), default=0)
        customer["support_tickets_90d"] = _safe_int(customer.get("support_tickets_90d", 1), default=1)
        customer["transcript"] = str(customer.get("transcript", "No transcript provided."))
        customer["plan"] = customer.get("plan", "Basic") or "Basic"
        customer["name"] = customer.get("name", customer["customer_id"])

        customers.append(customer)

    return customers


@st.cache_data(show_spinner="Running agent pipeline for all customers...")
def run_all(customers):
    compiled = build_graph()
    logger.info("Starting compiled graph pipeline for %d customers (batch=%d, rpm_limit=%d)",
                len(customers), BATCH_SIZE, GEMINI_RPM_LIMIT)
    customers = [c.copy() for c in customers]
    inputs = [{"customer": c} for c in customers]
    results = compiled.batch(
        inputs,
        config={"max_concurrency": BATCH_SIZE},
        return_exceptions=True,
    )

    rows = []
    for c, result in zip(customers, results):
        if isinstance(result, Exception):
            logger.error("Pipeline failed for customer %s: %s", c.get("customer_id"), result)
            rows.append({
                "Customer": c.get("name", c.get("customer_id")),
                "ID": c.get("customer_id"),
                "Plan": c.get("plan", "\u2014"),
                "Risk Score": None,
                "Risk Band": "error",
                "Sentiment": "\u2014",
                "Intent": "\u2014",
                "CSAT": c.get("csat_score"),
                "NPS": c.get("nps_score"),
                "Why flagged (SHAP)": str(result),
                "Recommended Action": "\u2014",
                "Transcript": c.get("transcript", ""),
            })
            continue

        sentiment = result.get("sentiment", {}) or {}
        satisfaction = result.get("satisfaction") or {}
        rows.append({
            "Customer": c.get("name", c.get("customer_id")),
            "ID": c.get("customer_id"),
            "Plan": c.get("plan", "\u2014"),
            "Risk Score": result.get("risk_score"),
            "Risk Band": result.get("risk_band", "unknown"),
            "Sentiment": sentiment.get("sentiment", "\u2014"),
            "Intent": sentiment.get("intent", "\u2014"),
            "CSAT": satisfaction.get("csat_score"),
            "NPS": satisfaction.get("nps_score"),
            "Why flagged (SHAP)": result.get("shap_rationale", ""),
            "Recommended Action": result.get("recommended_action") or "\u2014",
            "Transcript": c.get("transcript", ""),
        })

    logger.info("Finished compiled graph pipeline for %d customers", len(customers))
    df = pd.DataFrame(rows)
    df["CSAT"] = pd.array(df["CSAT"], dtype="Int64")
    df["NPS"] = pd.array(df["NPS"], dtype="Int64")
    return df


# --- UI --------------------------------------------------------------------
st.title("🚦 Intelligent Customer Signal Detector")
st.caption(
    "Upload billing, chat log, and satisfaction CSVs, or use the built-in sample data. "
    "Agentic pipeline (LangGraph): sentiment extraction \u2192 signal correlation \u2192 "
    "XGBoost risk scoring + SHAP explainability \u2192 conditional escalation to retention action."
)

if "start_agent" not in st.session_state:
    st.session_state.start_agent = False
if st.button("Start Detector"):
    st.session_state.start_agent = True

if USE_LIVE_LLM and LIVE_LLM_SUPPORTED:
    st.success(f"Live Gemini mode enabled (rate-limited to {GEMINI_RPM_LIMIT} req/min, batch size {BATCH_SIZE}).", icon="🔒")
elif USE_LIVE_LLM and not LIVE_LLM_SUPPORTED:
    st.warning("Gemini API key is present but the LLM package failed to load — falling back to offline mock mode.", icon="⚠️")
    if LLM_IMPORT_ERROR:
        st.code(LLM_IMPORT_ERROR)
else:
    st.info(
        "Running in **offline demo mode** (rule-based sentiment/action nodes) \u2014 "
        "set `GEMINI_API_KEY` in `.env` (see `.env.example`) to enable live LLM.",
        icon="\u2139\ufe0f",
    )

with st.sidebar:
    st.header("Upload detector data")
    billing_file = st.file_uploader("Billing usage CSV", type="csv")
    chat_file = st.file_uploader("Support chat logs CSV", type="csv")
    satisfaction_file = st.file_uploader("Satisfaction scores CSV", type="csv")
    st.markdown("Upload one, two, or all three files. Missing fields fall back to demo-safe defaults.")

billing_df = _normalize_billing(_read_csv(billing_file)) if billing_file else None
chat_map = _normalize_chat(_read_csv(chat_file)) if chat_file else None
satisfaction_map = _normalize_satisfaction(_read_csv(satisfaction_file)) if satisfaction_file else None

if billing_file or chat_file or satisfaction_file:
    customers = _build_customers(billing_df=billing_df, chat_map=chat_map, satisfaction_map=satisfaction_map)
    st.success(f"Loaded detector input for {len(customers)} customer(s).")
else:
    customers = load_customers()
    st.info("No upload detected — using the built-in sample dataset. Press Start to run the detector.")

if not st.session_state.start_agent:
    st.info("Press **Start Detector** above to begin agent processing with the current data.")
    st.stop()

try:
    df = run_all(customers)
except Exception as exc:
    st.error(f"Unable to run the detector: {exc}")
    st.stop()

if df.empty:
    st.warning("No customer data available to score.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers analyzed", len(df))
col2.metric("High risk (escalated)", int((df["Risk Band"] == "high").sum()))
col3.metric("Avg risk score", f"{df['Risk Score'].dropna().mean():.1f}" if df["Risk Score"].notna().any() else "\u2014")
col4.metric("Failed", int((df["Risk Band"] == "error").sum()))

st.divider()

band_filter = st.multiselect("Filter by risk band", ["high", "medium", "low", "error"], default=["high", "medium", "low"])
view = df[df["Risk Band"].isin(band_filter)].sort_values("Risk Score", ascending=False, na_position="last")


def highlight_band(row):
    color = {"high": "#ffe5e5", "medium": "#fff6e0", "low": "#e8f7ee", "error": "#e0e0e0"}.get(row["Risk Band"], "")
    return [f"background-color: {color}"] * len(row)


st.dataframe(
    view.style.apply(highlight_band, axis=1).set_properties(**{"color": "black"}),
    use_container_width=True,
    height=420,
    column_config={
        "Risk Score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%.1f"),
    },
)

st.divider()
st.subheader("Customer detail")
selected_id = st.selectbox("Select a customer", view["ID"].tolist())
row = df[df["ID"] == selected_id].iloc[0]

d1, d2 = st.columns([2, 1])
with d1:
    st.markdown(f"**{row['Customer']}** \u00b7 {row['Plan']} plan")
    st.markdown(f"> \u201c{row['Transcript']}\u201d")
    st.markdown(f"**Why flagged:** {row['Why flagged (SHAP)']}")
    if row["Recommended Action"] != "\u2014":
        st.success(f"**Recommended action:** {row['Recommended Action']}")
    else:
        st.caption("Below escalation threshold \u2014 no action node triggered (agentic routing).")
    if row["CSAT"] != "\u2014" or row["NPS"] != "\u2014":
        st.markdown(f"**CSAT:** {row['CSAT']}  |  **NPS:** {row['NPS']}")
with d2:
    st.metric("Risk score", f"{row['Risk Score']:.1f}/100" if pd.notna(row['Risk Score']) else "\u2014")
    st.metric("Risk band", str(row["Risk Band"]).capitalize())
    st.metric("Sentiment", str(row["Sentiment"]).capitalize())
    st.metric("Intent", str(row["Intent"]).replace("_", " ").capitalize())
