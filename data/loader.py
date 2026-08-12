"""
Loads and joins the three siloed sample input files by `customer_id`.
This is the "correlate signals across siloed sources" step from the brief,
made concrete: three systems that don't talk to each other in real life,
joined here into one record per customer.

Also provides normalization functions for CSV DataFrames from file uploads,
ensuring consistent schema across both file-based loading and streaming uploads.
"""
import csv
from pathlib import Path
from functools import lru_cache
import pandas as pd

RAW_DIR = Path(__file__).parent / "raw_inputs"

DEFAULT_CSAT = None  # None = "no survey on file", handled explicitly downstream
DEFAULT_NPS = None


def _read_csv(name: str) -> list[dict]:
    path = RAW_DIR / name
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def _load_all():
    chat_rows = _read_csv("support_chat_logs.csv")
    billing_rows = _read_csv("billing_usage_records.csv")
    satisfaction_rows = _read_csv("satisfaction_scores.csv")

    latest_transcript = {}
    ticket_counts = {}
    for row in chat_rows:
        cid = row["customer_id"]
        ticket_counts[cid] = ticket_counts.get(cid, 0) + 1
        if cid not in latest_transcript or row["timestamp"] > latest_transcript[cid]["timestamp"]:
            latest_transcript[cid] = row

    billing_by_id = {row["customer_id"]: row for row in billing_rows}
    satisfaction_by_id = {row["customer_id"]: row for row in satisfaction_rows}

    return latest_transcript, ticket_counts, billing_by_id, satisfaction_by_id


def list_customer_ids() -> list[str]:
    _, _, billing_by_id, _ = _load_all()
    return sorted(billing_by_id.keys())


def load_customer(customer_id: str) -> dict:
    latest_transcript, ticket_counts, billing_by_id, satisfaction_by_id = _load_all()

    billing = billing_by_id.get(customer_id)
    if billing is None:
        raise KeyError(f"No billing record found for {customer_id}")

    chat = latest_transcript.get(customer_id)
    satisfaction = satisfaction_by_id.get(customer_id)

    return {
        "customer_id": customer_id,
        "name": billing["customer_name"],
        "plan": billing["plan"],
        "tenure_months": int(billing["tenure_months"]),
        "monthly_bill_usd": int(billing["monthly_bill_usd"]),
        "late_payments_90d": int(billing["late_payments_90d"]),
        "usage_trend_pct_90d": int(billing["usage_trend_pct_90d"]),
        "support_tickets_90d": ticket_counts.get(customer_id, 0),
        "transcript": chat["transcript"] if chat else "No recent support contact on file.",
        "has_chat_data": chat is not None,
        "csat_score": int(satisfaction["csat_score"]) if satisfaction else DEFAULT_CSAT,
        "nps_score": int(satisfaction["nps_score"]) if satisfaction else DEFAULT_NPS,
        "has_satisfaction_data": satisfaction is not None,
    }


def load_all_customers() -> list[dict]:
    return [load_customer(cid) for cid in list_customer_ids()]


# CSV Upload / DataFrame Normalization (for Streamlit file uploads)
# ================================================================
# These functions normalize uploaded DataFrames to the same schema
# used by load_customer(), ensuring consistency across file-based and
# streaming data sources.


def _safe_int(value, default=0):
    """Safely convert a value to int, with a fallback default."""
    try:
        if pd.isna(value) or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _read_csv_from_file(file) -> pd.DataFrame:
    """Read a CSV file from Streamlit uploader, with graceful error handling."""
    try:
        return pd.read_csv(file, dtype=str, keep_default_na=False, na_values=[""], engine="python")
    except Exception:
        return pd.read_csv(file, dtype=str, keep_default_na=False, na_values=[""])


def normalize_billing_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame([])
    
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
    
    df = df.copy()
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    
    for col in ["monthly_bill_usd", "tenure_months", "late_payments_90d", "usage_trend_pct_90d", "support_tickets_90d"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _safe_int(v, default=0))
    
    if "support_tickets_90d" not in df.columns:
        df["support_tickets_90d"] = 1
    
    return df


def normalize_chat_dataframe(df: pd.DataFrame) -> dict:

    if df is None or df.empty or "customer_id" not in df.columns or "transcript" not in df.columns:
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


def normalize_satisfaction_dataframe(df: pd.DataFrame) -> dict:

    if df is None or df.empty or "customer_id" not in df.columns:
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


def merge_customers_from_dataframes(billing_df=None, chat_map=None, satisfaction_map=None) -> list[dict]:
    
    baseline = {c["customer_id"]: c.copy() for c in load_all_customers()}
    
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
        
        if billing_df is not None and not billing_df.empty and cid in billing_df["customer_id"].tolist():
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
        
        # Final type enforcement
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
