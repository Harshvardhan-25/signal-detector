"""
Loads and joins the three siloed sample input files by `customer_id`.
This is the "correlate signals across siloed sources" step from the brief,
made concrete: three systems that don't talk to each other in real life,
joined here into one record per customer.
"""
import csv
from pathlib import Path
from functools import lru_cache

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
