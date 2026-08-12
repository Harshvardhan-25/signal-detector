"""
Generates three siloed sample input files matching real system exports:
  1. support_chat_logs.csv     - helpdesk/chat platform
  2. billing_usage_records.csv - billing/product usage system
  3. satisfaction_scores.csv   - post-interaction CSAT/NPS survey tool

Not pre-joined, not clean: multiple/zero tickets per customer, partial
satisfaction survey coverage. data/loader.py does the actual joining.

Also writes data/customers.json -- a convenience pre-joined export used as
the Streamlit app's default/fallback dataset before any CSV is uploaded.
"""
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from faker import Faker

fake = Faker()
random.seed(7)
Faker.seed(7)

RAW_DIR = Path(__file__).parent / "raw_inputs"
RAW_DIR.mkdir(exist_ok=True)
CUSTOMERS_JSON = Path(__file__).parent / "customers.json"

N_CUSTOMERS = 40
TODAY = datetime(2026, 8, 11)

HIGH_RISK_LINES = [
    "This is the third time I've called about the same billing error. Honestly looking at switching providers next week.",
    "Your support has been unresponsive for two weeks. If this isn't fixed by Friday I'm cancelling my plan.",
    "I was charged twice again this month. This keeps happening and nobody fixes it. Extremely frustrating.",
    "We've barely used the platform this month, it keeps timing out. Considering moving to a competitor.",
    "Asked for a refund three weeks ago, heard nothing back. Unacceptable for what I'm paying.",
]
MED_RISK_LINES = [
    "The new update is a bit slower than before, hoping that gets addressed soon.",
    "Had a login issue this week, support helped but it took a couple of days.",
    "Pricing feels a little high for what we're using, might reconsider at renewal.",
    "Mostly happy, just wish the reporting dashboard loaded faster.",
    "A few minor app bugs but nothing that's stopped daily use.",
]
LOW_RISK_LINES = [
    "The new feature rollout has been great, saved our team a lot of time.",
    "Support resolved my question in minutes, really happy with responsiveness lately.",
    "Everything's been working smoothly, no complaints this quarter.",
    "Loving the recent UI refresh, much easier to navigate now.",
    "Renewed for another year without a second thought.",
]
PLANS = ["Basic", "Pro", "Enterprise"]


def main():
    customer_ids = [f"CUST-{i:04d}" for i in range(1, N_CUSTOMERS + 1)]
    risk_buckets = {
        cid: random.choices(["high", "medium", "low"], weights=[0.3, 0.35, 0.35])[0]
        for cid in customer_ids
    }

    # 1. Support chat logs
    chat_rows = []
    ticket_counter = 1
    for cid in customer_ids:
        bucket = risk_buckets[cid]
        n_tickets = {"high": random.randint(2, 4), "medium": random.randint(1, 2), "low": random.randint(0, 1)}[bucket]
        lines = {"high": HIGH_RISK_LINES, "medium": MED_RISK_LINES, "low": LOW_RISK_LINES}[bucket]
        for _ in range(n_tickets):
            days_ago = random.randint(1, 90)
            chat_rows.append({
                "ticket_id": f"TCK-{ticket_counter:05d}",
                "customer_id": cid,
                "channel": random.choice(["chat", "email", "phone"]),
                "timestamp": (TODAY - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M"),
                "transcript": random.choice(lines),
            })
            ticket_counter += 1
    with open(RAW_DIR / "support_chat_logs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticket_id", "customer_id", "channel", "timestamp", "transcript"])
        w.writeheader()
        w.writerows(sorted(chat_rows, key=lambda r: r["timestamp"]))

    # 2. Billing & usage records
    billing_rows = []
    for cid in customer_ids:
        bucket = risk_buckets[cid]
        if bucket == "high":
            late_payments, usage_trend, tenure = random.randint(2, 4), random.randint(-60, -25), random.randint(3, 36)
        elif bucket == "medium":
            late_payments, usage_trend, tenure = random.randint(0, 1), random.randint(-20, 5), random.randint(6, 48)
        else:
            late_payments, usage_trend, tenure = 0, random.randint(0, 40), random.randint(6, 60)
        billing_rows.append({
            "customer_id": cid,
            "customer_name": fake.name(),
            "plan": random.choice(PLANS),
            "monthly_bill_usd": random.choice([49, 99, 199, 499, 999]),
            "tenure_months": tenure,
            "late_payments_90d": late_payments,
            "usage_trend_pct_90d": usage_trend,
            "last_billing_date": (TODAY - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
        })
    with open(RAW_DIR / "billing_usage_records.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(billing_rows[0].keys()))
        w.writeheader()
        w.writerows(billing_rows)

    # 3. Satisfaction scores (partial coverage on purpose)
    satisfaction_rows = []
    for cid in customer_ids:
        bucket = risk_buckets[cid]
        response_rate = {"high": 0.5, "medium": 0.6, "low": 0.7}[bucket]
        if random.random() > response_rate:
            continue
        if bucket == "high":
            csat, nps = random.randint(1, 2), random.randint(-100, -30)
        elif bucket == "medium":
            csat, nps = random.randint(2, 4), random.randint(-20, 30)
        else:
            csat, nps = random.randint(4, 5), random.randint(40, 100)
        satisfaction_rows.append({
            "customer_id": cid,
            "survey_date": (TODAY - timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d"),
            "csat_score": csat,
            "nps_score": nps,
        })
    with open(RAW_DIR / "satisfaction_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["customer_id", "survey_date", "csat_score", "nps_score"])
        w.writeheader()
        w.writerows(satisfaction_rows)

    # Ground truth -- demo/eval + model training only, never read by the pipeline
    with open(RAW_DIR / "_ground_truth_risk.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["customer_id", "ground_truth_risk"])
        w.writeheader()
        for cid in customer_ids:
            w.writerow({"customer_id": cid, "ground_truth_risk": risk_buckets[cid]})

    print(f"Wrote siloed sample inputs to {RAW_DIR}/:")
    print(f"  support_chat_logs.csv       ({len(chat_rows)} tickets, {len(customer_ids)} customers)")
    print(f"  billing_usage_records.csv   ({len(billing_rows)} customers)")
    print(f"  satisfaction_scores.csv     ({len(satisfaction_rows)}/{len(customer_ids)} customers responded)")

    # Now build the pre-joined customers.json convenience export
    from data.loader import load_all_customers
    customers = load_all_customers()
    CUSTOMERS_JSON.write_text(json.dumps(customers, indent=2))
    print(f"  customers.json               (pre-joined export, {len(customers)} customers)")


if __name__ == "__main__":
    main()
