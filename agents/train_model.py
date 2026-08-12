"""
Trains a small XGBoost classifier using the joined siloed source files
(support chat logs, billing/usage records) via data/loader.py -- the same
join path the live pipeline uses. Satisfaction (CSAT/NPS) is intentionally
NOT a model feature here -- scoring.py applies it as a rule-based
post-hoc adjustment on top of the model's output instead (see scoring.py).

Run once: `python -m agents.train_model`
Produces: agents/risk_model.json
"""
import csv
from pathlib import Path

import pandas as pd
import xgboost as xgb

from agents.graph import _mock_sentiment
from data.loader import load_all_customers, RAW_DIR

MODEL_PATH = Path(__file__).parent / "risk_model.json"

FEATURE_COLS = [
    "tenure_months",
    "monthly_bill_usd",
    "late_payments_90d",
    "usage_trend_pct_90d",
    "support_tickets_90d",
    "sentiment_score",
]


def _load_ground_truth() -> dict:
    with open(RAW_DIR / "_ground_truth_risk.csv", newline="") as f:
        return {row["customer_id"]: row["ground_truth_risk"] for row in csv.DictReader(f)}


def build_feature_frame(customers: list[dict], ground_truth: dict) -> pd.DataFrame:
    rows = []
    for c in customers:
        sentiment = _mock_sentiment(c["transcript"])
        rows.append({
            "tenure_months": c["tenure_months"],
            "monthly_bill_usd": c["monthly_bill_usd"],
            "late_payments_90d": c["late_payments_90d"],
            "usage_trend_pct_90d": c["usage_trend_pct_90d"],
            "support_tickets_90d": c["support_tickets_90d"],
            "sentiment_score": sentiment["sentiment_score"],
            "label": 1 if ground_truth.get(c["customer_id"]) == "high" else 0,
        })
    return pd.DataFrame(rows)


def main():
    customers = load_all_customers()
    ground_truth = _load_ground_truth()
    df = build_feature_frame(customers, ground_truth)

    X = df[FEATURE_COLS]
    y = df["label"]

    model = xgb.XGBClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.15,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X, y)

    model.save_model(str(MODEL_PATH))
    print(f"Trained on {len(df)} customers (joined from siloed sources), {y.sum()} high-risk.")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
