"""
Risk scoring + explainability (SHAP) for a single customer, used by the
`score_risk` node in the LangGraph pipeline.
"""
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
import xgboost as xgb
import shap

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "risk_model.json"

FEATURE_COLS = [
    "tenure_months",
    "monthly_bill_usd",
    "late_payments_90d",
    "usage_trend_pct_90d",
    "support_tickets_90d",
    "sentiment_score",
]

FEATURE_LABELS = {
    "tenure_months": "tenure",
    "monthly_bill_usd": "monthly bill",
    "late_payments_90d": "late payments (90d)",
    "usage_trend_pct_90d": "usage trend (90d)",
    "support_tickets_90d": "support tickets (90d)",
    "sentiment_score": "conversation sentiment",
}


@lru_cache(maxsize=1)
def _load_model() -> xgb.XGBClassifier:
    logger.info("Loading XGBoost model from %s", MODEL_PATH)
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    logger.info("XGBoost model loaded")
    return model


@lru_cache(maxsize=1)
def _get_explainer():
    logger.info("Creating SHAP explainer")
    explainer = shap.TreeExplainer(_load_model())
    logger.info("SHAP explainer created")
    return explainer


def _band(score: float) -> str:
    if score >= 65:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def compute_risk_score(customer: dict, sentiment: dict, satisfaction: dict = None, model=None) -> dict:
    logger.info("Computing risk score for customer_id=%s", customer.get("customer_id"))
    model = model or _load_model()
    explainer = _get_explainer()

    row = pd.DataFrame([{
        "tenure_months": customer["tenure_months"],
        "monthly_bill_usd": customer["monthly_bill_usd"],
        "late_payments_90d": customer["late_payments_90d"],
        "usage_trend_pct_90d": customer["usage_trend_pct_90d"],
        "support_tickets_90d": customer["support_tickets_90d"],
        "sentiment_score": sentiment["sentiment_score"],
    }])[FEATURE_COLS]

    proba_high_risk = float(model.predict_proba(row)[0][1])
    risk_score = round(proba_high_risk * 100, 1)

    satisfaction_note = ""
    if satisfaction:
        csat = satisfaction.get("csat_score")
        nps = satisfaction.get("nps_score")
        logger.info("Applying satisfaction adjustments for customer_id=%s csat=%s nps=%s", customer.get("customer_id"), csat, nps)
        if csat is not None:
            if csat <= 2:
                risk_score = min(100, risk_score + 10)
                satisfaction_note = "Low satisfaction survey results also increased the risk."
            elif csat >= 4:
                risk_score = max(0, risk_score - 5)
                satisfaction_note = "Strong satisfaction survey results modestly lowered the risk."
        if nps is not None and nps <= -20 and not satisfaction_note:
            risk_score = min(100, risk_score + 5)
            satisfaction_note = "Negative NPS also contributed to higher risk."

    risk_score = round(risk_score, 1)

    shap_values = explainer.shap_values(row)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    contributions = list(zip(FEATURE_COLS, shap_values[0]))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top = contributions[:2]

    parts = []
    for feat, val in top:
        direction = "raised" if val > 0 else "lowered"
        parts.append(f"{FEATURE_LABELS[feat]} {direction} the risk score")
    rationale = "Flagged mainly because " + " and ".join(parts) + "."
    if satisfaction_note:
        rationale = rationale[:-1] + " " + satisfaction_note

    logger.info("Risk score completed for customer_id=%s risk_score=%s risk_band=%s", customer.get("customer_id"), risk_score, _band(risk_score))
    return {
        "risk_score": risk_score,
        "risk_band": _band(risk_score),
        "shap_rationale": rationale,
    }
