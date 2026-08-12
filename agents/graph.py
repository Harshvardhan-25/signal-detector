"""
Agentic pipeline for the Customer Signal Detector POC.

Graph structure (LangGraph, StateGraph):

    extract_sentiment
          |
    correlate_signals
          |
      score_risk
          |
    route_on_risk  --(risk >= threshold)--> recommend_action --> END
          |
       (else) ------------------------------------------------> END

Deliberately NOT a linear chain: `route_on_risk` is a conditional edge that
inspects state and decides whether the (more expensive) action
recommendation node runs at all -- only customers above the risk threshold
get an LLM-generated retention action. That branching is the agentic part.

Entry point expects state["customer"] to already be a fully joined record
(transcript, billing/usage fields, csat_score/nps_score) -- joining across
the siloed source files happens in data/loader.py (or the Streamlit upload
handler), before the graph is invoked.

All live LLM calls (sentiment + action) go through a shared RateLimiter so
that concurrent .batch() execution across many customers never exceeds
GEMINI_RPM_LIMIT requests/minute -- see agents/rate_limiter.py.
"""
import ast
import json
import logging
from typing import Literal, TypedDict, Optional, Any

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

from agents.config import (
    GEMINI_API_KEY, GEMINI_MODEL, USE_LIVE_LLM, RISK_ESCALATION_THRESHOLD,
    BATCH_SIZE, GEMINI_RPM_LIMIT,
)
from agents.rate_limiter import RateLimiter

# Shared across every concurrent customer pipeline -- this is what actually
# prevents 429s under .batch(), regardless of BATCH_SIZE.
_llm_rate_limiter = RateLimiter(max_calls=GEMINI_RPM_LIMIT, period=60.0)

# --- LLM setup -----------------------------------------------------------
LIVE_LLM_SUPPORTED = False
LLM_IMPORT_ERROR = None
llm = None

if USE_LIVE_LLM:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain.messages import HumanMessage, SystemMessage
    except Exception as exc:
        LIVE_LLM_SUPPORTED = False
        LLM_IMPORT_ERROR = str(exc)
        USE_LIVE_LLM = False
        logger.warning("Live Gemini LLM unavailable: %s", LLM_IMPORT_ERROR)
    else:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            api_key=GEMINI_API_KEY,
            temperature=0,
        )
        LIVE_LLM_SUPPORTED = True
        logger.info("Live Gemini LLM enabled using model %s (rate limit %d/min)", GEMINI_MODEL, GEMINI_RPM_LIMIT)


NEGATIVE_WORDS = [
    "cancel", "cancelling", "switch", "unacceptable", "frustrat",
    "done", "refund", "unresponsive", "charged twice", "competitor",
]
POSITIVE_WORDS = [
    "great", "happy", "smooth", "loving", "solid", "resolved", "renewed",
]


def _mock_sentiment(transcript: str) -> dict:
    """Rule-based stand-in for the LLM sentiment call, used when no API key
    is configured (or USE_MOCK_EVAL=1). Mirrors the JSON shape the real LLM
    call returns."""
    text = transcript.lower()
    neg_hits = sum(1 for w in NEGATIVE_WORDS if w in text)
    pos_hits = sum(1 for w in POSITIVE_WORDS if w in text)
    if neg_hits > pos_hits:
        sentiment, score = "negative", -0.6 - 0.1 * min(neg_hits, 3)
        intent = "cancellation_risk" if "cancel" in text or "switch" in text or "competitor" in text else "complaint"
    elif pos_hits > neg_hits:
        sentiment, score = "positive", 0.6
        intent = "satisfaction"
    else:
        sentiment, score = "neutral", 0.0
        intent = "general_feedback"
    return {
        "sentiment": sentiment,
        "sentiment_score": round(max(-1.0, min(1.0, score)), 2),
        "intent": intent,
        "key_phrase": transcript.split(".")[0][:80],
    }


def _mock_action(customer: dict, risk_score: float, sentiment: dict, satisfaction: dict = None) -> str:
    if satisfaction:
        csat = satisfaction.get("csat_score")
        if csat is not None and csat <= 2:
            return "Escalate to retention specialist with a personalized recovery offer; the recent survey shows low satisfaction."
    if sentiment.get("intent") == "cancellation_risk":
        return "Escalate to retention specialist within 24h; offer service credit tied to the billing issue raised."
    return "Assign to account manager for proactive check-in call this week; monitor next billing cycle."


def _normalize_content(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(_normalize_content(item) for item in content)
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], (str, bytes)):
            return str(content["text"])
        if "content" in content and isinstance(content["content"], (str, bytes)):
            return str(content["content"])
        for key in ("message", "output", "response", "data"):
            if key in content:
                return _normalize_content(content[key])
        return " ".join(_normalize_content(value) for value in content.values())
    return str(content)


def _extract_response_text(resp) -> str:
    if resp is None:
        return ""
    if hasattr(resp, "text") and isinstance(resp.text, (str, bytes)):
        content = resp.text
    elif hasattr(resp, "message"):
        content = resp.message
    else:
        content = getattr(resp, "content", resp)
    return _normalize_content(content).strip()


class SentimentResult(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    intent: Literal["cancellation_risk", "complaint", "general_feedback", "satisfaction"]
    key_phrase: str


def _build_sentiment_prompt(transcript: str) -> str:
    return f"""Analyze this customer support transcript/review. Return ONLY valid JSON,
no preamble, in this exact shape:
{{"sentiment": "positive|neutral|negative", "sentiment_score": <float -1 to 1>,
"intent": "cancellation_risk|complaint|general_feedback|satisfaction",
"key_phrase": "<short phrase capturing the core issue or sentiment>"}}

Transcript: \"{transcript}\"
"""


def _build_action_prompt(customer: dict, risk_score: float, sentiment: dict, satisfaction: dict = None) -> str:
    satisfaction_text = ""
    if satisfaction:
        csat = satisfaction.get("csat_score")
        nps = satisfaction.get("nps_score")
        satisfaction_text = (
            f" Latest satisfaction survey: CSAT {csat}/5, NPS {nps}."
            if csat is not None or nps is not None
            else ""
        )
    return f"""A customer has been flagged as high risk (score {risk_score}/100).
Plan: {customer['plan']}, tenure: {customer['tenure_months']} months,
late payments (90d): {customer['late_payments_90d']}, usage trend (90d): {customer['usage_trend_pct_90d']}%,
support tickets (90d): {customer['support_tickets_90d']}.
Sentiment: {sentiment['sentiment']} ({sentiment['intent']}). Key issue: \"{sentiment['key_phrase']}\".{satisfaction_text}

In ONE short sentence, recommend a specific retention action for the account team."""


def _load_json_or_literal(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        segment = text[start:end + 1]
        try:
            return json.loads(segment)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(segment)
        except (ValueError, SyntaxError):
            pass
    raise ValueError("Unable to parse sentiment response as JSON or Python literal.")


def _validate_sentiment_json(text: str) -> dict:
    raw = _load_json_or_literal(text)
    try:
        return SentimentResult.model_validate(raw).model_dump()
    except ValidationError as exc:
        raise ValueError(f"Sentiment response did not match expected schema: {exc}") from exc


def _llm_sentiment_batch(transcripts: list[str]) -> list[dict]:
    """Runs sentiment analysis for a list of transcripts. Each transcript
    still costs one API call (Gemini has no native batch-prompt endpoint
    here), but all calls go through the shared rate limiter."""
    if not transcripts:
        return []
    if not LIVE_LLM_SUPPORTED:
        raise RuntimeError("Live Gemini LLM is not available for batch sentiment analysis.")

    logger.info("Batch sentiment analysis for %d transcripts", len(transcripts))
    messages = [
        [
            SystemMessage(content="You are a customer support sentiment analyst. Return only valid JSON with the requested keys."),
            HumanMessage(content=_build_sentiment_prompt(transcript)),
        ]
        for transcript in transcripts
    ]

    _llm_rate_limiter.acquire()
    response = llm.generate(messages=messages)

    sentiments = []
    for candidate_list in response.generations:
        if not candidate_list:
            raise ValueError("LLM returned no generation for a sentiment prompt.")
        raw_text = _extract_response_text(candidate_list[0])
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        sentiments.append(_validate_sentiment_json(raw_text))
    return sentiments


def _llm_sentiment(transcript: str) -> dict:
    logger.info("Calling LLM sentiment analyzer for transcript length %d", len(transcript))
    return _llm_sentiment_batch([transcript])[0]


def _llm_action_batch(customers: list[dict], risk_scores: list[float], sentiments: list[dict], satisfactions: list[dict]) -> list[str]:
    if not customers:
        return []
    if not LIVE_LLM_SUPPORTED:
        raise RuntimeError("Live Gemini LLM is not available for batch action recommendations.")

    logger.info("Batch action recommendation for %d high-risk customers", len(customers))
    messages = [
        [
            SystemMessage(content="You are a customer retention action recommender. Respond with a single short sentence and no explanation."),
            HumanMessage(content=_build_action_prompt(customer, risk_score, sentiment, satisfaction)),
        ]
        for customer, risk_score, sentiment, satisfaction in zip(customers, risk_scores, sentiments, satisfactions)
    ]

    _llm_rate_limiter.acquire()
    response = llm.generate(messages=messages)

    actions = []
    for candidate_list in response.generations:
        if not candidate_list:
            raise ValueError("LLM returned no generation for an action prompt.")
        actions.append(_extract_response_text(candidate_list[0]))
    return actions


def _llm_action(customer: dict, risk_score: float, sentiment: dict, satisfaction: dict = None) -> str:
    logger.info("Calling LLM action recommender for customer %s", customer.get("customer_id"))
    return _llm_action_batch([customer], [risk_score], [sentiment], [satisfaction])[0]


# --- Graph state -----------------------------------------------------------
class SignalState(TypedDict, total=False):
    customer: dict
    sentiment: dict
    satisfaction: dict
    derived_satisfaction_message: str
    risk_score: float
    risk_band: str
    shap_rationale: str
    recommended_action: Optional[str]


# --- Nodes -----------------------------------------------------------------
def extract_sentiment(state: SignalState) -> SignalState:
    customer_id = state["customer"].get("customer_id")
    logger.info("extract_sentiment beginning for customer_id=%s", customer_id)
    sentiment = state["customer"].get("sentiment")
    if sentiment is None:
        transcript = state["customer"]["transcript"]
        sentiment = _llm_sentiment(transcript) if USE_LIVE_LLM else _mock_sentiment(transcript)
    logger.info("extract_sentiment completed for customer_id=%s: %s", customer_id, sentiment)
    return {"sentiment": sentiment}


def correlate_signals(state: SignalState) -> SignalState:
    """Pulls the satisfaction survey fields off the joined customer record
    into their own state key, so score_risk and recommend_action get a
    clean `satisfaction` dict rather than reaching into `customer` again."""
    customer = state["customer"]
    customer_id = customer.get("customer_id")
    satisfaction = {
        "csat_score": customer.get("csat_score"),
        "nps_score": customer.get("nps_score"),
    }
    logger.info("correlate_signals for customer_id=%s satisfaction=%s", customer_id, satisfaction)
    if satisfaction["csat_score"] is None and satisfaction["nps_score"] is None:
        logger.info("No satisfaction signals for customer_id=%s", customer_id)
        return {}
    derived_message = (
        "Strong satisfaction survey signals present."
        if satisfaction["csat_score"] is not None and satisfaction["csat_score"] >= 4
        else "Low satisfaction survey signals present."
    )
    return {"satisfaction": satisfaction, "derived_satisfaction_message": derived_message}


def score_risk(state: SignalState, model=None) -> SignalState:
    customer_id = state["customer"].get("customer_id")
    logger.info("score_risk starting for customer_id=%s", customer_id)
    from agents.scoring import compute_risk_score

    result = compute_risk_score(
        state["customer"], state["sentiment"], satisfaction=state.get("satisfaction"), model=model,
    )
    logger.info("score_risk result for customer_id=%s risk_score=%s risk_band=%s", customer_id, result["risk_score"], result["risk_band"])
    return {
        "risk_score": result["risk_score"],
        "risk_band": result["risk_band"],
        "shap_rationale": result["shap_rationale"],
    }


def recommend_action(state: SignalState) -> SignalState:
    customer_id = state["customer"].get("customer_id")
    logger.info("recommend_action starting for customer_id=%s risk_score=%s", customer_id, state["risk_score"])
    customer = state["customer"]
    sent = state["sentiment"]
    satisfaction = state.get("satisfaction", {})
    action = customer.get("recommended_action")
    if action is None:
        if USE_LIVE_LLM:
            action = _llm_action(customer, state["risk_score"], sent, satisfaction)
        else:
            action = _mock_action(customer, state["risk_score"], sent, satisfaction)
    logger.info("recommend_action completed for customer_id=%s action=%s", customer_id, action[:200])
    return {"recommended_action": action}


def route_on_risk(state: SignalState) -> str:
    decision = "recommend_action" if state["risk_score"] >= RISK_ESCALATION_THRESHOLD else END
    logger.info("route_on_risk decision=%s for risk_score=%s", decision, state["risk_score"])
    return decision


# --- Build graph -------------------------------------------------------
def build_graph(model=None):
    logger.info("Building LangGraph state graph")
    graph = StateGraph(SignalState)
    graph.add_node("extract_sentiment", extract_sentiment)
    graph.add_node("correlate_signals", correlate_signals)
    graph.add_node("score_risk", lambda s: score_risk(s, model=model))
    graph.add_node("recommend_action", recommend_action)

    graph.set_entry_point("extract_sentiment")
    graph.add_edge("extract_sentiment", "correlate_signals")
    graph.add_edge("correlate_signals", "score_risk")
    graph.add_conditional_edges(
        "score_risk", route_on_risk, {"recommend_action": "recommend_action", END: END}
    )
    graph.add_edge("recommend_action", END)
    compiled = graph.compile()
    logger.info("Graph built successfully")
    return compiled
