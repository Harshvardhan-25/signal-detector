"""
Central config. The ONLY thing you need to change to run this for real is
GEMINI_API_KEY — either export it as an env var, or copy .env.example to
.env at the project root and fill it in there.
"""
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parents[1]
    dotenv_path = root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
except Exception:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

USE_MOCK_EVAL = os.environ.get("USE_MOCK_EVAL", "0").strip().lower() in {"1", "true", "yes", "y"}
USE_LIVE_LLM = bool(GEMINI_API_KEY) and not USE_MOCK_EVAL


RISK_ESCALATION_THRESHOLD = int(os.environ.get("RISK_ESCALATION_THRESHOLD", "65"))

# How many customer pipelines LangGraph's .batch() runs concurrently.
try:
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5"))
    if BATCH_SIZE <= 0:
        BATCH_SIZE = 5
except Exception:
    BATCH_SIZE = 5

try:
    GEMINI_RPM_LIMIT = int(os.environ.get("GEMINI_RPM_LIMIT", "15"))
    if GEMINI_RPM_LIMIT <= 0:
        GEMINI_RPM_LIMIT = 15
except Exception:
    GEMINI_RPM_LIMIT = 15
