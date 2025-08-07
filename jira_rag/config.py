"""config.py – shared configuration, environment variables, and logging."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Load .env (if present) -----------------------------------------------------
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # falls back gracefully if .env is missing

# ---------------------------------------------------------------------------
# Runtime settings -----------------------------------------------------------
# ---------------------------------------------------------------------------
JIRA_URL: str = os.getenv("JIRA_URL", "")
JIRA_USERNAME: str = os.getenv("JIRA_USERNAME", "")
JIRA_PASSWORD: str = os.getenv("JIRA_PASSWORD", "")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")

# Optional CUDA detection (used by embedder)
try:
    import torch

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:  # pragma: no cover
    DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Logging -------------------------------------------------------------------
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)

log = logging.getLogger("jira_rag")

__all__ = [
    "JIRA_URL",
    "JIRA_USERNAME",
    "JIRA_PASSWORD",
    "OPENAI_API_KEY",
    "CHAT_MODEL",
    "DEVICE",
    "log",
]
