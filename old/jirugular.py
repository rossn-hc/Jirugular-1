"""
Jira → FAISS → RAG chat prototype (GPU‑aware • persistent)
=========================================================

A single‑file, class‑based proof‑of‑concept that can:

1. **Crawl** a Jira Server/Data‑Center instance with transparent pagination.
2. **Embed** every issue and store vectors + rich metadata in a persistent
   FAISS index on disk (uses GPU automatically when CUDA wheels are present).
3. **Chat** with the indexed data via any OpenAI‑compatible chat/completions
   endpoint. Responses cite Jira keys pulled from the context.

The module is framework‑agnostic so you can later drop the classes into Django
management commands, Celery tasks, or API views.

---
Configuration & secrets
-----------------------
Secrets come from **environment variables** or fallback defaults defined below.
Create a `.env` file next to this script or export the variables in your shell:

```ini
JIRA_URL=https://jill-stg.jack.hc-sc.gc.ca/jira
JIRA_USERNAME=rnorrie
JIRA_PASSWORD=Canad@150
OPENAI_API_KEY=<your key>
```

---
Usage examples
--------------
```bash
# Crawl Jira, build embeddings & save index
python jira_rag_chat.py crawl

# Ask a question against the saved index
python jira_rag_chat.py chat "Which epics are blocked?"
```
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List

# Third‑party deps ---------------------------------------------------------
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

# Vector / LLM libs
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:  # torch optional for CPU‑only embedding
    DEVICE = "cpu"

import numpy as np
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    import openai  # optional – only needed for chat
except ImportError:
    openai = None  # type: ignore

# -------------------------------------------------------------------------
# Config & logging
# -------------------------------------------------------------------------
load_dotenv()

JIRA_URL: str = os.getenv("JIRA_URL", "https://jill-stg.jack.hc-sc.gc.ca/jira")
JIRA_USERNAME: str = os.getenv("JIRA_USERNAME", "rnorrie")
JIRA_PASSWORD: str = os.getenv("JIRA_PASSWORD", "Canad@150")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("jira_rag")

# -------------------------------------------------------------------------
# Data model
# -------------------------------------------------------------------------

@dataclass
class IssueNode:
    key: str
    issue_type: str
    summary: str
    description: str
    status: str
    project_key: str
    parent_key: str | None = None
    dependencies: list[dict[str, Any]] | None = None

    # Convert to plain‑text doc for embedding
    def to_document(self) -> str:
        deps = ", ".join(f"{d['direction'].upper()} {d['key']}" for d in (self.dependencies or [])) or "None"
        return (
            f"[{self.issue_type}] {self.key}\n"
            f"Status: {self.status}\n"
            f"Project: {self.project_key} | Parent: {self.parent_key or '-'}\n"
            f"Dependencies: {deps}\n\n"
            f"Summary: {self.summary}\n\nDescription:\n{self.description}"
        )

    # Metadata without bulky fields
    def to_metadata(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("description", None)
        d.pop("summary", None)
        return d

# -------------------------------------------------------------------------
# Jira client
# -------------------------------------------------------------------------

class JiraClient:
    def __init__(self, base_url: str, username: str, password: str, verify_ssl: bool = True):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(username, password)
        self.session.headers.update({"Content-Type": "application/json"})
        self.session.verify = verify_ssl

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _get(self, endpoint: str, **params: Any) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def search(self, jql: str, *, fields: list[str], batch: int = 200) -> Iterator[dict[str, Any]]:
        start = 0
        total = sys.maxsize
        while start < total:
            page = self._get(
                "rest/api/2/search",
                jql=jql,
                startAt=start,
                maxResults=batch,
                fields=",".join(fields),
            )
            total = page.get("total", 0)
            issues = page.get("issues", [])
            for iss in issues:
                yield iss
            start += len(issues)

# -------------------------------------------------------------------------
# Crawler
# -------------------------------------------------------------------------

class JiraCrawler:
    def __init__(self, client: JiraClient):
        self.client = client

    FIELDS = [
        "summary",
        "description",
        "status",
        "issuetype",
        "issuelinks",
        "project",
        "parent",
    ]

    def crawl(self, jql_scope: str = "") -> list[IssueNode]:
        log.info("Crawling Jira …")
        nodes: list[IssueNode] = []
        for issue in self.client.search(jql_scope or "", fields=self.FIELDS):
            f = issue["fields"]
            node = IssueNode(
                key=issue["key"],
                issue_type=f["issuetype"]["name"],
                summary=f.get("summary", ""),
                description=f.get("description", ""),
                status=f["status"]["name"],
                project_key=f["project"]["key"],
                parent_key=(f.get("parent") or {}).get("key"),
                dependencies=self._deps(f.get("issuelinks", [])),
            )
            nodes.append(node)
        log.info("Crawl finished – %s issues", len(nodes))
        return nodes

    @staticmethod
    def _deps(links: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for l in links:
            if "outwardIssue" in l:
                out.append({"direction": "outward", "key": l["outwardIssue"]["key"]})
            if "inwardIssue" in l:
                out.append({"direction": "inward", "key": l["inwardIssue"]["key"]})
        return out

# -------------------------------------------------------------------------
# Faiss indexer
# -------------------------------------------------------------------------

class FaissIndexer:
    def __init__(self, dim: int, stem: str = "jira_vectors"):
        self.dim = dim
        self.index_path = Path(f"{stem}.faiss")
        self.meta_path = Path(f"{stem}.jsonl")
        self.index: faiss.Index | None = None
        self.meta: list[dict[str, Any]] = []
        if self.index_path.exists():
            self.load()

    # -- building ---------------------------------------------------------

    def add(self, vectors: np.ndarray, metas: list[dict[str, Any]]):
        if self.index is None:
            # Flat IP (cosine) index; GPU if available & faiss-gpu
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors.astype("float32"))
        self.meta.extend(metas)

    def save(self):
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as fh:
            for m in self.meta:
                fh.write(json.dumps(m) + "\n")
        log.info("Index saved → %s (%s vectors)", self.index_path, self.index.ntotal)

    # -- loading/search ---------------------------------------------------

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        with self.meta_path.open(encoding="utf-8") as fh:
            self.meta = [json.loads(l) for l in fh]
        assert len(self.meta) == self.index.ntotal
        log.info("Loaded index (%s vectors)", self.index.ntotal)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index not loaded")
        score, idx = self.index.search(query_vec.astype("float32"), k)
        out: list[dict[str, Any]] = []
        for s, i in zip(score[0], idx[0]):
            if i == -1:
                continue
            m = self.meta[i].copy()
            m["score"] = float(s)
            out.append(m)
        return out

# -------------------------------------------------------------------------
# Embedder
# -------------------------------------------------------------------------

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])

# -------------------------------------------------------------------------
# Chat service
# -------------------------------------------------------------------------

class ChatService:
    def __init__(self, idx: FaissIndexer, emb: Embedder, model: str = DEFAULT