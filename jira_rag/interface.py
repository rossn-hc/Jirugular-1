#!/usr/bin/env python
"""
interface.py – Headless functions for Jira-RAG logic

Adds persona/role knobs, language, intensity, temp/max_tokens and
*now* supports a separate MS Graph (People) datasource with its own FAISS stem.

Also prevents accidental Jira lookups when chatting against MS Graph stems by
injecting a NullJiraClient (no-op) instead of the real JiraClient.
"""
from typing import Optional, Dict, Any, List, Union, Tuple
import os
import numpy as np

from .embedder import Embedder
from .vector_store import FaissIndexer
from .chat import ChatService
from . import config as cfg
from .jira_client import JiraClient

# Optional: MS Graph crawler (provided in msgraph_crawler.py)
try:
    from .msgraph_crawler import crawl_msgraph_people as _crawl_msgraph_people
except Exception:
    _crawl_msgraph_people = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _is_msgraph_stem(stem: str) -> bool:
    return (stem or "").lower().startswith("msgraph")

def _read_graph_creds() -> Tuple[str, str, str]:
    """
    Read Graph creds defensively from cfg (if present) or environment.
    This avoids import-time errors when names aren't exported in config.py.
    """
    tenant_id = (getattr(cfg, "MSGRAPH_TENANT_ID", "") or os.getenv("MSGRAPH_TENANT_ID", "")).strip()
    client_id = (getattr(cfg, "MSGRAPH_CLIENT_ID", "") or os.getenv("MSGRAPH_CLIENT_ID", "")).strip()
    client_secret = (getattr(cfg, "MSGRAPH_CLIENT_SECRET", "") or os.getenv("MSGRAPH_CLIENT_SECRET", "")).strip()
    return tenant_id, client_id, client_secret


class NullJiraClient:
    """A no-op Jira client to prevent live Jira calls when chatting over MS Graph indexes."""
    def __init__(self, *_, **__):
        self.enabled = False

    # Methods the retriever might call:
    def search(self, *_, **__):
        return []
    def get_issue(self, *_, **__):
        return {}
    def issue(self, *_, **__):
        return {}
    def close(self):
        pass


# -----------------------------------------------------------------------------
# Jira crawl -> FAISS
# -----------------------------------------------------------------------------
def crawl_and_build(jql: str = "", stem: str = "jira_vectors") -> None:
    """Fetch issues using JQL, embed them, and build the vector index (Jira)."""
    client = JiraClient(cfg.JIRA_URL, cfg.JIRA_USERNAME, cfg.JIRA_PASSWORD, verify_ssl=False)
    embedder = Embedder()
    index = FaissIndexer(dim=embedder.dim, stem=stem)

    issues = list(client.search(jql))
    if not issues:
        print("[WARN] No issues returned from JQL query.")
        return

    vectors: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    for issue in issues:
        doc = issue.get("document", "") or ""
        vec = embedder.encode_one(doc)
        vectors.append(vec.squeeze())
        metadata.append(issue)

    index.add(vectors=np.stack(vectors), metas=metadata)
    index.save()
    print(f"[INFO] Indexed {len(issues)} issues to stem '{stem}'.")


# -----------------------------------------------------------------------------
# MS Graph People crawl -> FAISS
# -----------------------------------------------------------------------------
def crawl_msgraph_people(
    stem: str = "msgraph_people",
    top: Optional[int] = 5000,
    tenant_id: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> int:
    """
    Build the MS Graph (People/Users) FAISS index and always return a plain int.
    Credentials: explicit args > config.py attributes > environment.
    """
    if _crawl_msgraph_people is None:
        raise AttributeError("MS Graph crawler not available (msgraph_crawler.py missing or errored).")

    # Prefer explicit args; otherwise read lazily from cfg/env
    if not (tenant_id and client_id and client_secret):
        tid, cid, sec = _read_graph_creds()
        tenant_id = tenant_id or tid
        client_id = client_id or cid
        client_secret = client_secret or sec

    if not (tenant_id and client_id and client_secret):
        raise RuntimeError(
            "MSGRAPH_TENANT_ID / MSGRAPH_CLIENT_ID / MSGRAPH_CLIENT_SECRET are required "
            "in .env or passed to crawl_msgraph_people()."
        )

    result = _crawl_msgraph_people(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        stem=stem,
        top=top,
    )

    # Some implementations return (count, stem). Normalize to int.
    if isinstance(result, tuple):
        result = result[0]
    return int(result or 0)


# -----------------------------------------------------------------------------
# Q&A
# -----------------------------------------------------------------------------
def ask_question(
    question: str,
    top_k: int = 5,
    role: Optional[str] = None,
    character: Optional[str] = None,
    intensity: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    language: Optional[str] = None,
    verbose: bool = False,
    multi_format: bool = False,
    stem: str = "jira_vectors",
    pirate: Optional[bool] = None,
    patricize: bool = False,
) -> Union[Dict[str, Any], str]:
    """
    Load the FAISS index indicated by `stem`, initialize ChatService, and return model-generated answer.
    """
    # Infer datasource from the stem name
    ds = "jira"
    s = (stem or "").lower()
    if s.startswith("msgraph") or s.startswith("people") or s.startswith("hr_"):
        ds = "people"

    embedder = Embedder()
    idx = FaissIndexer(dim=embedder.dim, stem=stem)

    try:
        idx.load()
    except FileNotFoundError:
        return f"Index not found for stem '{stem}'. Run crawl first."

    # Only instantiate JiraClient when we’re actually using Jira
    jira_client = None
    if ds == "jira":
        jira_client = JiraClient(cfg.JIRA_URL, cfg.JIRA_USERNAME, cfg.JIRA_PASSWORD, verify_ssl=False)

    chat = ChatService(idx, embedder, jira_client)  # jira_client may be None for people mode

    try:
        result = chat.answer(
            question=question,
            top_k=top_k,
            role=role,
            character=character,
            intensity=intensity,
            temperature=temperature,
            max_tokens=max_tokens,
            language=language,
            verbose=verbose,
            multi_format=multi_format,
            pirate=pirate,
            patricize=patricize,
            datasource=ds,   # <<< NEW
        )
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            legacy_pirate = bool((character or "").strip().lower() == "pirate") or bool(pirate)
            result = chat.answer(
                question=question,
                top_k=top_k,
                pirate=legacy_pirate,
                verbose=verbose,
                multi_format=multi_format,
            )
        else:
            raise

    return result if isinstance(result, dict) else (str(result) if result is not None else "No response generated.")



def show_dependencies(issue_key: str) -> str:
    return f"[Placeholder] Dependencies for issue {issue_key} would be shown here."
