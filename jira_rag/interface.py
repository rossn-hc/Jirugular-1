#!/usr/bin/env python
"""
interface.py – Headless functions for Jira/Graph RAG logic

Adds persona/role knobs, language, intensity, temp/max_tokens and
supports:
  • Jira issues (stem default: "jira_vectors")
  • MS Graph People/Users (stem default: "msgraph_people")
  • MS Graph Audit Logs Sign-ins (stem default: "msgraph_signins")

This version is resilient to transient Jira 5xx/429, accepts either flavor of
JiraClient.search(..) (old: fields/batch; new: start_at/max_results/fields),
and treats blank JQL as "ORDER BY updated DESC".
"""

from typing import Optional, Dict, Any, List, Union, Tuple
import os
import time
import numpy as np

from .embedder import Embedder
from .vector_store import FaissIndexer
from .chat import ChatService
from . import config as cfg
from .jira_client import JiraClient

# Optional: MS Graph People crawler (provided in msgraph_crawler.py)
try:
    from .msgraph_crawler import crawl_msgraph_people as _crawl_msgraph_people
except Exception:
    _crawl_msgraph_people = None

# Optional: MS Graph Sign-ins crawler (provided in msgraph_signins_crawler.py)
try:
    from .msgraph_crawler_signins import crawl_msgraph_signins as _crawl_msgraph_signins
except Exception:
    _crawl_msgraph_signins = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _read_graph_creds() -> Tuple[str, str, str]:
    tenant_id = (getattr(cfg, "MSGRAPH_TENANT_ID", "") or os.getenv("MSGRAPH_TENANT_ID", "")).strip()
    client_id = (getattr(cfg, "MSGRAPH_CLIENT_ID", "") or os.getenv("MSGRAPH_CLIENT_ID", "")).strip()
    client_secret = (getattr(cfg, "MSGRAPH_CLIENT_SECRET", "") or os.getenv("MSGRAPH_CLIENT_SECRET", "")).strip()
    return tenant_id, client_id, client_secret


def _flex_search(client: JiraClient, jql: str, fields: List[str]) -> List[Dict[str, Any]]:
    """
    Call JiraClient.search regardless of which signature your implementation has.
    Retries on 429/502/503/504 with exponential backoff.
    """
    attempts = 6
    backoff = 0.6
    for i in range(attempts):
        try:
            # Try "new-ish" signature first (start_at/max_results/fields)
            try:
                return list(client.search(
                    jql,
                    start_at=0,
                    max_results=200,       # keep response size reasonable
                    fields=fields,
                ))
            except TypeError:
                # Fallback to "old" signature (fields/batch/expand)
                try:
                    return list(client.search(
                        jql,
                        fields=fields,
                        batch=200,
                    ))
                except TypeError:
                    # Last resort: plain call with just JQL
                    return list(client.search(jql))
        except Exception as e:
            msg = str(e).lower()
            transient = any(code in msg for code in (" 429", "429 ", " 502", " 503", " 504", "service unavailable"))
            if i < attempts - 1 and transient:
                time.sleep(min(8, backoff * (2 ** i)))
                continue
            raise
    return []  # unreachable, satisfies type checkers


# -----------------------------------------------------------------------------
# Jira crawl -> FAISS
# -----------------------------------------------------------------------------
def crawl_and_build(jql: str = "", stem: str = "jira_vectors") -> None:
    """Fetch issues using JQL, embed them, and build the vector index (Jira)."""
    effective_jql = (jql or "").strip() or "ORDER BY updated DESC"

    client = JiraClient(
        cfg.JIRA_URL,
        cfg.JIRA_USERNAME,
        cfg.JIRA_PASSWORD,
        verify_ssl=False,
    )

    embedder = Embedder()
    index = FaissIndexer(dim=embedder.dim, stem=stem)

    wanted_fields = [
        "summary", "description", "status", "assignee", "priority", "updated",
        "resolution", "labels", "components", "fixVersions", "reporter", "created",
        "issuetype", "project", "parent"
    ]

    try:
        issues = _flex_search(client, effective_jql, wanted_fields)
    except Exception as e:
        raise RuntimeError(f"Jira crawl failed for JQL '{effective_jql}': {e}")

    if not issues:
        print(f"[WARN] No issues returned for JQL: {effective_jql!r}.")
        return

    vectors: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []

    for issue in issues:
        doc = (issue.get("document") or "").strip()

        if not doc:
            f = issue.get("fields", {}) or {}
            key = issue.get("key") or "-"
            summary = (f.get("summary") or "No summary").strip()
            description = (f.get("description") or "No description").strip()
            status = (f.get("status", {}) or {}).get("name") or "-"
            assignee = (f.get("assignee", {}) or {}).get("displayName") or "-"
            priority = (f.get("priority", {}) or {}).get("name") or "-"
            updated = f.get("updated") or "-"
            labels = ", ".join(f.get("labels") or []) or "-"
            comps = ", ".join([c.get("name", "-") for c in (f.get("components") or [])]) or "-"
            fixes = ", ".join([v.get("name", "-") for v in (f.get("fixVersions") or [])]) or "-"
            project_key = (f.get("project", {}) or {}).get("key") or "-"
            issue_type = (f.get("issuetype", {}) or {}).get("name") or "-"
            parent_key = ((f.get("parent") or {}).get("key")) if f.get("parent") else None

            doc = (
                f"[{key}] {summary}\n"
                f"Project: {project_key} | Type: {issue_type}\n"
                f"Status: {status} | Priority: {priority} | Assignee: {assignee}\n"
                f"Labels: {labels} | Components: {comps} | Fix Versions: {fixes}\n"
                f"Updated: {updated}\n"
                f"{description}"
            ).strip()

            issue.setdefault("project_key", project_key)
            issue.setdefault("issue_type", issue_type)
            issue.setdefault("parent_key", parent_key)
            issue.setdefault("summary", summary)
            issue.setdefault("status", status)
            issue.setdefault("priority", priority)
            issue.setdefault("assignee", assignee)
            issue["document"] = doc

        vectors.append(Embedder().encode_one(doc).squeeze() if False else None)  # placeholder for linters
        # use the single embedder instance
        vectors[-1] = embedder.encode_one(doc).squeeze()
        metadata.append(issue)

    index.add(vectors=np.stack(vectors), metas=metadata)
    index.save()
    print(f"[INFO] Indexed {len(issues)} issues to stem '{stem}' (JQL={effective_jql!r}).")


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
    """Build the MS Graph (People/Users) FAISS index and return count."""
    if _crawl_msgraph_people is None:
        raise AttributeError("MS Graph people crawler not available (msgraph_crawler.py missing or errored).")

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

    if isinstance(result, tuple):
        result = result[0]
    return int(result or 0)


# -----------------------------------------------------------------------------
# MS Graph Sign-ins crawl -> FAISS
# -----------------------------------------------------------------------------
def crawl_msgraph_signins(
    stem: str = "msgraph_signins",
    start_date: Optional[str] = None,  # "YYYY-MM-DD" or ISO "YYYY-MM-DDThh:mm:ssZ"
    end_date: Optional[str] = None,
    app_display_name: Optional[str] = None,  # e.g., "Microsoft Teams"
    top: Optional[int] = None,
    tenant_id: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> int:
    """Build the MS Graph (Audit Logs /signIns) FAISS index and return count."""
    if _crawl_msgraph_signins is None:
        raise AttributeError("MS Graph sign-ins crawler not available (msgraph_signins_crawler.py missing or errored).")

    if not (tenant_id and client_id and client_secret):
        tid, cid, sec = _read_graph_creds()
        tenant_id = tenant_id or tid
        client_id = client_id or cid
        client_secret = client_secret or sec

    if not (tenant_id and client_id and client_secret):
        raise RuntimeError(
            "MSGRAPH_TENANT_ID / MSGRAPH_CLIENT_ID / MSGRAPH_CLIENT_SECRET are required "
            "in .env or passed to crawl_msgraph_signins()."
        )

    count, _stem = _crawl_msgraph_signins(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        stem=stem,
        start_date=start_date,
        end_date=end_date,
        app_display_name=app_display_name,
        top=top,
    )
    # If the crawler you pasted returns just an int, handle gracefully:
    if isinstance(count, tuple):
        count = count[0]
    return int(count or 0)


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
    Load the FAISS index indicated by `stem`, initialize ChatService, and return the answer.

    Datasource inference from stem:
      - stems starting with "msgraph_signins" -> datasource="signins"
      - stems starting with "msgraph", "people", "hr_" -> datasource="people"
      - otherwise -> datasource="jira"
    """
    s = (stem or "").lower()
    if s.startswith("msgraph_signins"):
        ds = "signins"
    elif s.startswith(("msgraph", "people", "hr_")):
        ds = "people"
    else:
        ds = "jira"

    embedder = Embedder()
    idx = FaissIndexer(dim=embedder.dim, stem=stem)

    try:
        idx.load()
    except FileNotFoundError:
        return f"Index not found for stem '{stem}'. Run crawl first."

    # Jira client only for Jira datasource; pass None otherwise (HybridRetriever should handle it)
    jira_client = JiraClient(cfg.JIRA_URL, cfg.JIRA_USERNAME, cfg.JIRA_PASSWORD, verify_ssl=False) if ds == "jira" else None
    chat = ChatService(idx, embedder, jira_client)

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
            datasource=ds,
        )
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            # Backward-compat path
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
