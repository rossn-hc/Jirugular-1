#!/usr/bin/env python
"""
interface.py – Headless functions for Jira-RAG logic

Adds persona (character) + role support and preserves backward compatibility
with the legacy `pirate` flag. Also exposes intensity, temperature, max_tokens,
language, multi_format, verbose, and patricize (Dad joke) controls.
"""

from typing import Optional, Dict, Any, List, Union
import numpy as np

from .embedder import Embedder
from .vector_store import FaissIndexer
from .chat import ChatService
from .config import JIRA_URL, JIRA_USERNAME, JIRA_PASSWORD
from .jira_client import JiraClient


def crawl_and_build(jql: str = "", stem: str = "jira_vectors") -> None:
    """Fetch issues using JQL, embed them, and build the vector index."""
    client = JiraClient(JIRA_URL, JIRA_USERNAME, JIRA_PASSWORD, verify_ssl=False)
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
    # Backward-compat: allow legacy callers to pass pirate=True
    pirate: Optional[bool] = None,
    # NEW: append a corny dad joke at the end
    patricize: bool = False,
) -> Union[Dict[str, Any], str]:
    """
    Load index, initialize ChatService, and return model-generated answer.

    Returns:
        Dict (preferred path, includes 'answer', 'sources', 'structured') or str fallback.
    """
    # Map legacy pirate flag if no explicit character was provided
    if character is None and pirate:
        character = "pirate"

    embedder = Embedder()
    idx = FaissIndexer(dim=embedder.dim, stem=stem)

    try:
        idx.load()
    except FileNotFoundError:
        return "Index not found. Run crawl first."

    client = JiraClient(JIRA_URL, JIRA_USERNAME, JIRA_PASSWORD, verify_ssl=False)
    chat = ChatService(idx, embedder, client)

    # Preferred call signature (pass everything through, including patricize)
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
            patricize=patricize,
            pirate=None,  # don't send legacy arg when using new signature
        )
    except TypeError as e:
        # If ChatService is still on the old signature, retry with legacy-safe subset
        if "unexpected keyword argument" in str(e):
            legacy_pirate = bool((character or "").strip().lower() == "pirate") or bool(pirate)
            result = chat.answer(
                question=question,
                top_k=top_k,
                role=role,
                pirate=legacy_pirate,
                verbose=verbose,
                multi_format=multi_format,
                # patricize omitted on legacy path
            )
        else:
            raise

    # GUI expects a string; return dict for programmatic callers if needed
    if isinstance(result, dict):
        return result
    return str(result) if result is not None else "No response generated."


def show_dependencies(issue_key: str) -> str:
    """Return placeholder response for dependency graph."""
    return f"[Placeholder] Dependencies for issue {issue_key} would be shown here."
