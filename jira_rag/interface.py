#!/usr/bin/env python
"""
interface.py – Headless functions for Jira-RAG logic
"""

from typing import Optional
import numpy as np

from .embedder import Embedder
from .vector_store import FaissIndexer
from .chat import ChatService
from .config import JIRA_URL, JIRA_USERNAME, JIRA_PASSWORD
from .jira_client import JiraClient

def crawl_and_build(jql: str = "", stem: str = "jira_vectors"):
    """Fetch issues using JQL, embed them, and build the vector index."""
    client = JiraClient(JIRA_URL, JIRA_USERNAME, JIRA_PASSWORD, verify_ssl=False)
    embedder = Embedder()
    index = FaissIndexer(dim=embedder.dim, stem=stem)

    issues = list(client.search(jql))
    if not issues:
        print("[WARN] No issues returned from JQL query.")
        return

    vectors = []
    metadata = []
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
    pirate: bool = False,
    verbose: bool = False,
    multi_format: bool = False,
    stem: str = "jira_vectors"
) -> str:
    """Load index, initialize ChatService, and return model-generated answer."""
    embedder = Embedder()
    idx = FaissIndexer(dim=embedder.dim, stem=stem)

    try:
        idx.load()
    except FileNotFoundError:
        return "Index not found. Run crawl first."

    client = JiraClient(JIRA_URL, JIRA_USERNAME, JIRA_PASSWORD, verify_ssl=False)
    chat = ChatService(idx, embedder, client)

    result = chat.answer(
        question=question,
        top_k=top_k,
        pirate=pirate,
        verbose=verbose,
        multi_format=multi_format
    )
    return result.get("answer", "No response generated.")

def show_dependencies(issue_key: str) -> str:
    """Return placeholder response for dependency graph."""
    return f"[Placeholder] Dependencies for issue {issue_key} would be shown here."
