#!/usr/bin/env python
"""
app.py – Interactive console UI for the *jira_rag* package
"""

from __future__ import annotations

import sys
from typing import List

from .config import JIRA_PASSWORD, JIRA_URL, JIRA_USERNAME, log
from .jira_client import JiraClient
from .crawler import JiraCrawler
from .vector_store import FaissIndexer
from .embedder import Embedder
from .chat import ChatService
from .models import IssueNode

def crawl_and_build(jql: str = "", stem: str = "jira_vectors") -> None:
    client = JiraClient(JIRA_URL, JIRA_USERNAME, JIRA_PASSWORD, verify_ssl=False)
    crawler = JiraCrawler(client)

    log.info("Starting Jira crawl …")
    nodes: List[IssueNode] = crawler.crawl(jql_scope=jql)

    log.info("Embedding %s issues …", len(nodes))
    embedder = Embedder()
    idx = FaissIndexer(dim=embedder.dim, stem=stem)

    docs = [n.to_document() for n in nodes]
    metas = [
        {**n.to_metadata(), "document": docs[i]}
        for i, n in enumerate(nodes)
    ]
    idx.add(embedder.encode(docs), metas)
    idx.save()

def ask_question(question: str, top_k: int = 5, stem: str = "jira_vectors", pirate: bool = False, verbose: bool = False, multi_format: bool = False) -> str:
    embedder = Embedder()
    idx = FaissIndexer(dim=embedder.dim, stem=stem)
    try:
        idx.load()
    except FileNotFoundError:
        return "Index not found. Run crawl first."

    jira = JiraClient(JIRA_URL, JIRA_USERNAME, JIRA_PASSWORD, verify_ssl=False)
    chat = ChatService(idx, embedder, jira)

    result = chat.answer(question, top_k=top_k, verbose=verbose, multi_format=multi_format)
    answer = result["answer"]
    return pirateify(answer) if pirate else answer

def pirateify(text: str) -> str:
    pirate_terms = {
        "you": "ye",
        "your": "yer",
        "are": "be",
        "have": "be havin'",
        "the": "th'",
        "is": "be",
        "it": "it be",
        "error": "blunder",
        "issue": "trouble",
        "problem": "squall",
        "solution": "treasure",
        "closed": "sent to Davy Jones' locker",
        "open": "floatin' in th' sea",
        "resolved": "patched up"
    }
    for normal, pirate in pirate_terms.items():
        text = text.replace(f" {normal} ", f" {pirate} ")
    return f"☠️ {text} Arrr!"

def show_dependencies(key: str, stem: str = "jira_vectors") -> None:
    embedder = Embedder()
    idx = FaissIndexer(dim=embedder.dim, stem=stem)
    try:
        idx.load()
    except FileNotFoundError:
        print("Index not found. Run crawl first.")
        return

    matches = [m for m in idx.metadata if m["key"].upper() == key.upper()]
    if not matches:
        print("Key not found in index.")
        return

    deps = matches[0].get("dependencies") or []
    if not deps:
        print(f"{key} has no dependencies.")
        return

    print(f"\nDependencies for {key}:")
    for d in deps:
        arrow = "→" if d.get("direction") == "outward" else "←"
        print(f"  {arrow} {d.get('link_type', '').capitalize()}: {d['key']}")

if __name__ == "__main__":
    print("Run 'app_gui.py' for GUI.")
