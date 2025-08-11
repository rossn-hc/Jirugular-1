#!/usr/bin/env python
"""
app.py – Interactive console UI for the *jira_rag* package
"""

from __future__ import annotations

import argparse
from typing import List, Optional

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
    metas = [{**n.to_metadata(), "document": docs[i]} for i, n in enumerate(nodes)]
    idx.add(embedder.encode(docs), metas)
    idx.save()
    log.info("Vector index saved to stem=%s", stem)


def ask_question(
    question: str,
    top_k: int = 5,
    stem: str = "jira_vectors",
    character: Optional[str] = None,
    role: Optional[str] = None,
    verbose: bool = False,
    multi_format: bool = False,
) -> str:
    embedder = Embedder()
    idx = FaissIndexer(dim=embedder.dim, stem=stem)
    try:
        idx.load()
    except FileNotFoundError:
        return "Index not found. Run crawl first."

    jira = JiraClient(JIRA_URL, JIRA_USERNAME, JIRA_PASSWORD, verify_ssl=False)
    chat = ChatService(idx, embedder, jira)

    result = chat.answer(
        question,
        top_k=top_k,
        verbose=verbose,
        multi_format=multi_format,
        role=role,
        character=character,
    )
    return result["answer"]


def show_dependencies(key: str, stem: str = "jira_vectors") -> None:
    embedder = Embedder()
    idx = FaissIndexer(dim=embedder.dim, stem=stem)
    try:
        idx.load()
    except FileNotFoundError:
        print("Index not found. Run crawl first.")
        return

    matches = [m for m in idx.metadata if m.get("key", "").upper() == key.upper()]
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


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jira-rag", description="Jira RAG Console")
    sub = p.add_subparsers(dest="cmd", required=True)

    # crawl
    c = sub.add_parser("crawl", help="Crawl Jira and build FAISS index")
    c.add_argument("--jql", default="", help="Optional JQL filter")
    c.add_argument("--stem", default="jira_vectors", help="Index stem name")

    # ask
    a = sub.add_parser("ask", help="Ask a question against the index")
    a.add_argument("question", help="User question")
    a.add_argument("--top-k", type=int, default=5, help="Results to retrieve")
    a.add_argument("--stem", default="jira_vectors", help="Index stem name")
    a.add_argument("--role", choices=["developer", "manager", "executive"], help="Audience framing")
    a.add_argument("--character", help="Persona (e.g., 'pirate', 'yoda', 'shakespeare')")
    a.add_argument("--verbose", action="store_true", help="Include full context blocks")
    a.add_argument("--multi-format", action="store_true", help="Emit multi-section summary")

    # deps
    d = sub.add_parser("deps", help="Show dependencies for an issue key")
    d.add_argument("key", help="Issue key, e.g., ABC-123")
    d.add_argument("--stem", default="jira_vectors", help="Index stem name")

    return p


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    if args.cmd == "crawl":
        crawl_and_build(jql=args.jql, stem=args.stem)
    elif args.cmd == "ask":
        ans = ask_question(
            question=args.question,
            top_k=args.top_k,
            stem=args.stem,
            character=args.character,
            role=args.role,
            verbose=bool(args.verbose),
            multi_format=bool(args.multi_format),
        )
        print(ans)
    elif args.cmd == "deps":
        show_dependencies(key=args.key, stem=args.stem)
    else:
        print("Unknown command")


if __name__ == "__main__":
    main()
