"""cli.py – Console entry‑point for the jira_rag package.

Run ``python -m jira_rag.cli crawl`` to fetch + embed, or ``… chat …`` to ask
questions.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from .config import JIRA_PASSWORD, JIRA_URL, JIRA_USERNAME
from .crawler import JiraCrawler
from .embedder import Embedder
from .jira_client import JiraClient
from .vector_store import FaissIndexer
from .chat import ChatService
from .models import IssueNode

log = logging.getLogger("jira_rag.cli")


def _save_raw(nodes: List[IssueNode], path: Path):
    import json

    path.write_text(json.dumps([n.__dict__ for n in nodes], indent=2))
    log.info("Saved raw nodes → %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Jira → FAISS → chat prototype")
    sub = p.add_subparsers(dest="cmd", required=True)

    crawl = sub.add_parser("crawl", help="Fetch Jira issues and build FAISS index")
    crawl.add_argument("--jql", default="", help="Optional JQL scope (default: all visible)")
    crawl.add_argument("--raw", default="jira_nodes.json", help="Path to save raw nodes JSON")

    chat = sub.add_parser("chat", help="Ask a question against the existing index")
    chat.add_argument("question", help="Natural language question")
    chat.add_argument("--top", type=int, default=5, help="Top‑k docs to retrieve")

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    embedder = Embedder()
    indexer = FaissIndexer(dim=embedder.dim)

    if args.cmd == "crawl":
        client = JiraClient(JIRA_URL, JIRA_USERNAME, JIRA_PASSWORD, verify_ssl=False)
        crawler = JiraCrawler(client)
        nodes = crawler.crawl(jql_scope=args.jql)
        _save_raw(nodes, Path(args.raw))

        docs = [n.to_document() for n in nodes]
        metas = [n.to_metadata() for n in nodes]
        vectors = embedder.encode(docs)
        indexer.add(vectors, metas)
        indexer.save()

    elif args.cmd == "chat":
        indexer.load()
        chat_svc = ChatService(indexer, embedder)
        result = chat_svc.answer(args.question, k=args.top)
        print("\nAnswer:\n", result["answer"], sep="")
        print("\nSources:")
        for src in result["sources"]:
            print(f"{src['key']}  (score {src['score']:.2f})")


if __name__ == "__main__":  # pragma: no cover
    main()
