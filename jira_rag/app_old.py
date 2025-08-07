#!/usr/bin/env python
"""
app.py – Interactive console UI for the *jira_rag* package
=========================================================

Run from the workspace root:

    python -m jira_rag.app
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def crawl_and_build(jql: str = "", stem: str = "jira_vectors") -> None:
    """Fetch Jira issues → embed → persist FAISS index."""
    client = JiraClient(JIRA_URL, JIRA_USERNAME, JIRA_PASSWORD, verify_ssl=False)
    crawler = JiraCrawler(client)

    log.info("Starting Jira crawl …")
    nodes: List[IssueNode] = crawler.crawl(jql_scope=jql)

    log.info("Embedding %s issues …", len(nodes))
    embedder = Embedder()
    idx = FaissIndexer(dim=embedder.dim, stem=stem)

    docs = [n.to_document() for n in nodes]
    # include full document text in metadata
    metas = [
        {**n.to_metadata(), "document": docs[i]}
        for i, n in enumerate(nodes)
    ]
    idx.add(embedder.encode(docs), metas)
    idx.save()

    print(f"\n✅  Indexed {len(nodes)} issues and saved FAISS → {stem}.faiss / .jsonl\n")


def ask_question(question: str, top_k: int = 5, stem: str = "jira_vectors", pirate: bool = False) -> None:

    """Load index, ask ChatService, and print detailed hits + summary."""
    # 1) Load index & ChatService
    embedder = Embedder()
    idx = FaissIndexer(dim=embedder.dim, stem=stem)
    try:
        idx.load()
    except FileNotFoundError:
        print("Index not found. Run option 1 (crawl) first.")
        return

    jira = JiraClient(JIRA_URL, JIRA_USERNAME, JIRA_PASSWORD, verify_ssl=False)
    chat = ChatService(idx, embedder, jira)

    # 2) Prompt for how many hits
    count_str = input(f"How many issues to retrieve? [{top_k}]: ").strip()
    try:
        top_k = int(count_str) if count_str else top_k
    except ValueError:
        print(f"Invalid number, defaulting to {top_k}")

    # 3) Prompt for verbose & multi-format if desired
    verb = input("Verbose output? (y/N): ").strip().lower() == "y"
    mf   = input("Multi‑format summary? (y/N): ").strip().lower() == "y"

    # 4) Ask the model
    result = chat.answer(question, top_k=top_k, verbose=verb, multi_format=mf)
    hits = result["sources"]

    # 5) Print the detailed metadata for each hit
    print(f"\nBased on your query “{question}”, here are the top {len(hits)} issues:\n")
    for h in hits:
        status    = h.get("live_status")   or h.get("status")      or "–"
        assignee  = h.get("live_assignee") or h.get("assignee")    or "Not assigned"
        reporter  = h.get("reporter")      or "Unknown"
        priority  = h.get("live_priority") or h.get("priority")    or "Unknown"
        resolution= h.get("resolution")    or "–"
        created   = h.get("created")       or "–"
        updated   = h.get("live_updated")  or h.get("updated")     or "–"
        # extract description from the stored document
        doc       = h.get("document", "")
        desc      = ""
        if "Description:" in doc:
            desc = doc.split("Description:")[1].strip().split("\n")[0]
        print(f"{h['key']} ({h['issue_type']}):")
        print(f"  - Status: {status}")
        print(f"  - Project: {h['project_key']}")
        print(f"  - Assignee: {assignee}")
        print(f"  - Reporter: {reporter}")
        print(f"  - Priority: {priority}")
        print(f"  - Resolution: {resolution}")
        print(f"  - Created: {created}")
        print(f"  - Updated: {updated}")
        print(f"  - Description: {desc}\n")

    # 6) Print the LLM’s summary/answer
    print("Summary:")
    answer = result["answer"]
    if pirate:
        answer = pirateify(answer)
    print(answer, "\n")

    # 7) Re‑print sources with scores
    print("Sources:")
    for h in hits:
        print(f"• {h['key']} (score {h['score']:.2f})")
    print()

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
    return f"☠️ {text} Arrr, that be the way of it! ☠️"

def show_dependencies(key: str, stem: str = "jira_vectors") -> None:
    """Print live dependencies (blocks / is‑blocked‑by) for a given issue."""
    embedder = Embedder()
    idx = FaissIndexer(dim=embedder.dim, stem=stem)
    try:
        idx.load()
    except FileNotFoundError:
        print("Index not found. Run option 1 (crawl) first.")
        return

    matches = [m for m in idx.metadata if m["key"].upper() == key.upper()]
    if not matches:
        print("Key not found in index.")
        return

    deps = matches[0].get("dependencies") or []
    if not deps:
        print(f"{key} has no dependencies recorded in the index.")
        return

    print(f"\nDependencies for {key}:")
    for d in deps:
        arrow = "→" if d.get("direction") == "outward" else "←"
        print(f"  {arrow} {d.get('link_type', '').capitalize()}: {d['key']}")
    print()


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------
def chat_loop(pirate_mode: bool = False) -> None:
    greeting = "Ahoy! Ask me yer questions, or press Enter to set sail back to the main menu." if pirate_mode else "Enter questions.  Empty line or 'q' to return to main menu."
    print(greeting)
    while True:
        q = input("Q> ").strip()
        if q == "" or q.lower() in {"q", "quit", "exit", "back"}:
            break
        ask_question(q, pirate=pirate_mode)

def menu() -> None:
    while True:
        print(
            "\nJira‑RAG Interactive\n"
            "────────────────────\n"
            "1) Crawl / rebuild index\n"
            "2) Chat (multi‑turn)\n"
            "3) Chat as a Pirate ☠️\n"
            "4) Show dependencies for an issue\n"
            "5) Quit\n",
            flush=True,
        )
        choice = input("> ").strip()

        if choice == "1":
            jql = input("Optional JQL filter (or leave blank for all visible issues): ").strip()
            crawl_and_build(jql)

        elif choice == "2":
            chat_loop(pirate_mode=False)

        elif choice == "3":
            chat_loop(pirate_mode=True)

        elif choice == "4":
            key = input("Issue key (e.g. KSDS-19): ").strip()
            if key:
                show_dependencies(key)

        elif choice == "5":
            print("Good‑bye!")
            sys.exit(0)

        else:
            print("Arrr! That selection be invalid, matey. Try again.")



def main() -> None:
    try:
        menu()
    except KeyboardInterrupt:
        print("\nInterrupted.  Good‑bye!")


if __name__ == "__main__":
    main()
