#!/usr/bin/env python
"""
chat.py – ChatService that blends semantic recall (FAISS) with live Jira data,
with optional verbose mode, expanded metadata context, and enhanced summaries.
"""
from __future__ import annotations

from typing import Any, Dict, List
from openai import OpenAI
import openai as oai_err

from .config import OPENAI_API_KEY, CHAT_MODEL, log
from .hybrid_retriever import HybridRetriever
from .vector_store import FaissIndexer
from .embedder import Embedder
from .jira_client import JiraClient


class ChatService:
    def __init__(self, indexer: FaissIndexer, embedder: Embedder, jira: JiraClient, model: str | None = None) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model or CHAT_MODEL or "gpt-4-turbo"
        self.retriever = HybridRetriever(indexer, embedder, jira)

    def answer(
        self,
        question: str,
        top_k: int = 5,
        verbose: bool = False,
        multi_format: bool = False,
        role: str | None = None,
        pirate: bool = False,  # ✅ Add this
    ) -> Dict[str, Any]:
        hits = self.retriever.retrieve(question, k=top_k)
        unique_hits: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()
        for h in hits:
            key = h.get("key") or h.get("issue_key")
            if key and key not in seen_keys:
                seen_keys.add(key)
                unique_hits.append(h)
        hits = unique_hits

        system_base = self._make_system_prompt(multi_format=multi_format, role=role)
        prompt = self._build_prompt(system_base, question, hits, verbose=verbose)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=0.6,
                max_tokens=4096,
            )
        except oai_err.PermissionDeniedError as exc:
            if "model" in str(exc):
                log.debug("Model %s not available – falling back to gpt-4-turbo", self.model)
                resp = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=prompt,
                    temperature=0.6,
                    max_tokens=2048,
                )
            else:
                raise

        answer = resp.choices[0].message.content.strip()
        structured = [
            {
                "key": h.get("key"),
                "issue_type": h.get("issue_type"),
                "project": h.get("project_key"),
                "parent": h.get("parent_key"),
                "assignee": h.get("live_assignee") or h.get("assignee"),
                "reporter": h.get("reporter"),
                "priority": h.get("live_priority") or h.get("priority"),
                "resolution": h.get("resolution"),
                "status": h.get("live_status") or h.get("status"),
                "created": h.get("created"),
                "updated": h.get("live_updated") or h.get("updated"),
                "labels": h.get("labels", []),
                "components": h.get("components", []),
                "fix_versions": h.get("fix_versions", []),
                "summary": h.get("summary"),
                "description": h.get("document", "").strip()
            }
            for h in hits
        ]
        return {"answer": answer, "sources": hits, "structured": structured}

    def _make_system_prompt(self, multi_format: bool, role: str | None = None) -> str:
        if role == "developer":
            return (
                "You are a senior Jira-savvy developer. Summarize issues with technical clarity, focusing on code impact, blockers, dependencies, and implementation progress. "
                "Include statuses, priorities, fix versions, and technical labels. Use [KEY] format for references."
            )
        elif role == "manager":
            return (
                "You are a project manager reviewing Jira issues. Your goal is to track task ownership, delays, risks, overdue work, and workload distribution. "
                "Summarize who is responsible, what's at risk, and what requires follow-up. Use [KEY] format to cite issues."
            )
        elif role == "executive":
            return (
                "You are preparing a briefing for senior leadership. Generate a high-level summary of Jira issues across projects, including project health, delivery risk, and resourcing trends. "
                "Do not mention individuals unless critical. Focus on portfolio-level risk and progress signals. Cite issue [KEY]s if relevant."
            )

        if multi_format:
            return (
                "You are a seasoned Jira expert and analyst tasked with generating comprehensive summaries for a cross-functional audience. "
                "Your goal is to produce verbose, insightful narratives, not lists or field dumps.\n\n"
                "Output must include **three sections**:\n"
                "1. **Detailed Summary** – Paragraphs per issue using metadata fields. Give context, owners, status, risks.\n"
                "2. **Technical Summary** – Developer-oriented overview. Focus on progress, blockers, and priorities.\n"
                "3. **Management Summary** – High-level report for leadership. Include overall status, overdue/risk items, trends.\n"
                "4. **Overall Project Summary** – Conclude with an integrated narrative summarizing the overall health and distribution of effort across projects. Identify concentrations of risk, under-resourced areas, and emerging patterns or trends across all issues provided.\n\n---\n\nExample:\n"
                "[ABC-123] – This issue addresses optimization of the reporting engine. It is assigned to Jane Doe and currently In Progress. "
                "Last updated on 2023-07-14. It is high priority, blocking downstream features, and tagged with 'performance' and 'core-engine'."
            )

        return (
            "You are a senior Jira analyst producing detailed summaries. Always write in verbose paragraph style.\n"
            "Start with an overall project-level overview (issue counts, open/closed status).\n"
            "Then, for each issue, write a paragraph covering:\n"
            "• Summary & description\n• Responsible parties (assignee, reporter)\n• Lifecycle status (status, resolution, updated date)\n"
            "• Labels, components, fix versions\n• Urgency or blocking context\n• Mention issue keys using [KEY] format\n"
            "Conclude with executive-style insights or risk highlights.\n\n"
            "Example:\n"
            "[ABC-123] – This issue is focused on improving data sync reliability. It was reported by Sarah Lee and assigned to Tom Chu. "
            "The item is In Progress and last updated on 2023-08-01. It has High priority and no resolution yet."
        )

    @staticmethod
    def _build_prompt(system_prompt: str, question: str, hits: List[Dict[str, Any]], verbose: bool = False) -> List[Dict[str, str]]:
        context_blocks: list[str] = []
        for h in hits:
            status = h.get("live_status") or h.get("status") or "-"
            assignee = h.get("live_assignee") or h.get("assignee") or "-"
            reporter = h.get("reporter") or "-"
            priority = h.get("live_priority") or h.get("priority") or "-"
            resolution = h.get("resolution") or "-"
            created = h.get("created") or "-"
            updated = h.get("live_updated") or h.get("updated") or "-"
            labels = ", ".join(h.get("labels", [])) or "-"
            comps = ", ".join(h.get("components", [])) or "-"
            fixes = ", ".join(h.get("fix_versions", [])) or "-"

            doc_text = h.get("document", "").strip()
            if not doc_text:
                doc_text = f"No description provided for this issue titled '{h.get('summary', 'Untitled')}' in project {h.get('project_key', '-')}. It is classified as {h.get('issue_type', '-')}, currently {status}, and was last updated on {updated}."

            if verbose:
                block = (
                    f"[{h['key']}] ({h['issue_type']})\n"
                    f"Project: {h['project_key']} | Assignee: {assignee} | Reporter: {reporter}\n"
                    f"Status: {status} | Resolution: {resolution} | Priority: {priority}\n"
                    f"Created: {created} | Updated: {updated}\n"
                    f"Labels: {labels} | Components: {comps} | Fix Versions: {fixes}\n"
                    f"Description:\n{doc_text}\n––––––\n"
                )
            else:
                block = (
                    f"[{h['key']}] – {h.get('summary', 'No summary')}. Status: {status}. Assignee: {assignee}. Updated: {updated}."
                )

            context_blocks.append(block)

        context_txt = "\n".join(context_blocks)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Context:\n{context_txt}"},
            {"role": "user", "content": question},
        ]


__all__ = ["ChatService"]
