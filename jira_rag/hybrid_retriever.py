#!/usr/bin/env python
"""
hybrid_retriever.py – Blend FAISS semantic recall with optional live Jira metadata.

Flow:
1) Embed query and search the FAISS index (semantic recall).
2) (Optional) If Jira enrichment is enabled, parse Jira keys and refresh live fields
   in batches via the Jira Search API.
3) Merge live fields into FAISS hit metadata and return enriched hits.

Jira enrichment is *disabled* when:
- no Jira client is provided (jira is None), or
- source_hint is "people"/"msgraph"/"hr", or
- indexer.stem looks like a people/msgraph stem.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set

from .embedder import Embedder
from .vector_store import FaissIndexer
# JiraClient is optional at runtime; only for typing
try:
    from .jira_client import JiraClient  # type: ignore
except Exception:  # pragma: no cover
    JiraClient = object  # type: ignore


KEY_RE = re.compile(r"[A-Z][A-Z0-9_]+-\d+")


class HybridRetriever:
    """FAISS + (optional) live Jira refresh."""

    FRESH_FIELDS = ["status", "assignee", "priority", "updated"]

    def __init__(
        self,
        indexer: FaissIndexer,
        embedder: Embedder,
        jira: Optional[JiraClient] = None,
        source_hint: Optional[str] = None,
    ) -> None:
        self.indexer = indexer
        self.embedder = embedder
        self.jira = jira
        # allow caller to explicitly tell us which datasource we're on
        self.source_hint = (source_hint or "").lower()

        # Heuristic: infer from index stem if not explicitly provided
        stem = getattr(indexer, "stem", "") or ""
        if not self.source_hint:
            if any(x in stem.lower() for x in ("people", "msgraph", "hr")):
                self.source_hint = "people"
            elif any(x in stem.lower() for x in ("jira", "issue", "tickets")):
                self.source_hint = "jira"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, *, k: int = 5) -> List[Dict[str, Any]]:
        """
        Return *k* hits for the given query.
        If Jira enrichment is enabled, each hit will include live_* fields.
        """
        # 1) Semantic recall --------------------------------------------------
        q_vec = self.embedder.encode_one(query)
        base_hits = self.indexer.search(q_vec, k=k)

        # 2) Optional Jira live refresh --------------------------------------
        if self._should_enrich_jira():
            # collect Jira keys from query + hits
            keys = self._extract_keys(query)
            for h in base_hits:
                key = h.get("key") or h.get("issue_key")
                if key:
                    keys.add(str(key))

            # batch fetch live fields
            live_map = self._fetch_live_fields(keys)

            # merge into hits
            enriched: List[Dict[str, Any]] = []
            for h in base_hits:
                meta = dict(h)
                key = meta.get("key") or meta.get("issue_key")
                live = live_map.get(key) if key else None
                if live:
                    meta.update(live)
                enriched.append(meta)
            return enriched

        # If enrichment is off, just return the FAISS hits as-is
        return base_hits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_keys(text: str) -> Set[str]:
        """Find Jira keys like ABC-123 in *text*."""
        if not text:
            return set()
        return {m.group(0) for m in KEY_RE.finditer(text)}

    def _should_enrich_jira(self) -> bool:
        """Only enrich when we have a Jira client and the datasource is Jira-like."""
        if self.jira is None:
            return False
        if self.source_hint in {"people", "msgraph", "hr"}:
            return False
        return True

    def _fetch_live_fields(self, keys: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """
        Batch-fetch fresh fields for the given keys.
        Returns {key: {live_status: …, live_assignee: …, live_priority: …, live_updated: …}}.
        Never raises – failures just produce empty enrichment.
        """
        try:
            key_list = [k for k in keys if k]
            if not key_list or self.jira is None:
                return {}

            live_data: Dict[str, Dict[str, Any]] = {}
            batch_size = 50
            for i in range(0, len(key_list), batch_size):
                batch = key_list[i : i + batch_size]
                jql = "key in ({})".format(", ".join(batch))
                # Be defensive: some JiraClient.search signatures differ
                search_kwargs = {"fields": self.FRESH_FIELDS}
                try:
                    # Some clients expect 'batch'/'maxResults'
                    search_kwargs["batch"] = len(batch)  # type: ignore
                except Exception:
                    pass

                results = []
                try:
                    results = list(self.jira.search(jql, **search_kwargs))  # type: ignore[attr-defined]
                except Exception:
                    # don't break retrieval if Jira call fails
                    results = []

                for issue in results or []:
                    key = issue.get("key")
                    fields = issue.get("fields", {}) if isinstance(issue, dict) else {}
                    status = fields.get("status") or {}
                    assignee = fields.get("assignee") or {}
                    priority = fields.get("priority") or {}
                    live_data[key] = {
                        "live_status": status.get("name"),
                        "live_assignee": assignee.get("displayName"),
                        "live_priority": priority.get("name"),
                        "live_updated": fields.get("updated"),
                    }
            return live_data
        except Exception:
            return {}


__all__ = ["HybridRetriever"]
