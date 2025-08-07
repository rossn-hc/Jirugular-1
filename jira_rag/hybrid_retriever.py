#!/usr/bin/env python
"""
hybrid_retriever.py – Blend FAISS semantic recall with live Jira metadata.

Given a natural‑language *query*, the retriever:

1. Embeds the query and searches the FAISS index (semantic recall).
2. Parses any explicit Jira keys in the query itself (e.g. “ETL‑123”).
3. Fetches **fresh** fields (status, assignee, priority, updated) for the union
   of `{query_keys} ∪ {top_k.hit_keys}` in *batches* via Jira’s Search API.
4. Merges the live fields into the FAISS hit metadata.
5. Returns a list of enriched hit dicts.

This lets ChatService answer with up‑to‑date status while still leveraging
semantic search for discovery.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set

from .embedder import Embedder
from .jira_client import JiraClient
from .vector_store import FaissIndexer


KEY_RE = re.compile(r"[A-Z][A-Z0-9_]+-\d+")


class HybridRetriever:
    """FAISS + live Jira refresh."""

    FRESH_FIELDS = ["status", "assignee", "priority", "updated"]

    def __init__(
        self,
        indexer: FaissIndexer,
        embedder: Embedder,
        jira: JiraClient,
    ) -> None:
        self.indexer = indexer
        self.embedder = embedder
        self.jira = jira

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, *, k: int = 5) -> List[Dict[str, Any]]:
        """
        Return *k* enriched hits for the given query.
        Each hit's metadata contains freshly‑fetched live fields.
        """
        # 1) Semantic recall --------------------------------------------------
        q_vec = self.embedder.encode_one(query)
        base_hits = self.indexer.search(q_vec, k=k)

        # 2) Collect Jira keys (prompt + hits) -------------------------------
        keys = self._extract_keys(query)
        keys.update(h["key"] for h in base_hits)

        # 3) Live refresh in batches -----------------------------------------
        live_map = self._fetch_live_fields(keys)

        # 4) Merge ------------------------------------------------------------
        enriched: list[Dict[str, Any]] = []
        for h in base_hits:
            meta = h.copy()
            live = live_map.get(meta["key"])
            if live:
                meta.update(live)
            enriched.append(meta)

        return enriched

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_keys(text: str) -> Set[str]:
        """Find Jira keys like ABC‑123 in *text*."""
        return set(m.group(0) for m in KEY_RE.finditer(text))

    def _fetch_live_fields(self, keys: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """
        Batch‑fetch fresh fields for the given keys.
        Returns {key: {live_status: …, live_assignee: …, …}}.
        """
        key_list = list(keys)
        if not key_list:
            return {}

        live_data: dict[str, dict[str, Any]] = {}
        batch_size = 50
        for i in range(0, len(key_list), batch_size):
            batch = key_list[i : i + batch_size]
            jql = "key in ({})".format(", ".join(batch))
            for issue in self.jira.search(
                jql,
                fields=self.FRESH_FIELDS,
                batch=len(batch),
            ):
                f = issue["fields"]
                live_data[issue["key"]] = {
                    "live_status": f["status"]["name"] if f.get("status") else None,
                    "live_assignee": f["assignee"]["displayName"] if f.get("assignee") else None,
                    "live_priority": f["priority"]["name"] if f.get("priority") else None,
                    "live_updated": f.get("updated"),
                }
        return live_data


__all__ = ["HybridRetriever"]
