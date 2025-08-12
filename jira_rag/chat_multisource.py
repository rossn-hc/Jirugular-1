#!/usr/bin/env python
"""
Multi-source retriever: runs a FAISS search per index, merges by score, and tags results with `source`.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np

from .vector_store import FaissIndexer
from .embedder import Embedder


class MultiSourceRetriever:
    def __init__(self, indexers: List[FaissIndexer], embedder: Embedder) -> None:
        self.indexers = indexers
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q = self.embedder.encode_one(query).astype(np.float32)
        merged: List[Tuple[float, Dict[str, Any]]] = []

        for idx in self.indexers:
            D, I = idx.search(q, k)
            for dist, ix in zip(D.tolist(), I.tolist()):
                if ix < 0:
                    continue
                meta = dict(idx.metadata[ix])  # shallow copy
                src = getattr(idx, "_source_name", meta.get("source") or "unknown")
                meta["source"] = src
                meta["_score"] = float(dist)
                merged.append((float(dist), meta))

        # NOTE: If your FaissIndexer uses cosine *similarity* (higher = better), reverse sorting as needed.
        merged.sort(key=lambda t: t[0])  # assuming lower distance is better
        return [m for _, m in merged[:k]]
