"""vector_store.py – Persistent FAISS index helper.

Depends only on numpy and faiss‑cpu/faiss‑gpu. No Jira‑specific logic here so it
can be reused for other RAG pipelines.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import faiss  # type: ignore

__all__ = ["FaissIndexer"]


class FaissIndexer:
    """Simple wrapper around a FAISS flat inner‑product index (IP = cosine)."""

    def __init__(self, dim: int, *, stem: str = "jira_vectors"):
        self.dim = dim
        self.index_path = Path(f"{stem}.faiss")
        self.meta_path = Path(f"{stem}.jsonl")

        self._index: faiss.Index | None = None
        self._meta: list[dict[str, Any]] = []

        if self.index_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Public API                                                         
    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def metadata(self) -> Sequence[dict[str, Any]]:
        return self._meta

    def add(self, vectors: np.ndarray, metas: list[dict[str, Any]]):
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dim mismatch (expected {self.dim})")

        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dim)

        self._index.add(vectors.astype("float32"))
        self._meta.extend(metas)

    def save(self):
        if self._index is None:
            raise RuntimeError("Index is empty – nothing to save")

        faiss.write_index(self._index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as fh:
            for m in self._meta:
                fh.write(json.dumps(m) + "\n")

    def load(self):
        self._index = faiss.read_index(str(self.index_path))
        with self.meta_path.open(encoding="utf-8") as fh:
            self._meta = [json.loads(line) for line in fh]

        if self._index.ntotal != len(self._meta):
            raise RuntimeError("Vector count mismatch between index and metadata file")

    def search(self, query: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        if self._index is None:
            raise RuntimeError("Index not loaded")

        if query.ndim == 1:
            query = query.reshape(1, -1)
        dists, idxs = self._index.search(query.astype("float32"), k)

        results: list[dict[str, Any]] = []
        for dist, idx in zip(dists[0], idxs[0]):
            if idx == -1:
                continue
            meta = self._meta[idx].copy()
            meta["score"] = float(dist)
            results.append(meta)
        return results
