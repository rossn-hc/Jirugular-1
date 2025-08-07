"""embedder.py – Thin wrapper around Sentence‑Transformers for easy GPU/CPU use.

The module is intentionally small so it can be reused by other parts of the
project (e.g. a background re‑embedding task) without pulling in FAISS or Jira
code.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import torch

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:  # pragma: no cover
    DEVICE = "cpu"

__all__ = ["Embedder"]


class Embedder:
    """Encapsulates a Sentence‑Transformers model and hides device logic."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.dim = self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(self, texts: List[str], *, batch_size: int = 64) -> np.ndarray:  # type: ignore
        """Encode a list of texts → NumPy array (float32) of shape (n, dim)."""

        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) >= batch_size,
        ).astype("float32")

    def encode_one(self, text: str) -> np.ndarray:  # type: ignore
        """Encode a single string and return a (1, dim) array."""

        return self.encode([text])
