#!/usr/bin/env python
"""
ChatServiceMulti – reuse your rich ChatService (personas/language/format lock/patricize)
but swap the retriever to MultiSourceRetriever so you can query multiple stems at once.
"""

from __future__ import annotations
from typing import Optional, List
from openai import OpenAI

from .config import OPENAI_API_KEY, CHAT_MODEL
from .vector_store import FaissIndexer
from .embedder import Embedder
from .chat_multisource import MultiSourceRetriever
from .chat import ChatService


class ChatServiceMulti(ChatService):
    def __init__(
        self,
        indexers: List[FaissIndexer],
        embedder: Embedder,
        model: Optional[str] = None,
    ) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model or CHAT_MODEL or "gpt-4-turbo"
        self.retriever = MultiSourceRetriever(indexers, embedder)
        self.embedder = embedder
