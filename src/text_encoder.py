"""Sentence-transformer wrapper with lightweight caching for repeated texts."""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
from sentence_transformers import SentenceTransformer

EmbeddingMode = Literal["query", "document"]


class TextEncoder:
    """Bi-encoder wrapper that caches repeated query and document embeddings."""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        batch_size: int = 128,
        normalize_embeddings: bool = True,
    ) -> None:
        """Store encoder configuration without loading weights eagerly."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.model: SentenceTransformer | None = None
        self.cache: dict[tuple[EmbeddingMode, str], np.ndarray] = {}

    def load_model(self) -> SentenceTransformer:
        """Load and memoize the underlying SentenceTransformer model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        return self.model

    def _prefix_for_mode(self, mode: EmbeddingMode) -> str:
        """Return an instruction prefix for models that use task-specific prompts."""
        if "nomic-embed-text" in self.model_name:
            return "search_query: " if mode == "query" else "search_document: "
        return ""

    def _prepare_text(self, text: str, mode: EmbeddingMode) -> str:
        """Format text before encoding so query/document roles stay consistent."""
        return f"{self._prefix_for_mode(mode)}{text}"

    def encode(self, texts: Iterable[str], mode: EmbeddingMode = "document", show_progress: bool = False) -> np.ndarray:
        """Encode a sequence of texts, only computing vectors for cache misses."""
        materialized_texts = list(texts)
        if not materialized_texts:
            return np.empty((0, 0), dtype=np.float32)

        missing_texts: list[str] = []
        seen: set[tuple[EmbeddingMode, str]] = set()

        for text in materialized_texts:
            key = (mode, text)
            if key not in self.cache and key not in seen:
                missing_texts.append(text)
                seen.add(key)

        if missing_texts:
            model = self.load_model()
            prepared = [self._prepare_text(text, mode) for text in missing_texts]
            embeddings = model.encode(
                prepared,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )

            for text, embedding in zip(missing_texts, embeddings, strict=True):
                self.cache[(mode, text)] = np.asarray(embedding, dtype=np.float32)

        return np.vstack([self.cache[(mode, text)] for text in materialized_texts])

    def encode_queries(self, texts: Iterable[str], show_progress: bool = False) -> np.ndarray:
        """Encode query texts with query-side prompting and caching."""
        return self.encode(texts, mode="query", show_progress=show_progress)

    def encode_documents(self, texts: Iterable[str], show_progress: bool = False) -> np.ndarray:
        """Encode candidate documents with document-side prompting and caching."""
        return self.encode(texts, mode="document", show_progress=show_progress)
