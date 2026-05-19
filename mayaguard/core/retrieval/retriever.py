"""
core/retrieval — Qdrant-backed vector retrieval.

Responsibilities:
  - Embed queries using a configurable embedding model
  - Store and search document chunks in Qdrant
  - Return ranked Document objects

Domain adapters supply their own collection name and corpus;
this module stays completely domain-agnostic.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from qdrant_client import AsyncQdrantClient, models as qmodels
from sentence_transformers import SentenceTransformer

from core.config import get_settings
from core.logging import get_logger
from core.models import Document, RetrievalResult

logger = get_logger(__name__)
_settings = get_settings()


class Retriever:
    """
    Async vector-search retriever backed by Qdrant.

    Usage::

        retriever = await Retriever.create()
        result = await retriever.retrieve("What is metformin?", top_k=5)
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        embed_model: SentenceTransformer,
        collection: str,
    ) -> None:
        self._client = client
        self._embed = embed_model
        self._collection = collection

    # ── Factory ───────────────────────────────────────────────────

    @classmethod
    async def create(
        cls,
        collection: str | None = None,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "Retriever":
        """
        Create and (optionally) initialise the Qdrant collection.

        Args:
            collection:       Qdrant collection name (defaults to settings value).
            embed_model_name: HuggingFace model name for embeddings.
        """
        col = collection or _settings.qdrant_collection
        client = AsyncQdrantClient(url=_settings.qdrant_url)
        embed = SentenceTransformer(embed_model_name)

        dim = embed.get_sentence_embedding_dimension()
        await cls._ensure_collection(client, col, dim)
        logger.info("retriever.ready", collection=col, embed_dim=dim)
        return cls(client, embed, col)

    @staticmethod
    async def _ensure_collection(
        client: AsyncQdrantClient, name: str, dim: int
    ) -> None:
        existing = {c.name for c in (await client.get_collections()).collections}
        if name not in existing:
            await client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=dim, distance=qmodels.Distance.COSINE
                ),
            )
            logger.info("retriever.collection_created", name=name)

    # ── Public API ────────────────────────────────────────────────

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Return the top-k most relevant documents for *query*."""
        vector = self._embed.encode(query).tolist()
        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )

        docs = [
            Document(
                id=str(hit.id),
                content=hit.payload.get("content", ""),
                source=hit.payload.get("source", "unknown"),
                score=float(hit.score),
                metadata={k: v for k, v in hit.payload.items() if k not in {"content", "source"}},
            )
            for hit in hits
        ]
        logger.debug("retriever.search", query=query[:80], hits=len(docs))
        return RetrievalResult(query=query, documents=docs, total_found=len(docs))

    async def upsert(self, documents: Sequence[Document]) -> int:
        """
        Index a list of Document objects into the collection.

        Returns the number of documents upserted.
        """
        points = [
            qmodels.PointStruct(
                id=self._stable_id(doc.id),
                vector=self._embed.encode(doc.content).tolist(),
                payload={
                    "content": doc.content,
                    "source": doc.source,
                    **doc.metadata,
                },
            )
            for doc in documents
        ]
        await self._client.upsert(collection_name=self._collection, points=points)
        logger.info("retriever.upsert", count=len(points), collection=self._collection)
        return len(points)

    @staticmethod
    def _stable_id(raw: str) -> int:
        """Convert an arbitrary string ID to a stable uint64."""
        return int(hashlib.sha256(raw.encode()).hexdigest()[:16], 16)
