"""Voyage AI embedding client for query-time embeddings.

Uses voyage-multilingual-2 model (1024 dimensions) matching
the pre-computed embeddings in the database.
"""

import logging
from typing import Any

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_MODEL = "voyage-multilingual-2"
EMBEDDING_DIM = 1024


class EmbeddingClient:
    """Async client for Voyage AI embeddings."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.voyage_api_key
        self._client = httpx.AsyncClient(timeout=30.0)

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for a single text.

        Returns 1024-dimensional vector for voyage-multilingual-2.
        Falls back to zero vector on error.
        """
        if not self._api_key:
            logger.warning("Voyage API key not configured, returning zero vector")
            return [0.0] * EMBEDDING_DIM

        try:
            response = await self._client.post(
                VOYAGE_API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": VOYAGE_MODEL,
                    "input": text,
                    "input_type": "query",  # Optimized for search queries
                },
            )
            response.raise_for_status()
            data = response.json()

            # Extract embedding from response
            embedding = data["data"][0]["embedding"]
            return embedding

        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            return [0.0] * EMBEDDING_DIM

    async def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts in a single request.

        More efficient for batch operations.
        """
        if not self._api_key or not texts:
            return [[0.0] * EMBEDDING_DIM for _ in texts]

        try:
            response = await self._client.post(
                VOYAGE_API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": VOYAGE_MODEL,
                    "input": texts,
                    "input_type": "query",
                },
            )
            response.raise_for_status()
            data = response.json()

            # Extract embeddings preserving order
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding request failed: {e}")
            return [[0.0] * EMBEDDING_DIM for _ in texts]

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


# Global client instance
_embedding_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    """Get or create embedding client singleton."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
