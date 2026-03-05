"""Embedding provider with Jina AI primary + Voyage AI fallback.

Strategy:
1. Try Jina AI first (89 languages, Hindi/Bengali/Urdu support, 10M free tokens)
2. Fall back to Voyage AI on Jina failure (rate limit, API error)
3. Return zero vector only as last resort (both fail)

This ensures reliable embedding retrieval for scheme matching while
preferring the better multilingual model for Indian languages.
"""

import logging
from typing import Protocol

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1024

# Voyage AI configuration
VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_MODEL = "voyage-multilingual-2"


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    async def get_embedding(self, text: str) -> list[float] | None: ...
    async def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]: ...


class FallbackEmbeddingClient:
    """Embedding client with Jina AI primary + Voyage AI fallback.

    Provides resilient embedding retrieval for the hybrid scheme search:
    - Primary: Jina AI jina-embeddings-v3 (best for Hindi, free tier)
    - Fallback: Voyage AI voyage-multilingual-2 (reliable backup)
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._jina_key = settings.jina_api_key
        self._voyage_key = settings.voyage_api_key
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def get_embedding(self, text: str) -> list[float] | None:
        """Get embedding with automatic fallback.

        Args:
            text: Text to embed (user query or scheme description)

        Returns:
            1024-dimensional embedding vector, or None if all providers fail.
        """
        # Try Jina first (better Indian language support)
        if self._jina_key:
            try:
                return await self._jina_embedding(text)
            except Exception as e:
                logger.warning(f"Jina embedding failed, trying Voyage: {e}")

        # Fallback to Voyage
        if self._voyage_key:
            try:
                return await self._voyage_embedding(text)
            except Exception as e:
                logger.error(f"Voyage embedding also failed: {e}")

        # Last resort: return None so callers can skip vector ranking.
        logger.error("All embedding providers failed, returning None")
        return None

    async def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts with fallback.

        Args:
            texts: List of texts to embed

        Returns:
            List of 1024-dimensional embedding vectors
        """
        if not texts:
            return []

        # Try Jina first
        if self._jina_key:
            try:
                return await self._jina_embeddings_batch(texts)
            except Exception as e:
                logger.warning(f"Jina batch embedding failed, trying Voyage: {e}")

        # Fallback to Voyage
        if self._voyage_key:
            try:
                return await self._voyage_embeddings_batch(texts)
            except Exception as e:
                logger.error(f"Voyage batch embedding also failed: {e}")

        # Last resort: return empty batch for explicit failure handling by caller.
        logger.error("All embedding providers failed for batch, returning empty list")
        return []

    async def _jina_embedding(self, text: str) -> list[float]:
        """Get embedding from Jina AI."""
        from src.integrations.jina_client import JINA_API_URL, JINA_MODEL

        client = await self._get_client()
        response = await client.post(
            JINA_API_URL,
            headers={
                "Authorization": f"Bearer {self._jina_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_MODEL,
                "input": [text],
                "task": "retrieval.query",  # Optimized for search queries
                "normalized": True,
                "dimensions": EMBEDDING_DIM,
            },
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    async def _jina_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get batch embeddings from Jina AI."""
        from src.integrations.jina_client import JINA_API_URL, JINA_MODEL

        client = await self._get_client()
        response = await client.post(
            JINA_API_URL,
            headers={
                "Authorization": f"Bearer {self._jina_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_MODEL,
                "input": texts,
                "task": "retrieval.query",
                "normalized": True,
                "dimensions": EMBEDDING_DIM,
            },
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    async def _voyage_embedding(self, text: str) -> list[float]:
        """Get embedding from Voyage AI."""
        client = await self._get_client()
        response = await client.post(
            VOYAGE_API_URL,
            headers={
                "Authorization": f"Bearer {self._voyage_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": VOYAGE_MODEL,
                "input": text,
                "input_type": "query",
            },
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    async def _voyage_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get batch embeddings from Voyage AI."""
        client = await self._get_client()
        response = await client.post(
            VOYAGE_API_URL,
            headers={
                "Authorization": f"Bearer {self._voyage_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": VOYAGE_MODEL,
                "input": texts,
                "input_type": "query",
            },
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Legacy EmbeddingClient for backward compatibility
class EmbeddingClient(FallbackEmbeddingClient):
    """Alias for FallbackEmbeddingClient (backward compatibility)."""
    pass


# Singleton instance
_embedding_client: FallbackEmbeddingClient | None = None


def get_embedding_client() -> FallbackEmbeddingClient:
    """Get or create embedding client singleton."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = FallbackEmbeddingClient()
    return _embedding_client
