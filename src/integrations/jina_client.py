"""Jina AI embedding client for multilingual Indian language support.

Uses jina-embeddings-v3 model:
- 89 languages including Hindi, Bengali, Urdu
- 1024 dimensions (matches pgvector schema)
- Task-specific LoRA adapters (retrieval.query, retrieval.passage)
- Matryoshka representations (dimension truncation if needed)

API Documentation: https://jina.ai/embeddings/
"""

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v3"
EMBEDDING_DIM = 1024


@dataclass
class EmbeddingResult:
    """Result from embedding request."""
    embedding: list[float]
    tokens_used: int = 0


class JinaEmbeddingClient:
    """Async client for Jina AI embeddings.

    Jina embeddings provide excellent multilingual support for Indian languages:
    - Hindi, Bengali, Urdu, Tamil, Telugu, and more
    - 10M free tokens on signup
    - Task-specific adapters for better retrieval quality
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize Jina client.

        Args:
            api_key: Jina AI API key (or from settings if not provided)
        """
        self._api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def get_embedding(
        self,
        text: str,
        task: str = "retrieval.query",
    ) -> list[float]:
        """Get embedding vector for text.

        Args:
            text: Text to embed (max ~8192 tokens)
            task: Task type for LoRA adapter selection:
                - "retrieval.query": Optimized for search queries (user input)
                - "retrieval.passage": Optimized for documents (scheme descriptions)
                - "text-matching": For semantic similarity
                - "classification": For text classification

        Returns:
            1024-dimensional embedding vector (L2 normalized)

        Raises:
            ValueError: If API key not configured
            httpx.HTTPError: On API request failure
        """
        if not self._api_key:
            logger.warning("Jina API key not configured")
            raise ValueError("Jina API key not configured")

        client = await self._get_client()

        try:
            response = await client.post(
                JINA_API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": JINA_MODEL,
                    "input": [text],
                    "task": task,
                    "normalized": True,  # L2 normalized for cosine similarity
                    "dimensions": EMBEDDING_DIM,
                },
            )
            response.raise_for_status()
            data = response.json()

            embedding = data["data"][0]["embedding"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            logger.debug(f"Jina embedding: {len(embedding)} dims, {tokens} tokens")

            return embedding

        except httpx.HTTPStatusError as e:
            logger.error(f"Jina API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Jina embedding request failed: {e}")
            raise

    async def get_embeddings_batch(
        self,
        texts: list[str],
        task: str = "retrieval.query",
    ) -> list[list[float]]:
        """Get embeddings for multiple texts in a single request.

        More efficient for batch operations (e.g., seeding database).

        Args:
            texts: List of texts to embed (max ~2048 per batch)
            task: Task type (see get_embedding for options)

        Returns:
            List of 1024-dimensional embedding vectors

        Raises:
            ValueError: If API key not configured or empty texts
        """
        if not self._api_key:
            raise ValueError("Jina API key not configured")

        if not texts:
            return []

        client = await self._get_client()

        try:
            response = await client.post(
                JINA_API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": JINA_MODEL,
                    "input": texts,
                    "task": task,
                    "normalized": True,
                    "dimensions": EMBEDDING_DIM,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Extract embeddings preserving order
            embeddings = [item["embedding"] for item in data["data"]]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            logger.debug(f"Jina batch: {len(embeddings)} embeddings, {tokens} tokens")

            return embeddings

        except httpx.HTTPStatusError as e:
            logger.error(f"Jina API batch error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Jina batch embedding failed: {e}")
            raise

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Singleton instance
_jina_client: Optional[JinaEmbeddingClient] = None


def get_jina_client() -> JinaEmbeddingClient:
    """Get or create Jina embedding client singleton."""
    global _jina_client
    if _jina_client is None:
        settings = get_settings()
        _jina_client = JinaEmbeddingClient(api_key=settings.jina_api_key)
    return _jina_client


def configure_jina_client(api_key: str) -> JinaEmbeddingClient:
    """Configure and return Jina client with specific API key."""
    global _jina_client
    _jina_client = JinaEmbeddingClient(api_key=api_key)
    return _jina_client
