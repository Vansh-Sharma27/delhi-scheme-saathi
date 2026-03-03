"""Application configuration loaded from environment variables.

Uses Pydantic settings for type-safe configuration with validation.
All configuration is immutable (frozen) after initialization.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Immutable application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/delhi_scheme_saathi",
        description="PostgreSQL connection URL"
    )

    # xAI (Grok) API - OpenAI-compatible
    xai_api_key: str = Field(default="", description="xAI API key")
    xai_base_url: str = Field(
        default="https://api.x.ai/v1",
        description="xAI API base URL"
    )
    xai_model: str = Field(
        default="grok-4-1-fast-reasoning",
        description="Grok model ID"
    )

    # Telegram
    telegram_bot_token: str = Field(default="", description="Telegram Bot API token")

    # Embeddings (Jina AI primary, Voyage AI fallback)
    # Jina AI - 10M free tokens, 89 languages including Hindi/Bengali/Urdu
    jina_api_key: str = Field(default="", description="Jina AI API key (primary)")
    jina_model: str = Field(default="jina-embeddings-v3", description="Jina embedding model")
    # Voyage AI - fallback for reliability
    voyage_api_key: str = Field(default="", description="Voyage AI API key (fallback)")

    # Sarvam AI Voice (primary voice provider)
    sarvam_api_key: str = Field(default="", description="Sarvam AI API key")

    # Bhashini Voice (fallback)
    bhashini_api_key: str = Field(default="", description="Bhashini API key")
    bhashini_user_id: str = Field(default="", description="Bhashini user ID")
    bhashini_ulca_api_key: str = Field(default="", description="Bhashini ULCA API key")

    # AWS (Phase 6)
    aws_region: str = Field(default="ap-south-1", description="AWS region")
    session_table_name: str = Field(default="dss-sessions", description="DynamoDB table")
    audio_bucket: str = Field(default="dss-audio", description="S3 bucket for audio")
    # AWS Bedrock (primary LLM)
    bedrock_model: str = Field(
        default="amazon.nova-2-lite-v1:0",
        description="Bedrock model ID for LLM"
    )
    use_bedrock: bool = Field(
        default=False,
        description="Use Bedrock as primary LLM (requires AWS credentials)"
    )

    # Application
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
