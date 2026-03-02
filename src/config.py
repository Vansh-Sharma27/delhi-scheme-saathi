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

    # Voyage AI for embeddings
    voyage_api_key: str = Field(default="", description="Voyage AI API key")

    # Bhashini Voice (Phase 5)
    bhashini_api_key: str = Field(default="", description="Bhashini API key")
    bhashini_user_id: str = Field(default="", description="Bhashini user ID")
    bhashini_ulca_api_key: str = Field(default="", description="Bhashini ULCA API key")

    # AWS (Phase 6)
    aws_region: str = Field(default="ap-south-1", description="AWS region")
    session_table_name: str = Field(default="dss-sessions", description="DynamoDB table")
    audio_bucket: str = Field(default="dss-audio", description="S3 bucket for audio")

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
