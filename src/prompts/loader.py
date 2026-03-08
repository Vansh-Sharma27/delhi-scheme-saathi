"""Prompt template loader."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template by name.

    Args:
        name: Prompt name without extension (e.g., "system_prompt")

    Returns:
        Prompt template string
    """
    prompt_path = PROMPTS_DIR / f"{name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt template not found: {name}")


def get_system_prompt() -> str:
    """Get the main system prompt for conversation."""
    return load_prompt("system_prompt")


def get_generate_response_prompt() -> str:
    """Get prompt for response generation."""
    return load_prompt("generate_response")
