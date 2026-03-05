"""AWS Bedrock Nova 2 Lite client for LLM operations.

Uses the Bedrock Converse API for chat-style interactions.
Model: amazon.nova-2-lite-v1:0

This is the primary LLM provider, with Grok as fallback.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from src.config import get_settings

logger = logging.getLogger(__name__)

NOVA_MODEL_ID = "amazon.nova-2-lite-v1:0"

# Thread pool for running synchronous boto3 calls
_executor = ThreadPoolExecutor(max_workers=4)


class BedrockLLMClient:
    """Async-compatible Bedrock client for Nova 2 Lite.

    Uses boto3's synchronous client wrapped in a thread pool executor
    for async compatibility. The Converse API provides a unified interface
    for multi-turn conversations.
    """

    def __init__(self) -> None:
        """Initialize Bedrock client with regional configuration."""
        settings = get_settings()
        config = Config(
            region_name=settings.aws_region,
            read_timeout=60,
            connect_timeout=10,
            retries={"max_attempts": 2},
        )
        self._client = boto3.client("bedrock-runtime", config=config)
        self._model_id = settings.bedrock_model or NOVA_MODEL_ID

    async def analyze_message(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]],
        current_state: str,
        user_profile: dict[str, Any],
        system_prompt: str,
    ) -> dict[str, Any]:
        """Analyze user message using Bedrock Converse API.

        Compatible interface with GrokLLMClient for fallback pattern.

        Args:
            user_message: The user's input text
            conversation_history: Previous conversation turns
            current_state: Current FSM state
            user_profile: Extracted user profile data
            system_prompt: System context for the LLM

        Returns:
            dict with intent, life_event, extracted_fields, language, etc.
        """
        import asyncio

        # Build messages for Converse API format
        messages = []

        # Add conversation history (last 10 messages)
        for msg in conversation_history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}],
            })

        # Add current message with analysis instruction
        analysis_prompt = f"""
Analyze the user's message and respond with a JSON object containing:

{{
  "intent": "greeting|question|clarification|selection|location|goodbye|unknown",
  "life_event": "HOUSING|MARRIAGE|CHILDBIRTH|EDUCATION|HEALTH_CRISIS|DEATH_IN_FAMILY|MARITAL_DISTRESS|JOB_LOSS|BUSINESS_STARTUP|WOMEN_EMPOWERMENT|null",
  "extracted_fields": {{
    "age": number or null,
    "gender": "male|female|other" or null,
    "category": "SC|ST|OBC|General|EWS" or null,
    "annual_income": number or null,
    "employment_status": "employed|unemployed|self-employed|student" or null,
    "marital_status": "single|married|widowed|divorced|separated" or null,
    "district": string or null,
    "has_bpl_card": boolean or null
  }},
  "language": "hi|en|hinglish",
  "selected_scheme_id": string or null,
  "needs_clarification": boolean,
  "clarification_question": string or null
}}

Current conversation state: {current_state}
Current user profile: {json.dumps(user_profile, default=str)}

User message: {user_message}

IMPORTANT: Extract information ONLY from what the user explicitly stated.
Do NOT infer or guess values. If unsure, use null.
Respond with ONLY the JSON object, no other text.
"""

        messages.append({
            "role": "user",
            "content": [{"text": analysis_prompt}],
        })

        try:
            # Run synchronous boto3 call in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                _executor,
                lambda: self._client.converse(
                    modelId=self._model_id,
                    system=[{"text": system_prompt}],
                    messages=messages,
                    inferenceConfig={
                        "maxTokens": 1024,
                        "temperature": 0.3,
                    },
                )
            )

            # Extract response text
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])

            if content_blocks:
                output_text = content_blocks[0].get("text", "")

                # Parse JSON from response
                # Sometimes the model wraps JSON in markdown code blocks
                if "```json" in output_text:
                    output_text = output_text.split("```json")[1].split("```")[0]
                elif "```" in output_text:
                    output_text = output_text.split("```")[1].split("```")[0]

                return json.loads(output_text.strip())

            return {"intent": "unknown", "language": "hi"}

        except (BotoCoreError, ClientError) as e:
            logger.error(f"Bedrock API error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Bedrock response as JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Bedrock analysis failed: {e}")
            raise

    async def generate_response(
        self,
        context: dict[str, Any],
        system_prompt: str,
        user_language: str = "hi",
    ) -> str:
        """Generate natural language response using database context.

        Args:
            context: Response context (schemes, documents, warnings, etc.)
            system_prompt: System prompt for response generation
            user_language: Target language (hi, en, hinglish)

        Returns:
            Generated response text
        """
        import asyncio

        generation_prompt = f"""
Generate a helpful, empathetic response in {'Hindi' if user_language == 'hi' else 'English' if user_language == 'en' else 'Hinglish (mix of Hindi and English)'}.

Context:
{json.dumps(context, ensure_ascii=False, indent=2, default=str)}

CRITICAL RULES:
1. NEVER generate scheme facts (eligibility, benefits, documents) - use ONLY data from context
2. Be empathetic for sensitive life events (death, widowhood, illness)
3. Keep response concise (2-4 sentences max)
4. Use simple language appropriate for rural users
5. If presenting schemes, mention key benefits
6. If presenting documents, explain WHERE and HOW to get them
7. If showing rejection warnings, explain how to AVOID rejection

Generate response:
"""

        messages = [{
            "role": "user",
            "content": [{"text": generation_prompt}],
        }]

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                _executor,
                lambda: self._client.converse(
                    modelId=self._model_id,
                    system=[{"text": system_prompt}],
                    messages=messages,
                    inferenceConfig={
                        "maxTokens": 500,
                        "temperature": 0.7,
                    },
                )
            )

            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])

            if content_blocks:
                return content_blocks[0].get("text", "")

            return "मुझे समझने में कठिनाई हो रही है। कृपया दोबारा बताएं।"

        except Exception as e:
            logger.error(f"Bedrock response generation failed: {e}")
            raise

    async def summarize_conversation(
        self,
        messages: list[dict[str, str]],
        current_summary: str | None = None,
    ) -> str:
        """Summarize conversation history.

        Args:
            messages: Recent conversation messages
            current_summary: Existing summary to build upon

        Returns:
            Updated conversation summary
        """
        import asyncio

        summary_prompt = f"""
Summarize this conversation, focusing on:
- User's life situation and needs
- Extracted profile information (age, income, category, etc.)
- Schemes discussed
- Documents mentioned
- Any decisions or next steps

Previous summary: {current_summary or 'None'}

Recent messages:
{json.dumps(messages, ensure_ascii=False, default=str)}

Provide a 2-3 sentence summary in English:
"""

        converse_messages = [{
            "role": "user",
            "content": [{"text": summary_prompt}],
        }]

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                _executor,
                lambda: self._client.converse(
                    modelId=self._model_id,
                    system=[{
                        "text": "You are a conversation summarizer. Create a concise summary of the key information exchanged."
                    }],
                    messages=converse_messages,
                    inferenceConfig={
                        "maxTokens": 200,
                        "temperature": 0.3,
                    },
                )
            )

            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])

            if content_blocks:
                return content_blocks[0].get("text", "")

            return current_summary or ""

        except Exception as e:
            logger.error(f"Bedrock summarization failed: {e}")
            return current_summary or ""


# Singleton instance
_bedrock_client: BedrockLLMClient | None = None


def get_bedrock_client() -> BedrockLLMClient:
    """Get or create Bedrock LLM client singleton."""
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = BedrockLLMClient()
    return _bedrock_client
