"""Grok (xAI) LLM client via OpenAI-compatible API.

Uses grok-4-1-fast-reasoning model for all tasks:
- Intent classification
- Life event detection
- Entity extraction
- Response generation

Single model approach minimizes complexity and latency.
This client is used as fallback when Bedrock is unavailable.
"""

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from src.config import get_settings

logger = logging.getLogger(__name__)


class GrokLLMClient:
    """Async LLM client using xAI Grok via OpenAI SDK."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=settings.xai_api_key,
            base_url=settings.xai_base_url,
        )
        self._model = settings.xai_model

    async def analyze_message(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]],
        current_state: str,
        user_profile: dict[str, Any],
        system_prompt: str,
    ) -> dict[str, Any]:
        """Analyze user message for intent, life event, and entities.

        Single LLM call extracts all needed information:
        - intent: greeting, question, clarification, selection, location, goodbye
        - life_event: HOUSING, HEALTH_CRISIS, etc. (if detected)
        - extracted_fields: age, income, category, etc.
        - language: hi, en, hinglish
        - selected_scheme_id: if user selected a scheme
        """
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (last 5 turns)
        for msg in conversation_history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current message
        messages.append({"role": "user", "content": user_message})

        # Add analysis instruction
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
Current user profile: {json.dumps(user_profile)}

IMPORTANT: Extract information ONLY from what the user explicitly stated.
Do NOT infer or guess values. If unsure, use null.
"""
        messages.append({"role": "user", "content": analysis_prompt})

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if content:
                return json.loads(content)
            return {"intent": "unknown", "language": "hi"}

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {
                "intent": "unknown",
                "life_event": None,
                "extracted_fields": {},
                "language": "hi",
                "selected_scheme_id": None,
                "needs_clarification": False,
                "error": str(e),
            }

    async def generate_response(
        self,
        context: dict[str, Any],
        system_prompt: str,
        user_language: str = "hi",
    ) -> str:
        """Generate natural language response using database context.

        Context includes:
        - user_profile: extracted user information
        - matched_schemes: list of matching schemes with eligibility
        - current_scheme: selected scheme details (if in DETAILS state)
        - documents: required documents with procurement info
        - rejection_warnings: applicable rejection rules
        - nearest_offices: nearby CSC/offices
        - conversation_state: current FSM state
        """
        messages = [{"role": "system", "content": system_prompt}]

        generation_prompt = f"""
Generate a helpful, empathetic response in {'Hindi' if user_language == 'hi' else 'English' if user_language == 'en' else 'Hinglish (mix of Hindi and English)'}.

Context:
{json.dumps(context, ensure_ascii=False, indent=2)}

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
        messages.append({"role": "user", "content": generation_prompt})

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )

            content = response.choices[0].message.content
            return content or "मुझे समझने में कठिनाई हो रही है। कृपया दोबारा बताएं।"

        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return "माफ़ कीजिए, कुछ तकनीकी समस्या है। कृपया थोड़ी देर बाद प्रयास करें।"

    async def summarize_conversation(
        self,
        messages: list[dict[str, str]],
        current_summary: str | None = None,
    ) -> str:
        """Summarize conversation history (called every 5 turns)."""
        prompt_messages = [
            {
                "role": "system",
                "content": "You are a conversation summarizer. Create a concise summary of the key information exchanged.",
            }
        ]

        summary_prompt = f"""
Summarize this conversation, focusing on:
- User's life situation and needs
- Extracted profile information (age, income, category, etc.)
- Schemes discussed
- Documents mentioned
- Any decisions or next steps

Previous summary: {current_summary or 'None'}

Recent messages:
{json.dumps(messages, ensure_ascii=False)}

Provide a 2-3 sentence summary in English:
"""
        prompt_messages.append({"role": "user", "content": summary_prompt})

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=prompt_messages,
                temperature=0.3,
                max_tokens=200,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"Conversation summarization failed: {e}")
            return current_summary or ""


# Global client instance
_grok_client: GrokLLMClient | None = None


def get_grok_client() -> GrokLLMClient:
    """Get or create Grok LLM client singleton."""
    global _grok_client
    if _grok_client is None:
        _grok_client = GrokLLMClient()
    return _grok_client
