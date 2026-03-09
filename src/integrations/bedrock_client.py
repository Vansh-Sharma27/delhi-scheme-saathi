"""AWS Bedrock Nova 2 Lite client for LLM operations.

Uses the Bedrock Converse API for chat-style interactions.
Model: global.amazon.nova-2-lite-v1:0

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
from src.utils.scheme_catalog import get_required_profile_fields_for_life_event

logger = logging.getLogger(__name__)

NOVA_MODEL_ID = "global.amazon.nova-2-lite-v1:0"

_inline_executor: ThreadPoolExecutor | None = None
_background_executor: ThreadPoolExecutor | None = None


def _get_executor(priority: str) -> ThreadPoolExecutor:
    """Return the configured executor for the given task priority."""
    global _inline_executor, _background_executor

    settings = get_settings()
    if priority == "background":
        if _background_executor is None:
            _background_executor = ThreadPoolExecutor(
                max_workers=max(1, settings.ai_background_concurrency)
            )
        return _background_executor

    if _inline_executor is None:
        _inline_executor = ThreadPoolExecutor(
            max_workers=max(1, settings.ai_inline_concurrency)
        )
    return _inline_executor


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
        session_language: str = "hi",
        working_memory: dict[str, Any] | None = None,
        priority: str = "inline",
    ) -> dict[str, Any]:
        """Analyze user message using Bedrock Converse API.

        Compatible interface with GrokLLMClient for fallback pattern.

        Args:
            user_message: The user's input text
            conversation_history: Previous conversation turns
            current_state: Current FSM state
            user_profile: Extracted user profile data
            system_prompt: System context for the LLM
            session_language: User's preferred language from session

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

        # Build missing-fields hint so the LLM knows what to ask next
        missing_fields = []
        field_descriptions = {
            "life_event": "what kind of help they need (housing, health, education, employment, etc.)",
            "age": "the applicant/beneficiary age",
            "gender": "gender (male/female)",
            "category": "caste category (SC/ST/OBC/General/EWS)",
            "annual_income": "approximate annual family income",
        }
        field_priority = get_required_profile_fields_for_life_event(user_profile.get("life_event"))
        for field in field_priority:
            if user_profile.get(field) is None:
                description = field_descriptions.get(field, field)
                missing_fields.append(description)

        missing_hint = ""
        if missing_fields:
            missing_hint = f"\nProfile fields still needed (ask for the FIRST one naturally): {', '.join(missing_fields)}"

        # What field the bot last asked about (for contextual interpretation)
        currently_asking = user_profile.pop("_currently_asking", None)
        currently_asking_hint = ""
        if currently_asking:
            field_descriptions = {
                "life_event": "what kind of help/situation they need",
                "age": "the applicant/beneficiary's age",
                "category": "caste category (SC/ST/OBC/General/EWS)",
                "annual_income": "approximate annual family income",
                "gender": "gender (male/female)",
            }
            desc = field_descriptions.get(currently_asking, currently_asking)
            currently_asking_hint = f"\nThe bot's LAST question asked about: {desc}"

        # Session language hint
        lang_name = {"hi": "Hindi (Devanagari script)", "en": "English", "hinglish": "Hinglish (Hindi-English mix)"}.get(session_language, "Hindi")
        session_lang_hint = f"\nUser's PREFERRED LANGUAGE: {lang_name} — ALL response_text MUST be in this language."

        working_memory_hint = ""
        if working_memory:
            working_memory_hint = (
                "\nWorking memory from earlier turns "
                "(use for continuity only, never override the current message): "
                f"{json.dumps(working_memory, ensure_ascii=False, default=str)}"
            )

        # Add current message with analysis instruction
        analysis_prompt = f"""
Analyze the user's message and respond with a JSON object containing:

{{
  "intent": "greeting|question|clarification|selection|location|goodbye|unknown",
  "action": "change_language|ask_field_reason|skip_field|select_scheme|switch_scheme|request_details|answer_scheme_question|request_application|request_handoff|start_over|goodbye|answer_field|none",
  "life_event": "HOUSING|MARRIAGE|CHILDBIRTH|EDUCATION|HEALTH_CRISIS|DEATH_IN_FAMILY|MARITAL_DISTRESS|JOB_LOSS|BUSINESS_STARTUP|WOMEN_EMPOWERMENT|null",
  "extracted_fields": {{
    "age": number or null,
    "gender": "male|female|other" or null (include when directly entailed by first-person self-description or gendered spouse terminology),
    "category": "SC|ST|OBC|General|EWS" or null,
    "annual_income": number or null,
    "employment_status": "employed|unemployed|self-employed|student" or null,
    "marital_status": "single|married|widowed|divorced|separated" or null (include widowed when directly entailed by first-person spouse-loss wording),
    "district": string or null,
    "has_bpl_card": boolean or null
  }},
  "language": "hi|en|hinglish",
  "selected_scheme_id": string or null,
  "needs_clarification": boolean,
  "clarification_question": string or null,
  "response_text": "A natural conversational response IN THE USER'S PREFERRED LANGUAGE (see rules below)"
}}

Current conversation state: {current_state}
Current user profile: {json.dumps(user_profile, default=str)}
{missing_hint}
{currently_asking_hint}
{session_lang_hint}
{working_memory_hint}

User message: {user_message}

IMPORTANT RULES:
1. Extract information from what the user explicitly states and from facts that are directly entailed by the user's own first-person wording.
1b. Use working memory only for continuity. If the current user message conflicts with memory, trust the current user message.
1c. Direct semantic entailment is allowed only when the wording itself determines the field through self-description, role labels, or relationship terms. Do NOT use weak stereotypes, demographic assumptions, or indirect guesses. If more than one interpretation is plausible, use null.
1d. Treat first-person spousal relationship terms as directly entailed evidence, not a guess. If the user refers to their own spouse in a way that logically determines widow/widower context, extract the corresponding marital_status. If the spouse term itself is gendered and therefore logically determines the user's schema gender, extract that gender too.
2. CONTEXTUAL EXTRACTION: If the bot last asked about a specific field and the user replies with a bare number or short answer, interpret it in that context. For example, if the bot asked about age and the user replies "19", extract age=19.
3. VALIDATION: When extracting fields, validate the values:
   - age: must be between 1-120. If user gives birth year (e.g., "2005"), calculate age. If invalid (0, negative, >120), set to null.
   - category: must be one of SC/ST/OBC/General/EWS. Map common variations ("open"→General, "backward class"→OBC).
   - annual_income: must be positive number. Convert lakhs/crores appropriately.
4. For response_text, generate a warm, natural reply following these scenarios:
   a) User answered the asked question correctly → acknowledge briefly and ask for the NEXT missing field
   b) User gave INVALID value (e.g., age=200, unrecognized category) → gently explain what's needed: "I need age as a number between 1-120" or "Please specify SC/ST/OBC/General/EWS"
   c) User gave DIFFERENT info than asked → acknowledge what they shared, then gently circle back: "Thanks for that! I also need to know [asked field] — could you share that?"
   d) User said something unrelated or confusing → be understanding and redirect: "I appreciate that! To find the right schemes, could you tell me [asked field]?"
   e) User says "I don't know" or refuses → be respectful: "No problem! We can work with approximate values or skip this for now." Then move to the next field.
   f) User asks a question like "why do you need this?" → explain briefly why the field matters (e.g., "Age helps us find age-appropriate schemes") and re-ask gently
   g) User sends just a number → interpret in context (age if asking age, income if asking income) and acknowledge accordingly
   h) CRITICAL: response_text MUST be in the user's PREFERRED LANGUAGE ({lang_name}). Do NOT switch languages.
   i) Keep it 1-3 sentences max, conversational tone
   j) NEVER mention any scheme names, benefits, eligibility, or document details — that comes from the database later
   k) If a field is already explicit or directly entailed by the user's wording, do not ask for that same field again. Ask only for the next missing field.
5. Set action conservatively:
   - change_language only when the user explicitly asks for another language
   - ask_field_reason when the user asks why a requested field matters
   - skip_field when they say they do not know / want to skip
   - request_application only for explicit apply/application requests
   - request_handoff only for explicit human-help/CSC/center requests
   - answer_scheme_question for a direct question about an already-selected or already-presented scheme
   - request_details for generic translation/document/detail follow-ups
   - select_scheme or switch_scheme only when the user picks a specific scheme
   - otherwise use answer_field or none
Respond with ONLY the JSON object, no other text.
"""

        messages.append({
            "role": "user",
            "content": [{"text": analysis_prompt}],
        })

        try:
            # Run synchronous boto3 call in thread pool
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                _get_executor(priority),
                lambda: self._client.converse(
                    modelId=self._model_id,
                    system=[{"text": system_prompt}],
                    messages=messages,
                    inferenceConfig={
                        "maxTokens": 1024,
                        "temperature": 0,
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

    async def judge_scheme_relevance(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]],
        current_state: str,
        user_profile: dict[str, Any],
        candidate_schemes: list[dict[str, Any]],
        session_language: str = "hi",
        working_memory: dict[str, Any] | None = None,
        priority: str = "inline",
    ) -> dict[str, Any]:
        """Ask Bedrock to sanity-check deterministic scheme candidates."""
        import asyncio

        language_label = {
            "hi": "Hindi (Devanagari)",
            "en": "English",
            "hinglish": "Hinglish",
        }.get(session_language, "English")

        messages = []
        for msg in conversation_history[-8:]:
            messages.append(
                {
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}],
                }
            )

        working_memory_hint = ""
        if working_memory:
            working_memory_hint = (
                f"\nWorking memory: {json.dumps(working_memory, ensure_ascii=False, default=str)}"
            )

        prompt = f"""
You are a relevance judge for a government-scheme assistant.

Return ONLY a JSON object with this shape:
{{
  "should_clarify": boolean,
  "clarification_question": string or null,
  "overall_confidence": number between 0 and 1,
  "candidate_scores": [
    {{
      "scheme_id": string,
      "relevance_score": number between 0 and 1,
      "topic_match": boolean or null,
      "reason": string or null
    }}
  ]
}}

Conversation state: {current_state}
User profile: {json.dumps(user_profile, ensure_ascii=False, default=str)}
{working_memory_hint}
Latest user message: {user_message}
Preferred clarification language: {language_label}
Deterministic candidate schemes: {json.dumps(candidate_schemes, ensure_ascii=False, default=str)}

Rules:
1. Never invent scheme facts beyond the provided candidate data.
2. Score semantic fit for the user's actual need, not just lexical overlap.
3. Use working memory only to understand continuity; never let it override the latest user request.
4. If the candidate list looks cross-domain, inconsistent with the conversation, or low-confidence, set should_clarify=true.
5. clarification_question must be short and in the preferred clarification language.
6. If the candidates look strong and on-topic, set should_clarify=false.
"""

        messages.append({"role": "user", "content": [{"text": prompt}]})

        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                _get_executor(priority),
                lambda: self._client.converse(
                    modelId=self._model_id,
                    system=[{"text": "You audit relevance between user needs and deterministic scheme candidates."}],
                    messages=messages,
                    inferenceConfig={"maxTokens": 768, "temperature": 0.1},
                ),
            )
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])
            if content_blocks:
                output_text = content_blocks[0].get("text", "")
                if "```json" in output_text:
                    output_text = output_text.split("```json")[1].split("```")[0]
                elif "```" in output_text:
                    output_text = output_text.split("```")[1].split("```")[0]
                return json.loads(output_text.strip())
            return {"should_clarify": False, "candidate_scores": []}
        except (BotoCoreError, ClientError) as e:
            logger.error("Bedrock relevance judge API error: %s", e)
            raise
        except json.JSONDecodeError as e:
            logger.error("Failed to parse Bedrock relevance response as JSON: %s", e)
            raise
        except Exception as e:
            logger.error("Bedrock relevance judging failed: %s", e)
            raise

    async def generate_response(
        self,
        context: dict[str, Any],
        system_prompt: str,
        user_language: str = "hi",
        priority: str = "inline",
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
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                _get_executor(priority),
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
        priority: str = "background",
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
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                _get_executor(priority),
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
