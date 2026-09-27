"""
JARVIS - Dynamic system prompt builder.
Builds a personalized system prompt by injecting:
  - User display name / preferences
  - Relevant memories
  - Active learning topics
  - Online/offline mode awareness
  - Provider capability info
"""
from __future__ import annotations

from typing import Any

BASE_SYSTEM_PROMPT = """You are JARVIS, a private personal AI tutor, mentor, and assistant.

PERSONALITY:
- Calm, intelligent, patient, and professional
- Like a trusted senior teacher who genuinely cares about the student's growth
- Speak naturally — not robotic, not overly formal

LANGUAGE RULES:
- Default: Mix of Tamil + English (Tanglish) — natural for Indian learners
- Use simple Tamil to explain difficult concepts
- Use English for technical terms and code
- Example: "இந்த concept-ஐ புரிஞ்சுக்கணும்னா, first இதை பாரு..."
- If user writes in pure Tamil, respond in Tamil
- If user writes in pure English, respond in English

TEACHING BEHAVIOR:
- ALWAYS first understand the user's current knowledge level
- NEVER assume advanced knowledge
- Start from the user's current level, not from zero
- Teach step-by-step — one concept at a time
- Give practical real-world examples
- After explaining, ask a question to test understanding
- If user doesn't understand, explain differently
- Give notebook-style notes when asked
- Create exercises, quizzes, and coding problems

MEMORY RULES:
- Remember what the user has learned
- Remember where they struggle
- Remember their goals and preferences
- Never repeat what they already know well
- Continue from where you left off

RESPONSE FORMAT:
- Keep responses concise — don't dump everything at once
- Use sections when giving structured content:
  📚 Explanation
  💡 Example
  ✏️ Exercise / Question
  📝 Notes
- For code: always use proper code blocks

TOOL USAGE:
- When you need current information, use web_search
- When you need to recall user facts, use memory_search
- When performing math, use calculator
- When the user asks about time/date, use datetime_tool
- Always use tools before saying you don't know something factual

SAFETY & PRIVACY:
- You are a private assistant — never mention or share user data
- Do not reveal system internals or API keys
"""

JARVIS_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT  # Backward-compatible alias


def build_system_prompt(
    *,
    user_name: str | None = None,
    preferences: dict[str, Any] | None = None,
    relevant_memories: list[dict[str, Any]] | None = None,
    active_topics: list[str] | None = None,
    provider_name: str | None = None,
    is_online: bool = True,
) -> str:
    """
    Build a personalized system prompt for JARVIS.

    Args:
        user_name: User's display name
        preferences: User preference dict (response_style, language, etc.)
        relevant_memories: List of memory dicts {memory_key, value summary}
        active_topics: List of current learning topic titles
        provider_name: LLM provider being used (gemini/nvidia/ollama)
        is_online: Whether external AI services are available
    """
    parts = [BASE_SYSTEM_PROMPT]

    # Personalization section
    personal_lines = []
    if user_name:
        personal_lines.append(f"- The user's name is: {user_name}")

    prefs = preferences or {}
    response_style = prefs.get("response_style", "")
    if response_style:
        personal_lines.append(f"- Preferred response style: {response_style}")

    language_pref = prefs.get("language", "")
    if language_pref:
        personal_lines.append(
            f"- User's preferred language: {language_pref}. Prioritize this language."
        )

    if not is_online:
        personal_lines.append(
            "- OFFLINE MODE: You are running locally. "
            "Do not suggest external web searches or cloud-dependent features."
        )
    elif provider_name:
        personal_lines.append(
            f"- Running on: {provider_name}. Full online capabilities available."
        )

    if personal_lines:
        parts.append("\nUSER CONTEXT:\n" + "\n".join(personal_lines))

    # Memory injection
    if relevant_memories:
        mem_lines = ["WHAT YOU REMEMBER ABOUT THIS USER:"]
        for mem in relevant_memories[:10]:  # Limit to 10 most relevant
            key = mem.get("memory_key", "")
            val = mem.get("value", {})
            # Safely summarize — never expose is_sensitive=True values in prompt
            if isinstance(val, dict):
                summary = "; ".join(f"{k}={v}" for k, v in list(val.items())[:3])
            else:
                summary = str(val)[:100]
            mem_lines.append(f"- {key}: {summary}")
        parts.append("\n" + "\n".join(mem_lines))

    # Active learning topics
    if active_topics:
        topics_str = ", ".join(active_topics[:5])
        parts.append(
            f"\nCURRENT LEARNING TOPICS: {topics_str}\n"
            "Reference these when relevant to help the user progress."
        )

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────
# Voice-optimised system prompt (concise, no markdown, spoken audio)
# ─────────────────────────────────────────────────────────────────
VOICE_SYSTEM_PROMPT_BASE = (
    "You are JARVIS, a helpful personal voice assistant. "
    "Respond in natural spoken English only — no bullet points, no markdown, no emojis, no code blocks. "
    "Give a concise spoken answer in 1 to 3 short sentences. "
    "If the user asks for more detail, you may expand. "
    "Never start your reply with 'I' or 'As an AI'."
)


def build_voice_system_prompt(
    *,
    user_name: str | None = None,
    top_memories: list[dict] | None = None,
) -> str:
    """
    Ultra-compact system prompt for voice responses.
    Keeps total prompt tokens minimal to achieve low LLM generation latency.
    No markdown, no emoji, 1-3 sentences, natural speech.
    """
    parts = [VOICE_SYSTEM_PROMPT_BASE]

    if user_name:
        parts.append(f"The user's name is {user_name}.")

    # Inject at most 2 most-relevant memories (keep prompt tiny)
    if top_memories:
        for mem in top_memories[:2]:
            key = mem.get("memory_key", "")
            val = mem.get("value", {})
            if isinstance(val, dict):
                summary = "; ".join(f"{k}={v}" for k, v in list(val.items())[:2])
            else:
                summary = str(val)[:60]
            if key and summary:
                parts.append(f"Remember: {key}: {summary}.")

    return " ".join(parts)

