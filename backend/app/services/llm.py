"""
JARVIS - LLM Service (Modular Dual AI Provider Router)
Supports:
  - Gemini Provider (Online - Google)
  - NVIDIA Provider (Online - NVIDIA NIM / OpenAI Compatible)
  - Ollama Provider (Offline local - qwen2.5:3b)

Centralized AI Router with task routing, capability-aware selection,
and bounded fallback.
"""
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import AsyncGenerator

import httpx
import google.generativeai as genai
import ollama as ollama_client

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def _safe_provider_message(exc: Exception) -> str:
    """Return a sanitized error summary without exposing keys, tokens, or raw internals."""
    message = str(exc).lower()
    if any(token in message for token in ("401", "403", "unauthorized", "forbidden", "api key", "apikey", "secret", "bearer", "token")):
        return "provider authentication failed or misconfigured"
    if any(token in message for token in ("429", "rate limit", "quota", "exceeded")):
        return "provider rate limit exceeded"
    if any(token in message for token in ("not found", "404", "model_not_found", "unsupported")):
        return "provider model unsupported or not found"
    if any(token in message for token in ("connect", "timeout", "unreachable", "econnrefused", "name resolution")):
        return "provider network connection failed"
    return "provider service unavailable"


class TaskType(str, Enum):
    GENERAL_CHAT = "general_chat"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    TOOL_CALLING = "tool_calling"
    CODING = "coding"
    SPECIALIZED = "specialized"


# ─────────────────────────────────────────────
# Base Interface — all providers must implement this
# ─────────────────────────────────────────────
class BaseLLMProvider(ABC):
    name: str = "base"
    model_name: str = ""

    @abstractmethod
    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate a response for the given prompt."""
        ...

    @abstractmethod
    async def stream(self, prompt: str, system: str = "") -> AsyncGenerator[str, None]:
        """Stream a response token by token."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this provider is currently reachable."""
        ...

    def supports_task(self, task_type: str) -> bool:
        """Return True if this provider/model capability supports the given task type."""
        return True


# ─────────────────────────────────────────────
# Gemini Provider (Online - Google)
# ─────────────────────────────────────────────
class GeminiProvider(BaseLLMProvider):
    name = "gemini"

    def __init__(self):
        self.model_name = (settings.gemini_model or "gemini-1.5-flash").strip()
        self.api_key = (settings.gemini_api_key or "").strip()
        if self.api_key:
            genai.configure(api_key=self.api_key)
        self.model = None
        if self.api_key and self.model_name:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 2048,
                },
            )

    def supports_task(self, task_type: str) -> bool:
        # Gemini 1.5 Flash supports general chat, vision, multimodal, tool calling, coding, specialized
        return True

    async def is_available(self) -> bool:
        if not self.api_key or not self.model_name:
            logger.warning("Gemini unavailable: missing API key or model configuration.")
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("https://www.google.com")
                return response.status_code == 200
        except Exception as exc:
            logger.warning("Gemini unavailable: %s", _safe_provider_message(exc))
            return False

    async def generate(self, prompt: str, system: str = "") -> str:
        if self.model is None:
            raise RuntimeError("Gemini provider is missing API key or model configuration.")
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        try:
            response = await self.model.generate_content_async(full_prompt)
            if not response or not hasattr(response, "text") or not response.text:
                raise RuntimeError("Gemini returned an empty response.")
            return response.text
        except Exception as exc:
            logger.warning("Gemini generation failed: %s", _safe_provider_message(exc))
            raise RuntimeError(f"Gemini is unavailable: {_safe_provider_message(exc)}") from exc

    async def stream(self, prompt: str, system: str = "") -> AsyncGenerator[str, None]:
        if self.model is None:
            raise RuntimeError("Gemini provider is missing API key or model configuration.")
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        try:
            async for chunk in await self.model.generate_content_async(full_prompt, stream=True):
                if chunk and hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
        except Exception as exc:
            logger.warning("Gemini streaming failed: %s", _safe_provider_message(exc))
            raise RuntimeError(f"Gemini streaming unavailable: {_safe_provider_message(exc)}") from exc


# ─────────────────────────────────────────────
# NVIDIA Provider (Online - NVIDIA NIM / OpenAI compatible API)
# ─────────────────────────────────────────────
class NVIDIAProvider(BaseLLMProvider):
    name = "nvidia"

    def __init__(self):
        self.api_key = (settings.nvidia_api_key or "").strip()
        self.base_url = (settings.nvidia_base_url or "https://integrate.api.nvidia.com/v1").strip().rstrip("/")
        self.model_name = (settings.nvidia_model or "meta/llama-3.1-70b-instruct").strip()

    def supports_task(self, task_type: str) -> bool:
        model_lower = self.model_name.lower()
        if task_type in (TaskType.VISION, TaskType.MULTIMODAL):
            return any(k in model_lower for k in ("vision", "vl", "multimodal", "neva"))
        if task_type == TaskType.TOOL_CALLING:
            return any(k in model_lower for k in ("tool", "fc", "function"))
        return True

    async def is_available(self) -> bool:
        if not self.api_key or not self.model_name:
            return False
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with httpx.AsyncClient(timeout=5.0) as client:
                url = f"{self.base_url}/models"
                response = await client.get(url, headers=headers)
                if response.status_code in (200, 204):
                    data = response.json()
                    model_ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
                    return self.model_name in model_ids
                return False
        except Exception as exc:
            logger.warning("NVIDIA unavailable: %s", _safe_provider_message(exc))
            return False

    async def generate(self, prompt: str, system: str = "") -> str:
        if not self.api_key or not self.model_name:
            raise RuntimeError("NVIDIA provider is missing API key or model configuration.")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{self.base_url}/chat/completions"
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    raise RuntimeError(f"NVIDIA API status code {response.status_code}")
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning("NVIDIA generation failed: %s", _safe_provider_message(exc))
            raise RuntimeError(f"NVIDIA provider unavailable: {_safe_provider_message(exc)}") from exc

    async def stream(self, prompt: str, system: str = "") -> AsyncGenerator[str, None]:
        if not self.api_key or not self.model_name:
            raise RuntimeError("NVIDIA provider is missing API key or model configuration.")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{self.base_url}/chat/completions"
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"NVIDIA stream status code {response.status_code}")
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                content = data_json["choices"][0]["delta"].get("content")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
        except Exception as exc:
            logger.warning("NVIDIA streaming failed: %s", _safe_provider_message(exc))
            raise RuntimeError(f"NVIDIA provider streaming unavailable: {_safe_provider_message(exc)}") from exc


# ─────────────────────────────────────────────
# Ollama Provider (Offline - Local qwen2.5:3b)
# ─────────────────────────────────────────────
class OllamaProvider(BaseLLMProvider):
    name = "ollama"

    def __init__(self):
        host = settings.ollama_base_url
        if "localhost" in host:
            host = host.replace("localhost", "127.0.0.1")
        self.client = ollama_client.AsyncClient(host=host)
        self.model_name = settings.ollama_model

    def supports_task(self, task_type: str) -> bool:
        if task_type in (TaskType.VISION, TaskType.MULTIMODAL, TaskType.TOOL_CALLING):
            return False
        return True

    async def is_available(self) -> bool:
        base_url = settings.ollama_base_url.rstrip("/")
        urls_to_try = [base_url]
        if "localhost" in base_url:
            urls_to_try.append(base_url.replace("localhost", "127.0.0.1"))

        for url in urls_to_try:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(f"{url}/api/tags", timeout=3.0)
                    if response.status_code == 200:
                        data = response.json()
                        models = data.get("models", [])
                        installed = [m.get("name", "") for m in models if isinstance(m, dict)]
                        if self.model_name in installed or any(name.startswith(f"{self.model_name}:") for name in installed):
                            return True
            except Exception as exc:
                logger.warning("Ollama check failed for %s: %s", url, _safe_provider_message(exc))
                continue

        return False

    async def generate(self, prompt: str, system: str = "", num_predict: int = -1) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options: dict = {
            "keep_alive": "60m",   # Keep model loaded 60 min; prevents cold reload
            "num_ctx": 2048,       # Reduced context → lower RAM pressure & faster decode
        }
        if num_predict > 0:
            options["num_predict"] = num_predict  # Voice mode: limit output tokens

        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
            )
            if isinstance(response, dict):
                return response["message"]["content"]
            return response.message.content
        except Exception as exc:
            logger.warning("Ollama generation failed: %s", _safe_provider_message(exc))
            raise RuntimeError(f"Ollama is unavailable or model is not installed: {_safe_provider_message(exc)}") from exc

    async def stream(self, prompt: str, system: str = "", num_predict: int = -1) -> AsyncGenerator[str, None]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options: dict = {
            "keep_alive": "60m",   # Keep model hot
            "num_ctx": 2048,       # Smaller context = faster decode
        }
        if num_predict > 0:
            options["num_predict"] = num_predict  # Voice: cap output at ~120 tokens

        try:
            async for chunk in await self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options=options,
            ):
                content = chunk["message"]["content"] if isinstance(chunk, dict) else chunk.message.content
                if content:
                    yield content
        except Exception as exc:
            logger.warning("Ollama streaming failed: %s", _safe_provider_message(exc))
            raise RuntimeError(f"Ollama streaming unavailable: {_safe_provider_message(exc)}") from exc


# ─────────────────────────────────────────────
# Centralized AI Router & LLM Service
# ─────────────────────────────────────────────
class LLMService:
    def __init__(self):
        self.gemini = GeminiProvider()
        self.nvidia = NVIDIAProvider()
        self.ollama = OllamaProvider()

        self._providers: dict[str, BaseLLMProvider] = {
            "gemini": self.gemini,
            "nvidia": self.nvidia,
            "ollama": self.ollama,
        }

        self.last_provider_used: str | None = None
        self.last_model_used: str | None = None
        self.last_fallback_info: dict | None = None

    def detect_task_type(self, prompt: str, system: str = "") -> str:
        text = f"{system} {prompt}".lower()
        if any(k in text for k in ("image", "photo", "picture", "visual", "vision")):
            return TaskType.VISION.value
        if any(k in text for k in ("function_call", "tool_call", "json_schema")):
            return TaskType.TOOL_CALLING.value
        if any(k in text for k in ("```", "def ", "class ", "function ", "import ", "python", "javascript", "code", "debug", "refactor", "algorithm")):
            return TaskType.CODING.value
        if any(k in text for k in ("solve math", "equation", "proof", "specialized")):
            return TaskType.SPECIALIZED.value
        return TaskType.GENERAL_CHAT.value

    def get_provider_chain(self, task_type: str = TaskType.GENERAL_CHAT.value, provider_override: str | None = None) -> list[BaseLLMProvider]:
        configured_provider = (provider_override or settings.active_default_provider).lower()

        # Explicit provider requested by caller or config
        if configured_provider in self._providers:
            primary = self._providers[configured_provider]
            chain = [primary]
            for p_name, p_inst in self._providers.items():
                if p_name != configured_provider:
                    chain.append(p_inst)
            return chain

        # Default priority:
        # 1. Ollama  - local/offline primary
        # 2. NVIDIA  - online secondary
        # 3. Gemini  - online fallback
        if task_type in {
            TaskType.CODING.value,
            TaskType.GENERAL_CHAT.value,
        }:
            chain = [self.ollama, self.nvidia, self.gemini]
        else:
            chain = [self.ollama, self.nvidia, self.gemini]

        return chain

    async def generate(
        self,
        prompt: str,
        system: str = "",
        task_type: str | None = None,
        provider_override: str | None = None,
    ) -> str:
        resolved_task = task_type or self.detect_task_type(prompt, system)
        chain = self.get_provider_chain(resolved_task, provider_override)

        attempted = []
        last_error = None

        for provider in chain:
            if not provider.supports_task(resolved_task):
                logger.info("Skipping %s for task %s (capability unsupported)", provider.name, resolved_task)
                continue

            if not await provider.is_available():
                logger.info("Skipping %s for task %s (not reachable/configured)", provider.name, resolved_task)
                continue

            attempted.append(provider.name)
            try:
                result = await provider.generate(prompt, system)
                self.last_provider_used = provider.name
                self.last_model_used = provider.model_name
                self.last_fallback_info = {
                    "fallback_used": len(attempted) > 1,
                    "primary_provider": attempted[0],
                    "actual_provider": provider.name,
                    "attempted_providers": attempted,
                }
                return result
            except Exception as exc:
                logger.warning("Provider %s failed during generate: %s", provider.name, _safe_provider_message(exc))
                last_error = exc
                continue

        raise RuntimeError("No available AI provider could fulfill the request.") from last_error

    async def stream(
        self,
        prompt: str,
        system: str = "",
        task_type: str | None = None,
        provider_override: str | None = None,
    ) -> AsyncGenerator[str, None]:
        resolved_task = task_type or self.detect_task_type(prompt, system)
        chain = self.get_provider_chain(resolved_task, provider_override)

        attempted = []
        last_error = None

        for provider in chain:
            if not provider.supports_task(resolved_task):
                logger.info("Skipping %s for task %s (capability unsupported)", provider.name, resolved_task)
                continue

            if not await provider.is_available():
                logger.info("Skipping %s for task %s (not reachable/configured)", provider.name, resolved_task)
                continue

            attempted.append(provider.name)
            try:
                first_token = True
                async for token in provider.stream(prompt, system):
                    if first_token:
                        self.last_provider_used = provider.name
                        self.last_model_used = provider.model_name
                        self.last_fallback_info = {
                            "fallback_used": len(attempted) > 1,
                            "primary_provider": attempted[0],
                            "actual_provider": provider.name,
                            "attempted_providers": attempted,
                        }
                        first_token = False
                    yield token
                return
            except Exception as exc:
                logger.warning("Provider %s failed during stream: %s", provider.name, _safe_provider_message(exc))
                last_error = exc
                continue

        raise RuntimeError("No available AI provider could fulfill the request.") from last_error

    async def status(self) -> dict:
        """Return availability status of all providers."""
        return {
            "gemini_available": await self.gemini.is_available(),
            "nvidia_available": await self.nvidia.is_available(),
            "ollama_available": await self.ollama.is_available(),
            "active_provider": settings.active_default_provider,
            "last_provider_used": self.last_provider_used,
            "last_model_used": self.last_model_used,
        }


# Singleton instance
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
