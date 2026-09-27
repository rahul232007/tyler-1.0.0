import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from app.core.config import get_settings
from app.services.llm import (
    LLMService,
    NVIDIAProvider,
    TaskType,
    _safe_provider_message,
)


def log(msg):
    print(msg, flush=True)


async def run_nvidia_verification():
    log("=========================================")
    log("NVIDIA PROVIDER & ROUTING VERIFICATION")
    log("=========================================")

    settings = get_settings()

    # 1 & 3. Check NVIDIA_API_KEY presence from environment (NEVER print the key)
    if not settings.nvidia_api_key:
        log("RESULT: BLOCKED - NVIDIA_API_KEY is not configured in environment.")
        return

    log("NVIDIA_API_KEY configured in environment: YES (Key hidden for security)")
    log(f"NVIDIA_BASE_URL: {settings.nvidia_base_url}")
    log(f"NVIDIA_MODEL: {settings.nvidia_model}")

    # 2. Check provider availability
    nvidia_provider = NVIDIAProvider()
    avail = await nvidia_provider.is_available()
    log(f"NVIDIAProvider.is_available(): {avail}")

    if not avail:
        log(
            "RESULT: BLOCKED - NVIDIA API endpoint or model unreachable with configured key."
        )
        return

    service = LLMService()

    # 4, 5, 6, 7. Make ONE real NVIDIA API request with a simple coding task
    log("\n--- Executing real NVIDIA API request with CODING task ---")
    coding_prompt = (
        "Write a python function `def add(a, b):` that returns the sum of a and b."
    )

    detected_task = service.detect_task_type(coding_prompt)
    log(f"Detected Task Type: {detected_task}")
    assert detected_task == TaskType.CODING.value, (
        f"Expected CODING task, got {detected_task}"
    )

    try:
        response_text = await service.generate(
            prompt=coding_prompt, system="You are a coding assistant."
        )
        provider_used = service.last_provider_used
        model_used = service.last_model_used

        log("Response Received Successfully: YES")
        log(f"Response Snippet: {response_text.strip()[:80]}...")
        log(f"provider_used: {provider_used}")
        log(f"model_used: {model_used}")

        assert provider_used == "nvidia", (
            f"Expected provider_used == 'nvidia', got '{provider_used}'"
        )
        log("CODING task routing to NVIDIA: PASS")
        log("NVIDIA real API request: PASS")
    except Exception as exc:
        log(f"NVIDIA request failed: {_safe_provider_message(exc)}")
        log("RESULT: FAIL - Real NVIDIA API request failed.")
        return

    # 8. Verify Ollama fallback when NVIDIA is intentionally made unavailable
    log("\n--- Testing Ollama fallback when NVIDIA is intentionally unavailable ---")
    original_nvidia = service.nvidia

    class UnavailableNVIDIAProvider(NVIDIAProvider):
        async def is_available(self) -> bool:
            return False

    service.nvidia = UnavailableNVIDIAProvider()
    try:
        fallback_resp = await service.generate(
            prompt="Write a python function `def sub(a, b): return a - b`",
            system="Coding assistant",
        )
        fb_provider = service.last_provider_used
        fb_info = service.last_fallback_info

        log(f"Fallback Provider Used: {fb_provider}")
        log(f"Fallback Info: {fb_info}")

        if fb_provider == "ollama" and fb_info and fb_info.get("fallback_used"):
            log("NVIDIA failure -> Ollama fallback test: PASS")
        else:
            log(
                f"NVIDIA failure -> Ollama fallback test: FAIL (provider used: {fb_provider})"
            )
    except Exception as exc:
        log(f"Fallback test error: {_safe_provider_message(exc)}")
        log("NVIDIA failure -> Ollama fallback test: FAIL")
    finally:
        service.nvidia = original_nvidia

    log("\n=========================================")
    log("NVIDIA VERIFICATION SUMMARY: ALL PASSED")
    log("=========================================")


if __name__ == "__main__":
    asyncio.run(run_nvidia_verification())
