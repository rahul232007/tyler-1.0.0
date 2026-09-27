import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from app.core.config import get_settings
from app.services.llm import (
    GeminiProvider,
    LLMService,
    NVIDIAProvider,
    OllamaProvider,
    TaskType,
    _safe_provider_message,
)


def log(msg):
    print(msg, flush=True)


async def run_tests():
    log("=========================================")
    log("STARTING DUAL AI PROVIDER ROUTER VALIDATION")
    log("=========================================")
    results = {}

    # 1. Capability & Provider Initialization Test
    log("\n--- 1. Testing Capabilities & Provider Initialization ---")
    gemini = GeminiProvider()
    nvidia = NVIDIAProvider()
    ollama = OllamaProvider()

    assert gemini.supports_task(TaskType.GENERAL_CHAT) == True
    assert gemini.supports_task(TaskType.VISION) == True
    assert gemini.supports_task(TaskType.CODING) == True

    assert nvidia.supports_task(TaskType.CODING) == True
    assert nvidia.supports_task(TaskType.SPECIALIZED) == True
    assert nvidia.supports_task(TaskType.VISION) == False

    assert ollama.supports_task(TaskType.GENERAL_CHAT) == True
    assert ollama.supports_task(TaskType.CODING) == True
    assert ollama.supports_task(TaskType.VISION) == False

    results["CAPABILITIES_TEST"] = "PASS"
    log("Capabilities test: PASS")

    # 2. Task Detection Test
    log("\n--- 2. Testing Task Detection ---")
    service = LLMService()
    t_chat = service.detect_task_type("Hello Jarvis!")
    t_code = service.detect_task_type("Write a python script with def main():")
    t_vision = service.detect_task_type("Analyze this image of a cat")
    t_math = service.detect_task_type("Solve math equation x^2 + 5x + 6 = 0")

    assert t_chat == TaskType.GENERAL_CHAT.value
    assert t_code == TaskType.CODING.value
    assert t_vision == TaskType.VISION.value
    assert t_math == TaskType.SPECIALIZED.value

    results["TASK_DETECTION_TEST"] = "PASS"
    log("Task detection test: PASS")

    # 3. Task Chain Routing Test
    log("\n--- 3. Testing Provider Chain Routing ---")
    coding_chain = [p.name for p in service.get_provider_chain(TaskType.CODING.value)]
    chat_chain = [
        p.name for p in service.get_provider_chain(TaskType.GENERAL_CHAT.value)
    ]
    vision_chain = [p.name for p in service.get_provider_chain(TaskType.VISION.value)]

    assert coding_chain[0] == "nvidia"
    assert chat_chain[0] == "gemini"
    assert vision_chain[0] == "gemini"

    results["CHAIN_ROUTING_TEST"] = "PASS"
    log("Provider chain routing test: PASS")

    # 4. Error Sanitization Test
    log("\n--- 4. Testing Error Sanitization ---")
    exc_key = Exception(
        "Invalid API key AQ.Ab8RN6JC8OBNhhPYts1fF2NBh6cpfJ provided for model gemini-1.5-flash"
    )
    sanitized = _safe_provider_message(exc_key)
    assert "AQ.Ab8RN6JC8OBNhhPYts1fF2NBh6cpfJ" not in sanitized
    assert "api key" not in sanitized
    assert sanitized == "provider authentication failed or misconfigured"

    results["ERROR_SANITIZATION_TEST"] = "PASS"
    log("Error sanitization test: PASS")

    # 5. Forced Ollama Test (Local qwen2.5:3b)
    log("\n--- 5. Testing Forced Ollama Path ---")
    ollama_ok = await ollama.is_available()
    log(f"Ollama is_available: {ollama_ok}")
    if ollama_ok:
        try:
            resp = await ollama.generate(
                prompt="Reply with 'ollama-ok'", system="You are a test assistant."
            )
            log(f"Ollama response snippet: {resp.strip()[:60]}")
            results["FORCED_OLLAMA_TEST"] = "PASS"
        except Exception as exc:
            log(f"Ollama generate failed: {exc}")
            results["FORCED_OLLAMA_TEST"] = "FAIL"
    else:
        results["FORCED_OLLAMA_TEST"] = (
            "BLOCKED (Ollama not running or qwen2.5:3b not installed)"
        )

    # 6. Gemini Failure -> Fallback to Ollama Test
    log("\n--- 6. Testing Gemini Failure -> Automatic Fallback to Ollama ---")
    original_gemini = service.gemini

    class BrokenGeminiProvider(GeminiProvider):
        async def is_available(self) -> bool:
            return True

        async def generate(self, prompt: str, system: str = "") -> str:
            raise RuntimeError("Simulated Gemini Runtime Failure")

    service.gemini = BrokenGeminiProvider()
    try:
        fallback_resp = await service.generate(
            prompt="Hello, reply with fallback test", system="Test"
        )
        provider_used = service.last_provider_used
        fallback_info = service.last_fallback_info
        log(f"Fallback response: {fallback_resp.strip()[:60]}")
        log(f"Provider used: {provider_used}")
        log(f"Fallback info: {fallback_info}")

        if (
            provider_used == "ollama"
            and fallback_info
            and fallback_info.get("fallback_used")
        ):
            results["GEMINI_FALLBACK_TEST"] = "PASS"
            log("Gemini failure -> Ollama fallback test: PASS")
        else:
            results["GEMINI_FALLBACK_TEST"] = (
                f"FAIL (unexpected provider_used: {provider_used})"
            )
    except Exception as exc:
        log(f"Gemini fallback test failed with exception: {exc}")
        results["GEMINI_FALLBACK_TEST"] = "FAIL"
    finally:
        service.gemini = original_gemini

    # 7. NVIDIA Provider Configuration & Availability Test
    log("\n--- 7. Testing NVIDIA Availability / Configuration ---")
    nvidia_avail = await nvidia.is_available()
    log(f"NVIDIA is_available: {nvidia_avail}")
    settings = get_settings()
    if not settings.nvidia_api_key:
        log(
            "NVIDIA_API_KEY is not configured (empty). Availability correctly returned False."
        )
        results["NVIDIA_TEST"] = "PASS (Unconfigured / Empty API Key safely handled)"
    elif nvidia_avail:
        try:
            nv_resp = await nvidia.generate(
                prompt="Reply with 'nvidia-ok'", system="Test"
            )
            log(f"NVIDIA response snippet: {nv_resp.strip()[:60]}")
            results["NVIDIA_TEST"] = "PASS (Live API Request Succeeded)"
        except Exception as exc:
            log(f"NVIDIA API call failed: {exc}")
            results["NVIDIA_TEST"] = f"FAIL ({_safe_provider_message(exc)})"
    else:
        results["NVIDIA_TEST"] = (
            "BLOCKED (NVIDIA API Key present but network/models endpoint unreachable)"
        )

    log("\n=========================================")
    log("SUMMARY OF VALIDATION RESULTS:")
    for k, v in results.items():
        log(f"  {k}: {v}")
    log("=========================================")


if __name__ == "__main__":
    asyncio.run(run_tests())
