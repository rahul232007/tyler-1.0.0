"""Unit tests for intelligent LLM Router (task detection, routing chain, and fallback)."""

import pytest
from app.services.llm import LLMService, TaskType, get_llm_service


def test_task_detection():
    llm = LLMService()

    assert (
        llm.detect_task_type("Can you write a python function to sort a list?")
        == TaskType.CODING.value
    )
    assert llm.detect_task_type("def calculate_sum(a, b):") == TaskType.CODING.value
    assert (
        llm.detect_task_type("Can you describe what is in this image or photo?")
        == TaskType.VISION.value
    )
    assert (
        llm.detect_task_type("Solve this math equation and prove the theorem")
        == TaskType.SPECIALIZED.value
    )
    assert (
        llm.detect_task_type("Tell me a funny story about space travel")
        == TaskType.GENERAL_CHAT.value
    )


def test_provider_chain_auto_mode():
    llm = LLMService()

    coding_chain = llm.get_provider_chain(task_type=TaskType.CODING.value)
    assert len(coding_chain) == 3
    # In auto mode, priority is Ollama -> NVIDIA -> Gemini
    assert coding_chain[0].name == "ollama"

    general_chain = llm.get_provider_chain(task_type=TaskType.GENERAL_CHAT.value)
    assert len(general_chain) == 3
    assert general_chain[0].name == "ollama"


def test_provider_chain_override():
    llm = LLMService()

    override_chain = llm.get_provider_chain(provider_override="ollama")
    assert override_chain[0].name == "ollama"


@pytest.mark.asyncio
async def test_llm_status_check():
    llm = get_llm_service()
    status_dict = await llm.status()

    assert "gemini_available" in status_dict
    assert "nvidia_available" in status_dict
    assert "ollama_available" in status_dict
    assert "active_provider" in status_dict
