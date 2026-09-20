"""
Unit tests for the local LLM generation module (src/generator.py).

All tests in this suite mock Ollama HTTP/API interactions to guarantee
isolated, offline, and deterministic execution without requiring Ollama to run.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch
import pytest
import requests

from src.generator import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    LLMGenerator,
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    build_rag_prompt,
    get_generator,
)


# 1. Empty prompt rejected
def test_empty_prompt_rejected():
    gen = LLMGenerator()
    with pytest.raises(ValueError, match="Prompt cannot be empty or whitespace-only"):
        gen.generate("")


# 2. Whitespace-only prompt rejected
def test_whitespace_only_prompt_rejected():
    gen = LLMGenerator()
    with pytest.raises(ValueError, match="Prompt cannot be empty or whitespace-only"):
        gen.generate("    \n\t   ")

    with pytest.raises(TypeError, match="Prompt must be a string"):
        gen.generate(None)  # type: ignore


# 3. Default configuration
def test_default_configuration():
    gen = LLMGenerator()
    assert gen.base_url == DEFAULT_OLLAMA_BASE_URL
    assert gen.model == DEFAULT_OLLAMA_MODEL
    assert gen.temperature == DEFAULT_TEMPERATURE
    assert gen.timeout == DEFAULT_TIMEOUT


# 4. Configuration loading from environment variables
def test_configuration_loading_from_env():
    custom_env = {
        "OLLAMA_BASE_URL": "http://192.168.1.100:11434/",
        "OLLAMA_MODEL": "qwen2.5:3b",
        "OLLAMA_TEMPERATURE": "0.2",
        "OLLAMA_TIMEOUT": "45.0",
    }
    with patch.dict("os.environ", custom_env, clear=False):
        gen = LLMGenerator()
        assert gen.base_url == "http://192.168.1.100:11434"  # Trailing slash stripped
        assert gen.model == "qwen2.5:3b"
        assert gen.temperature == 0.2
        assert gen.timeout == 45.0


# 5. Custom constructor configuration
def test_custom_constructor_configuration():
    gen = LLMGenerator(
        model="custom-qwen:7b",
        base_url="http://custom-host:8000/",
        temperature=0.05,
        timeout=15.0,
    )
    assert gen.model == "custom-qwen:7b"
    assert gen.base_url == "http://custom-host:8000"
    assert gen.temperature == 0.05
    assert gen.timeout == 15.0


# 6. Successful mocked generation & 7. Response text extraction
@patch("requests.post")
def test_successful_mocked_generation(mock_post: MagicMock):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "qwen2.5:1.5b",
        "response": "  Agent-as-a-Judge saves 97.72% of evaluation time.  \n",
        "done": True,
    }
    mock_post.return_value = mock_response

    gen = LLMGenerator()
    output = gen.generate("What are the evaluation time savings?")

    assert output == "Agent-as-a-Judge saves 97.72% of evaluation time."
    mock_post.assert_called_once()
    called_args, called_kwargs = mock_post.call_args
    assert called_args[0] == "http://localhost:11434/api/generate"
    assert called_kwargs["json"]["model"] == "qwen2.5:1.5b"
    assert called_kwargs["json"]["prompt"] == "What are the evaluation time savings?"
    assert called_kwargs["json"]["system"] == DEFAULT_SYSTEM_PROMPT
    assert called_kwargs["json"]["options"]["temperature"] == 0.0
    assert called_kwargs["timeout"] == 60.0


# 8. Connection failure handling
@patch("requests.post")
def test_connection_failure_handling(mock_post: MagicMock):
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    gen = LLMGenerator()
    with pytest.raises(OllamaConnectionError, match="Unable to connect to Ollama at http://localhost:11434"):
        gen.generate("Hello?")


# 9. Model-not-found handling (HTTP 404 & JSON error body)
@patch("requests.post")
def test_model_not_found_handling_http_404(mock_post: MagicMock):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = '{"error": "model \'missing-model\' not found"}'
    mock_post.return_value = mock_response

    gen = LLMGenerator(model="missing-model")
    with pytest.raises(OllamaModelNotFoundError, match="Configured Ollama model 'missing-model' is not available locally"):
        gen.generate("Test prompt")


@patch("requests.post")
def test_model_not_found_handling_json_error(mock_post: MagicMock):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = '{"error": "model \'nonexistent\' not found, try pulling it"}'
    mock_post.return_value = mock_response

    gen = LLMGenerator(model="nonexistent")
    with pytest.raises(OllamaModelNotFoundError, match="is not available locally"):
        gen.generate("Test prompt")


# 10. Malformed response handling
@patch("requests.post")
def test_malformed_response_handling_non_json(mock_post: MagicMock):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("No JSON object could be decoded")
    mock_response.text = "Internal Proxy Error"
    mock_post.return_value = mock_response

    gen = LLMGenerator()
    with pytest.raises(OllamaResponseError, match="malformed non-JSON response"):
        gen.generate("Test prompt")


@patch("requests.post")
def test_malformed_response_handling_missing_response_field(mock_post: MagicMock):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"done": True, "eval_count": 42}
    mock_post.return_value = mock_response

    gen = LLMGenerator()
    with pytest.raises(OllamaResponseError, match="missing required 'response' field"):
        gen.generate("Test prompt")


# 11. Timeout handling
@patch("requests.post")
def test_timeout_handling(mock_post: MagicMock):
    mock_post.side_effect = requests.exceptions.ReadTimeout("Read timed out")

    gen = LLMGenerator(timeout=10.0)
    with pytest.raises(OllamaTimeoutError, match="timed out after 10.0s"):
        gen.generate("Test prompt")


# 12. Generation parameter handling overrides
@patch("requests.post")
def test_generation_parameter_overrides(mock_post: MagicMock):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Custom answer", "done": True}
    mock_post.return_value = mock_response

    gen = LLMGenerator(temperature=0.0, timeout=60.0)
    gen.generate(
        "Tell me about DevAI",
        system_prompt="Custom system instructions.",
        temperature=0.7,
        timeout=12.5,
    )

    _, called_kwargs = mock_post.call_args
    assert called_kwargs["json"]["options"]["temperature"] == 0.7
    assert called_kwargs["json"]["system"] == "Custom system instructions."
    assert called_kwargs["timeout"] == 12.5


# 13. health_check behavior
@patch("requests.get")
def test_health_check_behavior(mock_get: MagicMock):
    gen = LLMGenerator()

    # Success case (HTTP 200)
    mock_resp_ok = MagicMock()
    mock_resp_ok.status_code = 200
    mock_get.return_value = mock_resp_ok
    assert gen.health_check() is True

    # Server returned 503
    mock_resp_err = MagicMock()
    mock_resp_err.status_code = 503
    mock_get.return_value = mock_resp_err
    assert gen.health_check() is False

    # Connection error
    mock_get.side_effect = requests.exceptions.ConnectionError("Offline")
    assert gen.health_check() is False


# 14. model availability behavior
@patch("requests.get")
def test_model_availability_behavior(mock_get: MagicMock):
    gen = LLMGenerator(model="qwen2.5:1.5b")

    # Available model
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {"name": "qwen2.5:1.5b"},
            {"name": "nomic-embed-text:latest"},
        ]
    }
    mock_get.return_value = mock_resp
    assert gen.model_available() is True
    assert gen.model_available("nomic-embed-text:latest") is True
    assert gen.model_available("llama3:8b") is False

    # Connection error during check
    mock_get.side_effect = requests.exceptions.ConnectionError("Unreachable")
    with pytest.raises(OllamaConnectionError):
        gen.model_available()


# Prompt contract: build_rag_prompt
def test_build_rag_prompt_structure():
    evidence = [
        {
            "chunk_id": "chunk_p06_001",
            "page_number": 6,
            "section": "2.3 Preliminary Benchmark",
            "text": "Table 1 Preliminary Statistics of AI Developers.",
        },
        {
            "chunk_id": "chunk_p06_002",
            "page_number": 6,
            "section": "2.3 Preliminary Benchmark",
            "text": "MetaGPT generates fewer saved code files.",
        },
    ]

    prompt = build_rag_prompt("What are the costs?", evidence)
    assert "USER QUESTION:\nWhat are the costs?" in prompt
    assert "RETRIEVED EVIDENCE:" in prompt
    assert "[1]\nChunk ID: chunk_p06_001\nPage: 6\nSection: 2.3 Preliminary Benchmark" in prompt
    assert "Table 1 Preliminary Statistics of AI Developers." in prompt
    assert "[2]\nChunk ID: chunk_p06_002" in prompt
    assert "ANSWER:" in prompt


def test_build_rag_prompt_validation():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        build_rag_prompt("", [])

    with pytest.raises(ValueError, match="Question cannot be empty"):
        build_rag_prompt("   ", [])


def test_factory_helper():
    gen = get_generator(model="qwen2.5:3b")
    assert isinstance(gen, LLMGenerator)
    assert gen.model == "qwen2.5:3b"
