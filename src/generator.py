"""
Local LLM generation module for the Enverus RAG pipeline.

Provides independent LLM generation capabilities using a local Ollama instance
and Qwen models, enforcing strict grounding and factual generation standards.
"""

import os
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
import requests

load_dotenv()

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 60.0

DEFAULT_SYSTEM_PROMPT = """You are a factual research assistant answering questions about the research paper "Agent-as-a-Judge: Evaluate Agents with Agents".

Follow these strict rules without exception:
1. Answer using ONLY the supplied evidence below. Do not use outside knowledge.
2. Do not invent or assume any facts, findings, or methodology not present in the evidence.
3. Do not invent numbers, percentages, costs, or runtimes.
4. Preserve all numerical values, equations, and metric names exactly as provided in the evidence.
5. If the supplied evidence is insufficient or does not contain the answer, explicitly state: "The provided evidence is insufficient to answer this question."
6. Do not pretend that a retrieved chunk contains information that it does not.
7. Keep answers concise, direct, and factual. Avoid filler and editorializing.
8. Cite the source page number and chunk ID when referencing specific facts or numbers.
9. Never fabricate citations, page numbers, or chunk identifiers.
10. If the question asks for multiple items (e.g., both SVM and LSTM), address only the components supported by the evidence and explicitly note any missing components."""


class OllamaError(Exception):
    """Base exception for Ollama generator errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when communication with the local Ollama server fails."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when the configured model is not installed or available locally."""
    pass


class OllamaTimeoutError(OllamaError):
    """Raised when an Ollama API request times out."""
    pass


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an unexpected or malformed response."""
    pass


def build_rag_prompt(question: str, evidence: List[Any]) -> str:
    """
    Contracts and formats user question and retrieved evidence chunks into a clean prompt.

    Args:
        question: The user query string.
        evidence: List of RetrievalResult objects or dictionaries containing
                  chunk_id, page_number, section, and text.

    Returns:
        Structured RAG prompt ready for LLM consumption.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty or whitespace-only.")

    evidence_blocks: List[str] = []
    for idx, item in enumerate(evidence, 1):
        if hasattr(item, "chunk_id"):
            cid = getattr(item, "chunk_id", "N/A")
            page = getattr(item, "page_number", "N/A")
            sec = getattr(item, "section", None) or "N/A"
            text = getattr(item, "text", "").strip()
        elif isinstance(item, dict):
            cid = item.get("chunk_id", "N/A")
            page = item.get("page_number", item.get("page", "N/A"))
            sec = item.get("section", "N/A") or "N/A"
            text = item.get("text", item.get("document", "")).strip()
        else:
            cid = f"evidence_{idx}"
            page = "N/A"
            sec = "N/A"
            text = str(item).strip()

        block = f"[{idx}]\nChunk ID: {cid}\nPage: {page}\nSection: {sec}\nText:\n{text}"
        evidence_blocks.append(block)

    formatted_evidence = "\n\n".join(evidence_blocks) if evidence_blocks else "No evidence provided."

    return (
        f"USER QUESTION:\n{question.strip()}\n\n"
        f"RETRIEVED EVIDENCE:\n\n{formatted_evidence}\n\n"
        "ANSWER:"
    )


class LLMGenerator:
    """
    Local LLM generation client interfacing with Ollama.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initializes the generator client.

        Args:
            model: Ollama model name (e.g., 'qwen2.5:1.5b').
                   Defaults to OLLAMA_MODEL env var or 'qwen2.5:1.5b'.
            base_url: Base URL of the Ollama server.
                      Defaults to OLLAMA_BASE_URL env var or 'http://localhost:11434'.
            temperature: Sampling temperature (0.0 for deterministic factual generation).
                         Defaults to OLLAMA_TEMPERATURE env var or 0.0.
            timeout: Request timeout in seconds.
                     Defaults to OLLAMA_TIMEOUT env var or 60.0.
        """
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        raw_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        self.base_url = raw_url.rstrip("/")

        if temperature is not None:
            self.temperature = float(temperature)
        else:
            self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", str(DEFAULT_TEMPERATURE)))

        if timeout is not None:
            self.timeout = float(timeout)
        else:
            self.timeout = float(os.getenv("OLLAMA_TIMEOUT", str(DEFAULT_TIMEOUT)))

    def health_check(self) -> bool:
        """
        Checks whether the Ollama server is reachable.

        Returns:
            True if the server responds successfully, False otherwise.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/version", timeout=min(5.0, self.timeout))
            return resp.status_code == 200
        except (requests.exceptions.RequestException, Exception):
            return False

    def get_available_models(self) -> List[str]:
        """
        Queries the Ollama server for currently installed local models.

        Returns:
            List of model names available locally.

        Raises:
            OllamaConnectionError: If connection to Ollama fails.
            OllamaTimeoutError: If request times out.
            OllamaResponseError: If response cannot be parsed.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=min(10.0, self.timeout))
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as err:
            raise OllamaTimeoutError(
                f"Request to Ollama at {self.base_url} timed out."
            ) from err
        except requests.exceptions.ConnectionError as err:
            raise OllamaConnectionError(
                f"Unable to connect to Ollama at {self.base_url}. Ensure Ollama is running."
            ) from err
        except requests.exceptions.RequestException as err:
            raise OllamaConnectionError(
                f"Failed to communicate with Ollama at {self.base_url}: {err}"
            ) from err

        if resp.status_code != 200:
            raise OllamaResponseError(
                f"Ollama returned HTTP status {resp.status_code} when listing models: {resp.text}"
            )

        try:
            data = resp.json()
            models_list = data.get("models", [])
            return [m.get("name", "") for m in models_list if isinstance(m, dict) and m.get("name")]
        except Exception as err:
            raise OllamaResponseError(
                f"Failed to parse model list from Ollama response: {err}"
            ) from err

    def model_available(self, model_name: Optional[str] = None) -> bool:
        """
        Verifies if the specified model (or configured self.model) is installed locally.

        Args:
            model_name: Optional model name to check. Defaults to self.model.

        Returns:
            True if the model is installed, False otherwise.
        """
        target = model_name or self.model
        available = self.get_available_models()

        # Check exact match or base match (e.g. 'qwen2.5:1.5b' vs 'qwen2.5:1.5b:latest')
        for m in available:
            if m == target or m.split(":")[0] == target.split(":")[0]:
                return True
        return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Generates text using the configured local Ollama model.

        Args:
            prompt: User prompt or completed RAG context string.
            system_prompt: Optional system prompt instructions. Defaults to DEFAULT_SYSTEM_PROMPT.
            temperature: Optional per-request temperature override.
            timeout: Optional per-request timeout override in seconds.

        Returns:
            Clean generated text response.

        Raises:
            TypeError: If prompt is not a string.
            ValueError: If prompt is empty or whitespace-only.
            OllamaConnectionError: If unable to reach the Ollama server.
            OllamaModelNotFoundError: If the requested model is not found on Ollama.
            OllamaTimeoutError: If the generation request exceeds the timeout.
            OllamaResponseError: If Ollama returns an unexpected or malformed response.
        """
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}.")
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Prompt cannot be empty or whitespace-only.")

        active_temp = temperature if temperature is not None else self.temperature
        active_timeout = timeout if timeout is not None else self.timeout
        active_system = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": cleaned_prompt,
            "stream": False,
            "options": {
                "temperature": active_temp,
            },
        }

        if active_system and active_system.strip():
            payload["system"] = active_system.strip()

        endpoint = f"{self.base_url}/api/generate"

        try:
            resp = requests.post(
                endpoint,
                json=payload,
                timeout=active_timeout,
            )
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as err:
            raise OllamaTimeoutError(
                f"Generation request to Ollama at {self.base_url} timed out after {active_timeout}s."
            ) from err
        except requests.exceptions.ConnectionError as err:
            raise OllamaConnectionError(
                f"Unable to connect to Ollama at {self.base_url}. Ensure Ollama is running."
            ) from err
        except requests.exceptions.RequestException as err:
            raise OllamaConnectionError(
                f"Failed to communicate with Ollama server at {self.base_url}: {err}"
            ) from err

        # Handle HTTP 404 or specific error responses
        if resp.status_code == 404:
            raise OllamaModelNotFoundError(
                f"Configured Ollama model '{self.model}' is not available locally. "
                f"Run 'ollama pull {self.model}' to download it."
            )

        if resp.status_code != 200:
            # Check if error message indicates model missing
            error_body = resp.text
            if "not found" in error_body.lower() or "try pulling" in error_body.lower():
                raise OllamaModelNotFoundError(
                    f"Configured Ollama model '{self.model}' is not available locally. Details: {error_body}"
                )
            raise OllamaResponseError(
                f"Ollama returned HTTP error {resp.status_code}: {error_body}"
            )

        try:
            data = resp.json()
        except Exception as err:
            raise OllamaResponseError(
                f"Ollama returned malformed non-JSON response: {resp.text}"
            ) from err

        if not isinstance(data, dict):
            raise OllamaResponseError(
                f"Expected JSON object response from Ollama, got {type(data).__name__}."
            )

        # Ollama errors can occasionally be returned inside JSON payload
        if "error" in data:
            error_msg = str(data["error"])
            if "not found" in error_msg.lower():
                raise OllamaModelNotFoundError(
                    f"Configured Ollama model '{self.model}' is not available locally. Details: {error_msg}"
                )
            raise OllamaResponseError(f"Ollama error: {error_msg}")

        if "response" not in data:
            raise OllamaResponseError(
                f"Ollama JSON response missing required 'response' field: {data}"
            )

        return str(data["response"]).strip()


def get_generator(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: Optional[float] = None,
) -> LLMGenerator:
    """Factory helper to obtain an LLMGenerator instance."""
    return LLMGenerator(
        model=model,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
    )
