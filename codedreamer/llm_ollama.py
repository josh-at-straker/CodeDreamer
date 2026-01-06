"""
Ollama LLM client for CodeDreamer.

Provides a compatible interface to the existing LLMClient but uses
Ollama's HTTP API instead of llama-cpp-python. This enables:
- No local model loading (uses existing Ollama installation)
- Easy model switching via environment variables
- Docker compatibility via host.docker.internal
"""

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from .config import settings

logger = logging.getLogger(__name__)

# Ollama API endpoint - use host.docker.internal when in Docker
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


@dataclass
class GenerationResult:
    """Result from LLM generation."""

    text: str
    tokens_used: int
    finish_reason: str


class OllamaClient:
    """
    Ollama-based LLM client compatible with CodeDreamer's LLMClient interface.
    
    Uses Ollama's HTTP API for inference, making it ideal for:
    - Mac systems without CUDA
    - Docker deployments
    - Reusing existing Ollama model installations
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        """
        Initialize the Ollama client.

        Args:
            model: Ollama model name (e.g., "qwen2.5-coder:32b").
                   Defaults to OLLAMA_MODEL env var or "qwen2.5-coder:7b".
            base_url: Ollama API base URL. 
                      Defaults to OLLAMA_HOST env var or "http://localhost:11434".
            timeout: Request timeout in seconds.
        """
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
        self.base_url = base_url or OLLAMA_HOST
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        
        logger.info(f"OllamaClient initialized: model={self.model}, url={self.base_url}")

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """
        Generate text completion via Ollama API.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate (Ollama uses num_predict).
            temperature: Sampling temperature (higher = more creative).
            top_p: Nucleus sampling threshold.
            repeat_penalty: Penalty for repeated tokens.
            stop: Stop sequences to end generation.

        Returns:
            GenerationResult with generated text and metadata.
        """
        max_tokens = max_tokens or settings.dream_max_tokens
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "num_predict": max_tokens,
            },
        }
        
        if stop:
            payload["options"]["stop"] = stop

        try:
            response = self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            return GenerationResult(
                text=data.get("response", ""),
                tokens_used=data.get("eval_count", 0),
                finish_reason=data.get("done_reason", "stop"),
            )
            
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise RuntimeError(f"Ollama generation failed: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        **kwargs: float,
    ) -> Iterator[str]:
        """
        Stream text generation token by token.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional generation parameters.

        Yields:
            Generated text chunks.
        """
        max_tokens = max_tokens or settings.dream_max_tokens
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs,
            },
        }

        try:
            with self._client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                            
        except httpx.HTTPError as e:
            logger.error(f"Ollama streaming error: {e}")
            raise RuntimeError(f"Ollama streaming failed: {e}") from e

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding vector for text using Ollama embeddings API.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        
        try:
            response = self._client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": embed_model,
                    "prompt": text,
                },
            )
            response.raise_for_status()
            data = response.json()
            
            return data.get("embedding", [])
            
        except httpx.HTTPError as e:
            logger.error(f"Ollama embedding error: {e}")
            raise RuntimeError(f"Ollama embedding failed: {e}") from e

    def close(self) -> None:
        """Release client resources."""
        self._client.close()
        logger.info("OllamaClient closed")

    def is_available(self) -> bool:
        """Check if Ollama is reachable and model is available."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Check if our model (or base name) is available
                base_model = self.model.split(":")[0]
                return any(base_model in name for name in model_names)
            return False
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            return False


# Convenience function to get a shared client instance
_default_client: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    """Get or create the default Ollama client."""
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client


# Alias for compatibility with existing code
LLMClient = OllamaClient
get_llm_client = get_ollama_client

