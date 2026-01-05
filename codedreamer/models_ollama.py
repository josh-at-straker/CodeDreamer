"""
Multi-model management for cognitive architecture using Ollama.

Manages separate models for different cognitive functions via Ollama API:
- Reasoning (32B): Deliberate thought, planning, response generation
- Coder (7B): Code generation, tool execution, fact extraction
- Embedding: Semantic similarity and retrieval

This module provides the same interface as models.py but uses Ollama
instead of llama-cpp-python, making it ideal for Mac and Docker deployments.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto

import httpx

logger = logging.getLogger(__name__)

# Ollama API configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_REASONING_MODEL = os.environ.get("OLLAMA_REASONING_MODEL", "qwen2.5-coder:32b")
OLLAMA_CODER_MODEL = os.environ.get("OLLAMA_CODER_MODEL", "qwen2.5-coder:7b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


class ModelRole(Enum):
    """Cognitive role a model serves."""

    REASONING = auto()  # System 2: deliberate thought (32B)
    CODER = auto()  # System 1: fast execution (7B)
    EMBEDDING = auto()  # Semantic encoding


@dataclass
class GenerationParams:
    """Parameters for text generation."""

    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    stop: list[str] = field(default_factory=list)


class OllamaModel:
    """
    Wrapper around Ollama API for a specific model.
    
    Provides lazy connection and thread-safe generation.
    """

    def __init__(
        self,
        model_name: str,
        role: ModelRole,
        base_url: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.model_name = model_name
        self.role = role
        self.base_url = base_url or OLLAMA_HOST
        self.timeout = timeout
        self._client: httpx.Client | None = None
        self._available: bool | None = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is available (for compatibility)."""
        if self._available is None:
            self._check_availability()
        return self._available or False

    @property
    def client(self) -> httpx.Client:
        """Lazy-load HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _check_availability(self) -> None:
        """Check if the model is available in Ollama."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                base_name = self.model_name.split(":")[0]
                self._available = any(
                    self.model_name in name or base_name in name
                    for name in model_names
                )
                if self._available:
                    logger.info(f"Ollama model available: {self.model_name}")
                else:
                    logger.warning(
                        f"Ollama model not found: {self.model_name}. "
                        f"Available: {model_names[:5]}"
                    )
            else:
                self._available = False
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            self._available = False

    def load(self) -> None:
        """For compatibility - Ollama models don't need explicit loading."""
        self._check_availability()

    def unload(self) -> None:
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info(f"Ollama {self.role.name} client closed")

    def generate(self, prompt: str, params: GenerationParams) -> str:
        """Generate text completion via Ollama API."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "repeat_penalty": params.repeat_penalty,
                "num_predict": params.max_tokens,
            },
        }

        if params.stop:
            payload["options"]["stop"] = params.stop

        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            text = data.get("response", "")
            tokens = data.get("eval_count", 0)
            logger.debug(f"Ollama {self.role.name}: generated {tokens} tokens")
            
            return text

        except httpx.HTTPError as e:
            logger.error(f"Ollama API error ({self.model_name}): {e}")
            raise RuntimeError(f"Ollama generation failed: {e}") from e

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector."""
        try:
            response = self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text,
                },
            )
            response.raise_for_status()
            return response.json().get("embedding", [])

        except httpx.HTTPError as e:
            logger.error(f"Ollama embedding error: {e}")
            raise RuntimeError(f"Ollama embedding failed: {e}") from e


class ModelOrchestra:
    """
    Orchestrates multiple Ollama models for cognitive tasks.

    Provides unified interface for reasoning, coding, and embedding operations.
    Uses Ollama API for all model interactions.
    """

    def __init__(
        self,
        reasoning_model: str | None = None,
        coder_model: str | None = None,
        embed_model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize the orchestra with Ollama model names.

        Args:
            reasoning_model: Name of 32B reasoning model in Ollama.
            coder_model: Name of 7B coder model in Ollama.
            embed_model: Name of embedding model in Ollama.
            base_url: Ollama API base URL.
        """
        base_url = base_url or OLLAMA_HOST

        # Reasoning model
        self._reasoning = OllamaModel(
            model_name=reasoning_model or OLLAMA_REASONING_MODEL,
            role=ModelRole.REASONING,
            base_url=base_url,
        )

        # Coder model (can be same as reasoning if not specified differently)
        coder_name = coder_model or OLLAMA_CODER_MODEL
        if coder_name != self._reasoning.model_name:
            self._coder = OllamaModel(
                model_name=coder_name,
                role=ModelRole.CODER,
                base_url=base_url,
            )
        else:
            self._coder = None

        # Embedding model
        self._embed = OllamaModel(
            model_name=embed_model or OLLAMA_EMBED_MODEL,
            role=ModelRole.EMBEDDING,
            base_url=base_url,
        )

        logger.info(
            f"Orchestra initialized (Ollama): "
            f"reasoning={self._reasoning.model_name}, "
            f"coder={self._coder.model_name if self._coder else 'shared'}, "
            f"embed={self._embed.model_name}"
        )

    @property
    def reasoning(self) -> OllamaModel:
        """Get the reasoning model."""
        return self._reasoning

    @property
    def coder(self) -> OllamaModel:
        """Get the coder model (or reasoning if not configured)."""
        return self._coder if self._coder else self._reasoning

    @property
    def embed(self) -> OllamaModel:
        """Get the embedding model."""
        return self._embed

    def reason(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate deliberate reasoning response.

        Uses the 32B model for complex thought, planning, synthesis.
        """
        params = GenerationParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        return self._reasoning.generate(prompt, params)

    def code(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate code or execute tool-like operations.

        Uses the 7B coder model for fast, focused generation.
        Lower temperature for more deterministic output.
        """
        params = GenerationParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            repeat_penalty=1.15,
        )
        return self.coder.generate(prompt, params)

    def dream(
        self,
        prompt: str,
        temperature: float = 0.9,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate creative, exploratory content.

        Uses reasoning model at high temperature for novelty.
        """
        params = GenerationParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            repeat_penalty=1.15,
        )
        return self._reasoning.generate(prompt, params)

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for semantic operations."""
        return self._embed.embed(text)

    def preload(self, roles: list[ModelRole] | None = None) -> None:
        """Check model availability (Ollama doesn't need preloading)."""
        self._reasoning.load()
        if self._coder:
            self._coder.load()
        self._embed.load()

    def unload_all(self) -> None:
        """Close all HTTP clients."""
        self._reasoning.unload()
        if self._coder:
            self._coder.unload()
        self._embed.unload()

    def status(self) -> dict[str, bool]:
        """Get availability status of all models."""
        return {
            "reasoning": self._reasoning.is_loaded,
            "coder": self._coder.is_loaded if self._coder else self._reasoning.is_loaded,
            "embed": self._embed.is_loaded,
        }


# Singleton instance
_orchestra: ModelOrchestra | None = None


def get_orchestra() -> ModelOrchestra:
    """Get or create the model orchestra singleton."""
    global _orchestra
    if _orchestra is None:
        _orchestra = ModelOrchestra()
    return _orchestra


# Alias for compatibility with original models.py
ManagedModel = OllamaModel

