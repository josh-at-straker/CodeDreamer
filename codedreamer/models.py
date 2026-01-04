"""
Multi-model management for cognitive architecture.

Manages separate models for different cognitive functions:
- Reasoning (14B): Deliberate thought, planning, response generation
- Coder (7B): Code generation, tool execution, fact extraction
- Embedding: Semantic similarity and retrieval
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Protocol

from llama_cpp import Llama

from .config import settings

logger = logging.getLogger(__name__)


class ModelRole(Enum):
    """Cognitive role a model serves."""

    REASONING = auto()  # System 2: deliberate thought (14B)
    CODER = auto()  # System 1: fast execution (7B)
    EMBEDDING = auto()  # Semantic encoding


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    path: Path
    role: ModelRole
    n_ctx: int = 4096
    n_gpu_layers: int = 99
    n_threads: int = 4
    embedding: bool = False

    @property
    def name(self) -> str:
        """Human-readable name from path."""
        return self.path.stem if self.path else f"{self.role.name.lower()}_model"


@dataclass
class GenerationParams:
    """Parameters for text generation."""

    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    stop: list[str] = field(default_factory=list)


class ModelInterface(Protocol):
    """Protocol for model implementations (allows mocking)."""

    def generate(self, prompt: str, params: GenerationParams) -> str: ...
    def embed(self, text: str) -> list[float]: ...
    def unload(self) -> None: ...


class ManagedModel:
    """
    Wrapper around a llama.cpp model with lazy loading.

    Thread-safe with support for context manager pattern.
    (Improved per dream_20251228_123928_code_fix.md)
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model: Llama | None = None
        self._loaded = False
        self._lock = threading.Lock()  # Thread safety for load/unload

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @contextmanager
    def managed(self) -> Iterator["ManagedModel"]:
        """
        Context manager for automatic model lifecycle.

        Usage:
            with model.managed() as m:
                result = m.generate(prompt, params)
            # Model automatically unloaded
        """
        self.load()
        try:
            yield self
        finally:
            self.unload()

    def load(self) -> None:
        """Load model into memory (thread-safe)."""
        with self._lock:
            if self._loaded:
                return

            logger.info(f"Loading {self.config.role.name} model: {self.config.path}")

            self._model = Llama(
                model_path=str(self.config.path),
                n_ctx=self.config.n_ctx,
                n_gpu_layers=self.config.n_gpu_layers,
                n_threads=self.config.n_threads,
                embedding=self.config.embedding,
                verbose=False,
            )
            self._loaded = True
            logger.info(f"{self.config.role.name} model loaded")

    def unload(self) -> None:
        """Release model from memory (thread-safe)."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                self._loaded = False
                logger.info(f"{self.config.role.name} model unloaded")

    def generate(self, prompt: str, params: GenerationParams) -> str:
        """Generate text completion."""
        if not self._loaded:
            self.load()

        assert self._model is not None

        response = self._model(
            prompt,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            repeat_penalty=params.repeat_penalty,
            stop=params.stop or None,
        )

        return response["choices"][0]["text"]

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector."""
        if not self.config.embedding:
            raise RuntimeError(f"{self.config.role.name} model not configured for embeddings")

        if not self._loaded:
            self.load()

        assert self._model is not None
        return list(self._model.embed(text))


class ModelOrchestra:
    """
    Orchestrates multiple models for cognitive tasks.

    Manages model lifecycle and provides unified interface for
    reasoning, coding, and embedding operations.
    """

    def __init__(
        self,
        reasoning_path: Path | None = None,
        coder_path: Path | None = None,
        embed_path: Path | None = None,
    ) -> None:
        """
        Initialize the orchestra with model paths.

        Args:
            reasoning_path: Path to 14B reasoning model. Defaults to settings.
            coder_path: Path to 7B coder model. If None, uses reasoning model.
            embed_path: Path to embedding model. If None, uses reasoning model.
        """
        # Reasoning model (required)
        reasoning_config = ModelConfig(
            path=reasoning_path or settings.model_path,
            role=ModelRole.REASONING,
            n_ctx=settings.n_ctx,
            n_gpu_layers=settings.n_gpu_layers,
            n_threads=settings.n_threads,
        )
        self._reasoning = ManagedModel(reasoning_config)

        # Coder model (optional, falls back to reasoning)
        coder_path = coder_path or settings.coder_model_path
        if coder_path and coder_path.exists():
            coder_config = ModelConfig(
                path=coder_path,
                role=ModelRole.CODER,
                n_ctx=4096,  # Coder typically needs less context
                n_gpu_layers=settings.n_gpu_layers,
                n_threads=settings.n_threads,
            )
            self._coder = ManagedModel(coder_config)
        else:
            self._coder = None  # Will use reasoning model

        # Embedding model (optional, falls back to reasoning)
        embed_path = embed_path or settings.embed_model_path
        if embed_path and embed_path.exists():
            embed_config = ModelConfig(
                path=embed_path,
                role=ModelRole.EMBEDDING,
                n_ctx=512,  # Embeddings need minimal context
                n_gpu_layers=0,  # CPU for embeddings to save VRAM
                n_threads=settings.n_threads,
                embedding=True,
            )
            self._embed = ManagedModel(embed_config)
        else:
            self._embed = None

        logger.info(
            f"Orchestra initialized: reasoning={reasoning_config.name}, "
            f"coder={'dedicated' if self._coder else 'shared'}, "
            f"embed={'dedicated' if self._embed else 'shared'}"
        )

    @property
    def reasoning(self) -> ManagedModel:
        """Get the reasoning model."""
        return self._reasoning

    @property
    def coder(self) -> ManagedModel:
        """Get the coder model (or reasoning if not configured)."""
        return self._coder if self._coder else self._reasoning

    @property
    def embed(self) -> ManagedModel:
        """Get the embedding model (or reasoning if not configured)."""
        return self._embed if self._embed else self._reasoning

    def reason(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate deliberate reasoning response.

        Uses the 14B model for complex thought, planning, synthesis.
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
            repeat_penalty=1.15,  # Prevent repetition loops
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
            repeat_penalty=1.15,  # Prevent repetition loops
        )
        return self._reasoning.generate(prompt, params)

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for semantic operations."""
        return self.embed.embed(text)

    def preload(self, roles: list[ModelRole] | None = None) -> None:
        """
        Preload specified models into memory.

        Args:
            roles: Which models to load. If None, loads reasoning only.
        """
        roles = roles or [ModelRole.REASONING]

        if ModelRole.REASONING in roles:
            self._reasoning.load()
        if ModelRole.CODER in roles and self._coder:
            self._coder.load()
        if ModelRole.EMBEDDING in roles and self._embed:
            self._embed.load()

    def unload_all(self) -> None:
        """Release all models from memory."""
        self._reasoning.unload()
        if self._coder:
            self._coder.unload()
        if self._embed:
            self._embed.unload()

    def status(self) -> dict[str, bool]:
        """Get load status of all models."""
        return {
            "reasoning": self._reasoning.is_loaded,
            "coder": self._coder.is_loaded if self._coder else False,
            "embed": self._embed.is_loaded if self._embed else False,
        }


# Singleton instance
_orchestra: ModelOrchestra | None = None


def get_orchestra() -> ModelOrchestra:
    """Get or create the model orchestra singleton."""
    global _orchestra
    if _orchestra is None:
        _orchestra = ModelOrchestra()
    return _orchestra

