"""
LLM wrapper for llama-cpp-python.

Provides a clean interface for text generation and embeddings.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from llama_cpp import Llama

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from LLM generation."""

    text: str
    tokens_used: int
    finish_reason: str


class LLMClient:
    """Wrapper around llama-cpp-python for inference."""

    def __init__(
        self,
        model_path: Path | None = None,
        n_gpu_layers: int | None = None,
        n_ctx: int | None = None,
        n_threads: int | None = None,
        embedding: bool = False,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            model_path: Path to GGUF model file. Defaults to settings.model_path.
            n_gpu_layers: GPU layers to offload. Defaults to settings.n_gpu_layers.
            n_ctx: Context window size. Defaults to settings.n_ctx.
            n_threads: CPU threads. Defaults to settings.n_threads.
            embedding: If True, initialize for embeddings only.
        """
        # Use explicit None checks to handle falsy values correctly (per dream_20251229_171425)
        self.model_path = model_path if model_path is not None else settings.model_path
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else settings.n_gpu_layers
        self.n_ctx = n_ctx if n_ctx is not None else settings.n_ctx
        self.n_threads = n_threads if n_threads is not None else settings.n_threads
        self.embedding = embedding

        self._model: Llama | None = None

    @property
    def model(self) -> Llama:
        """Lazy-load the model on first access."""
        if self._model is None:
            logger.info(f"Loading model: {self.model_path}")
            self._model = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                embedding=self.embedding,
                verbose=False,
            )
            logger.info("Model loaded successfully")
        return self._model

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
        Generate text completion.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate. Defaults to settings.dream_max_tokens.
            temperature: Sampling temperature (higher = more creative).
            top_p: Nucleus sampling threshold.
            repeat_penalty: Penalty for repeated tokens.
            stop: Stop sequences to end generation.

        Returns:
            GenerationResult with generated text and metadata.
        """
        max_tokens = max_tokens or settings.dream_max_tokens
        stop = stop or []

        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
        )

        choice = response["choices"][0]
        return GenerationResult(
            text=choice["text"],
            tokens_used=response["usage"]["completion_tokens"],
            finish_reason=choice["finish_reason"],
        )

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

        for chunk in self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        ):
            if "choices" in chunk and chunk["choices"]:
                text = chunk["choices"][0].get("text", "")
                if text:
                    yield text

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        if not self.embedding:
            raise RuntimeError("Model not initialized for embeddings. Set embedding=True.")

        response = self.model.embed(text)
        return list(response)

    def close(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Model unloaded")


# Convenience function to get a shared client instance
_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get or create the default LLM client."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client



