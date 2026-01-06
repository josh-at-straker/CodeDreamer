"""
Configuration management using Pydantic Settings.

Environment variables are automatically loaded with DREAMER_ prefix.
Example: DREAMER_MODEL_PATH=/models/main.gguf
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="DREAMER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Model paths (multi-model architecture)
    model_path: Path = Field(
        default=Path("/models/qwen2.5-14b-instruct-q4_k_m.gguf"),
        description="Path to 14B reasoning model (GGUF format)",
    )
    coder_model_path: Path | None = Field(
        default=None,
        description="Path to 7B coder model. If None, uses reasoning model.",
    )
    embed_model_path: Path | None = Field(
        default=None,
        description="Path to embedding model. If None, uses reasoning model.",
    )

    # Model parameters
    n_gpu_layers: int = Field(default=99, ge=0, description="Layers to offload to GPU")
    n_ctx: int = Field(default=32768, ge=512, description="Context window size (Qwen2.5 supports 128K)")
    n_threads: int = Field(default=4, ge=1, description="CPU threads for inference")

    # Adaptive context sizing (Dream #2: dream_20251228_142330_code_fix.md)
    min_ctx: int = Field(default=2048, ge=512, description="Minimum context window")
    max_ctx: int = Field(default=16384, ge=1024, description="Maximum context window")
    adaptive_ctx_enabled: bool = Field(
        default=False, description="Auto-adjust n_ctx based on codebase complexity"
    )

    # Dream parameters
    dream_temperature: float = Field(
        default=0.9, ge=0.0, le=2.0, description="Temperature for dream generation"
    )
    dream_max_tokens: int = Field(default=1024, ge=64, description="Max tokens per dream (legacy)")
    dream_interval_sec: int = Field(default=300, ge=10, description="Seconds between dream cycles")
    decay_interval_sec: int = Field(default=3600, ge=60, description="Seconds between graph decay cycles")
    max_dreams_per_cycle: int = Field(default=5, ge=1, description="Max dreams per cycle")

    # Deep reasoning token limits (higher = deeper thinking)
    # Optimized for Qwen2.5-Coder which supports 128K context
    reasoning_max_tokens: int = Field(
        default=8000, ge=256, description="Max tokens for reasoning/analysis stage"
    )
    coder_max_tokens: int = Field(
        default=6000, ge=256, description="Max tokens for code generation stage"
    )

    # Chunk sizing (higher = more context per chunk, but more tokens used)
    chunk_size: int = Field(
        default=5000, ge=500, description="Target size for code chunks in characters"
    )
    prompt_context_limit: int = Field(
        default=12000, ge=1000, description="Max chars of code context in prompts"
    )
    
    # Extended context mode - for high-VRAM users (32GB+)
    # Enables: larger context, import file inclusion, codebase overview
    extended_context: bool = Field(
        default=False,
        description="Enable extended context mode: 2x context limit, import snippets, codebase overview",
    )
    extended_context_limit: int = Field(
        default=12000, ge=4000, description="Context limit when extended_context=True"
    )

    # Validation thresholds
    novelty_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Minimum novelty score to save dream"
    )
    lucid_check_enabled: bool = Field(
        default=True, description="Enable self-validation of dreams"
    )

    # Storage
    dreams_dir: Path = Field(default=Path("./dreams"), description="Directory for saved dreams")
    db_path: Path = Field(default=Path("./codedreamer.db"), description="ChromaDB storage path")
    graph_path: Path = Field(default=Path("./graph.json"), description="Knowledge graph storage")

    # Codebase (for daemon mode)
    codebase_path: Path | None = Field(
        default=None,
        description="Path to codebase to analyze. Required for daemon mode.",
    )

    # Indexing exclusions (comma-separated patterns)
    exclude_patterns: str = Field(
        default="tests,test_*,*_test.py,conftest.py,__init__.py",
        description="Comma-separated patterns to exclude from indexing (directories or file globs)",
    )
    clear_index_on_start: bool = Field(
        default=False,
        description="Clear and rebuild index on every startup (useful when changing exclude patterns)",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # Ollama settings (used when OLLAMA_HOST is set)
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama API endpoint",
    )
    ollama_reasoning_model: str = Field(
        default="qwen2.5-coder:32b",
        description="Ollama model for reasoning/analysis",
    )
    ollama_coder_model: str = Field(
        default="qwen2.5-coder:7b",
        description="Ollama model for code generation",
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text",
        description="Ollama model for embeddings",
    )

    @field_validator("model_path", "coder_model_path", "embed_model_path", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path | None) -> Path | None:
        """Convert string paths to Path objects and resolve them."""
        if v is None:
            return None
        # Handle empty strings or directory-only paths (no actual model file)
        if isinstance(v, str):
            v = v.strip()
            if not v or v.endswith("/") or not v.endswith(".gguf"):
                return None
        path = Path(v)
        if path.exists():
            return path.resolve()
        return path  # Return as-is for Docker paths that may not exist locally

    @field_validator("dreams_dir", "db_path", mode="before")
    @classmethod
    def ensure_parent_exists(cls, v: str | Path) -> Path:
        """Ensure parent directory exists for output paths."""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# Singleton instance - import this in other modules
settings = Settings()

