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
    n_ctx: int = Field(default=4096, ge=512, description="Context window size")
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
    reasoning_max_tokens: int = Field(
        default=4000, ge=256, description="Max tokens for reasoning/analysis stage"
    )
    coder_max_tokens: int = Field(
        default=3000, ge=256, description="Max tokens for code generation stage"
    )

    # Chunk sizing (higher = more context per chunk, but more tokens used)
    chunk_size: int = Field(
        default=3000, ge=500, description="Target size for code chunks in characters"
    )
    prompt_context_limit: int = Field(
        default=4000, ge=1000, description="Max chars of code context in prompts"
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

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    @field_validator("model_path", "coder_model_path", "embed_model_path", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path | None) -> Path | None:
        """Convert string paths to Path objects and resolve them."""
        if v is None:
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

