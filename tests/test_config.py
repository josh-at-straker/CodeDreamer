"""Tests for configuration module."""

import os
from pathlib import Path

import pytest

from codedreamer.config import Settings


def test_settings_defaults() -> None:
    """Verify default settings are reasonable."""
    s = Settings()

    assert s.n_gpu_layers == 99
    assert s.n_ctx == 4096
    assert s.dream_temperature == 0.9
    assert s.novelty_threshold == 0.4


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify settings can be loaded from environment."""
    monkeypatch.setenv("DREAMER_N_GPU_LAYERS", "50")
    monkeypatch.setenv("DREAMER_DREAM_TEMPERATURE", "0.5")

    s = Settings()

    assert s.n_gpu_layers == 50
    assert s.dream_temperature == 0.5


def test_settings_path_validation() -> None:
    """Verify path fields are properly converted."""
    s = Settings()

    assert isinstance(s.model_path, Path)
    assert isinstance(s.dreams_dir, Path)


def test_settings_bounds_validation() -> None:
    """Verify validation on bounded fields."""
    with pytest.raises(Exception):
        Settings(dream_temperature=3.0)  # Max is 2.0

    with pytest.raises(Exception):
        Settings(novelty_threshold=-0.5)  # Min is 0.0




