"""Shared pytest fixtures for dyson_cnn test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the DysonianLineCNN repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def config_dir(repo_root: Path) -> Path:
    """Absolute path to the config/ directory."""
    return repo_root / "config"
