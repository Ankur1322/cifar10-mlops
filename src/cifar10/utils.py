"""Shared helpers: reproducibility, logging, and config loading."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and TensorFlow so runs are reproducible.

    Args:
        seed: The seed value applied to every random number generator.
    """
    import tensorflow as tf  # imported here so config-only code stays light

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_logging(level: str = "INFO") -> None:
    """Configure one consistent log format for the whole project.

    Args:
        level: Logging level name, for example "INFO" or "DEBUG".
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base` without modifying either."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Load one or more YAML files; later files override earlier ones.

    Args:
        *paths: YAML file paths, for example base.yaml then cnn.yaml.

    Returns:
        The merged configuration dictionary.
    """
    config: dict[str, Any] = {}
    for path in paths:
        with open(path, encoding="utf-8") as file:
            config = _deep_merge(config, yaml.safe_load(file) or {})
    return config
