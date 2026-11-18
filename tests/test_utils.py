"""Tests for seeding and config loading."""

import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from cifar10.utils import load_config, set_seed

ROOT = Path(__file__).resolve().parents[1]


def test_same_seed_gives_identical_tensors() -> None:
    set_seed(123)
    first = tf.random.uniform((4, 4)).numpy()
    set_seed(123)
    second = tf.random.uniform((4, 4)).numpy()
    np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_tensors() -> None:
    set_seed(1)
    first = tf.random.uniform((4, 4)).numpy()
    set_seed(2)
    second = tf.random.uniform((4, 4)).numpy()
    assert not np.array_equal(first, second)


def test_seed_controls_numpy_and_python_random() -> None:
    set_seed(7)
    np_first, py_first = np.random.rand(3), random.random()
    set_seed(7)
    np_second, py_second = np.random.rand(3), random.random()
    np.testing.assert_array_equal(np_first, np_second)
    assert py_first == py_second


def test_base_config_loads() -> None:
    config = load_config(ROOT / "configs" / "base.yaml")
    assert config["project"]["seed"] == 42
    assert config["logging"]["level"] == "INFO"


def test_later_config_overrides_earlier(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    override.write_text("project:\n  seed: 99\n", encoding="utf-8")
    config = load_config(ROOT / "configs" / "base.yaml", override)
    assert config["project"]["seed"] == 99
    assert config["project"]["name"] == "cifar10-mlops"  # untouched keys survive
