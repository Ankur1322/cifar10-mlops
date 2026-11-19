"""Data loading, stratified splitting, and tf.data pipelines for CIFAR-10."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
NUM_CLASSES = len(CLASS_NAMES)

Arrays = tuple[np.ndarray, np.ndarray]
Transform = Callable[[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]


LOCAL_DIR = Path.home() / ".keras" / "datasets" / "cifar-10-batches-py"


def _read_batch(path: Path) -> Arrays:
    """Read one CIFAR-10 pickle batch into (N, 32, 32, 3) uint8 images and int64 labels."""
    with open(path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, np.array(batch[b"labels"], dtype=np.int64)


def load_cifar10() -> tuple[Arrays, Arrays]:
    """Load CIFAR-10, preferring a local extracted copy, else downloading via Keras.

    Returns:
        ((x_train, y_train), (x_test, y_test)); images are uint8 (N, 32, 32, 3),
        labels are int64 vectors of shape (N,).
    """
    if (LOCAL_DIR / "test_batch").exists():
        parts = [_read_batch(LOCAL_DIR / f"data_batch_{i}") for i in range(1, 6)]
        x_train = np.concatenate([p[0] for p in parts])
        y_train = np.concatenate([p[1] for p in parts])
        return (x_train, y_train), _read_batch(LOCAL_DIR / "test_batch")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (
        (x_train, y_train.ravel().astype(np.int64)),
        (x_test, y_test.ravel().astype(np.int64)),
    )


def stratified_split_indices(
    labels: np.ndarray, val_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices into train and validation, keeping class proportions.

    Args:
        labels: Integer class labels.
        val_size: Number of validation samples.
        seed: Random seed, so the split is reproducible.

    Returns:
        Sorted (train_indices, val_indices) with no overlap.
    """
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, stratify=labels, random_state=seed
    )
    return np.sort(train_idx), np.sort(val_idx)


def make_splits(train: Arrays, test: Arrays, val_size: int, seed: int) -> dict[str, Arrays]:
    """Create train/val/test splits. The test set is passed through untouched."""
    x, y = train
    train_idx, val_idx = stratified_split_indices(y, val_size, seed)
    return {
        "train": (x[train_idx], y[train_idx]),
        "val": (x[val_idx], y[val_idx]),
        "test": test,
    }


def preprocess(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Scale pixels to [0, 1] float32 and one-hot encode the label."""
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    return image, label


def build_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    training: bool,
    seed: int = 42,
    shuffle_buffer: int = 10_000,
    augment_fn: Transform | None = None,
    batch_augment_fn: Transform | None = None,
    cache: bool = True,
) -> tf.data.Dataset:
    """Build an efficient tf.data pipeline.

    Args:
        images: uint8 images (N, 32, 32, 3).
        labels: Integer labels (N,).
        batch_size: Samples per batch.
        training: If True, shuffle and apply the augmentation hooks.
        seed: Shuffle seed.
        shuffle_buffer: Shuffle buffer size (capped at the dataset size).
        augment_fn: Optional per-image transform, training only.
        batch_augment_fn: Optional per-batch transform (MixUp/CutMix), training only.
        cache: Cache the preprocessed data in memory.

    Returns:
        A dataset yielding (images float32 in [0, 1], one-hot float32 labels).
    """
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(min(shuffle_buffer, len(images)), seed=seed, reshuffle_each_iteration=True)
        if augment_fn is not None:
            ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if training and batch_augment_fn is not None:
        ds = ds.map(batch_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def get_datasets(
    config: dict[str, Any],
    augment_fn: Transform | None = None,
    batch_augment_fn: Transform | None = None,
) -> dict[str, tf.data.Dataset]:
    """Load CIFAR-10 and return the train, val, and test datasets from a config."""
    seed = config["project"]["seed"]
    data_cfg = config["data"]
    train, test = load_cifar10()
    splits = make_splits(train, test, data_cfg["val_size"], seed)
    datasets = {}
    for name, (x, y) in splits.items():
        logger.info("%s: %d images", name, len(x))
        is_train = name == "train"
        datasets[name] = build_dataset(
            x,
            y,
            batch_size=data_cfg["batch_size"],
            training=is_train,
            seed=seed,
            shuffle_buffer=data_cfg["shuffle_buffer"],
            augment_fn=augment_fn,
            batch_augment_fn=batch_augment_fn,
        )
    return datasets
