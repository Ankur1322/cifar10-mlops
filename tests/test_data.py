"""Tests for splitting and the tf.data pipeline (synthetic data, no download)."""

import numpy as np
import tensorflow as tf

from cifar10.data import (
    build_dataset,
    make_splits,
    preprocess,
    stratified_split_indices,
)


def _fake_data(per_class: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(10), per_class).astype(np.int64)
    images = rng.integers(0, 256, size=(len(labels), 32, 32, 3), dtype=np.uint8)
    return images, labels


def test_split_sizes() -> None:
    images, labels = _fake_data()
    test = (images[:50], labels[:50])
    splits = make_splits((images, labels), test, val_size=100, seed=42)
    assert len(splits["train"][0]) == 900
    assert len(splits["val"][0]) == 100
    assert len(splits["test"][0]) == 50


def test_train_val_indices_are_disjoint_and_complete() -> None:
    _, labels = _fake_data()
    train_idx, val_idx = stratified_split_indices(labels, val_size=100, seed=42)
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx) | set(val_idx) == set(range(len(labels)))


def test_val_split_is_class_balanced() -> None:
    _, labels = _fake_data()
    _, val_idx = stratified_split_indices(labels, val_size=100, seed=42)
    counts = np.bincount(labels[val_idx], minlength=10)
    assert counts.tolist() == [10] * 10


def test_same_seed_same_split() -> None:
    _, labels = _fake_data()
    first = stratified_split_indices(labels, 100, seed=7)
    second = stratified_split_indices(labels, 100, seed=7)
    np.testing.assert_array_equal(first[1], second[1])


def test_different_seed_different_split() -> None:
    _, labels = _fake_data()
    first = stratified_split_indices(labels, 100, seed=1)
    second = stratified_split_indices(labels, 100, seed=2)
    assert not np.array_equal(first[1], second[1])


def test_preprocess_scales_and_one_hots() -> None:
    image = tf.constant(np.full((32, 32, 3), 255, dtype=np.uint8))
    out_image, out_label = preprocess(image, tf.constant(3, dtype=tf.int64))
    assert out_image.dtype == tf.float32
    assert float(tf.reduce_max(out_image)) == 1.0
    assert out_label.numpy().tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


def test_batch_shapes_dtypes_and_range() -> None:
    images, labels = _fake_data()
    ds = build_dataset(images, labels, batch_size=32, training=False)
    batch_x, batch_y = next(iter(ds))
    assert batch_x.shape == (32, 32, 32, 3)
    assert batch_x.dtype == tf.float32
    assert 0.0 <= float(tf.reduce_min(batch_x)) <= float(tf.reduce_max(batch_x)) <= 1.0
    assert batch_y.shape == (32, 10)
    np.testing.assert_allclose(batch_y.numpy().sum(axis=1), 1.0)


def test_eval_dataset_keeps_order() -> None:
    images, labels = _fake_data()
    ds = build_dataset(images, labels, batch_size=64, training=False)
    seen = np.concatenate([y.numpy().argmax(axis=1) for _, y in ds])
    np.testing.assert_array_equal(seen, labels)


def test_training_shuffle_is_seeded() -> None:
    images, labels = _fake_data()

    def first_labels(seed: int) -> np.ndarray:
        ds = build_dataset(images, labels, 32, training=True, seed=seed, cache=False)
        return next(iter(ds))[1].numpy().argmax(axis=1)

    np.testing.assert_array_equal(first_labels(5), first_labels(5))
    assert not np.array_equal(first_labels(5), first_labels(6))


def test_augmentation_hooks_are_applied() -> None:
    images, labels = _fake_data()
    ds = build_dataset(
        images,
        labels,
        32,
        training=True,
        cache=False,
        augment_fn=lambda x, y: (tf.zeros_like(x), y),
        batch_augment_fn=lambda x, y: (x + 1.0, y),
    )
    batch_x, _ = next(iter(ds))
    np.testing.assert_allclose(batch_x.numpy(), 1.0)
