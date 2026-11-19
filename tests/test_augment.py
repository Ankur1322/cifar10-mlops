"""Tests for per-image and batch augmentation (synthetic data)."""

import numpy as np
import tensorflow as tf

from cifar10.augment import (
    augment_image,
    cutmix,
    mix_batch,
    mixup,
    random_crop_with_padding,
    random_erasing,
    rotate,
)
from cifar10.data import build_dataset


def _image(seed: int = 0) -> tf.Tensor:
    return tf.random.stateless_uniform((32, 32, 3), seed=[seed, 1])


def _batch(n: int = 8) -> tuple[tf.Tensor, tf.Tensor]:
    images = tf.random.stateless_uniform((n, 32, 32, 3), seed=[3, 4])
    labels = tf.one_hot(tf.range(n) % 10, 10)
    return images, labels


def test_augment_image_preserves_shape_dtype_and_range() -> None:
    out, _ = augment_image(_image(), tf.one_hot(2, 10))
    assert out.shape == (32, 32, 3)
    assert out.dtype == tf.float32
    assert 0.0 <= float(tf.reduce_min(out)) and float(tf.reduce_max(out)) <= 1.0


def test_augment_image_keeps_label() -> None:
    label = tf.one_hot(5, 10)
    _, out_label = augment_image(_image(), label)
    np.testing.assert_array_equal(out_label.numpy(), label.numpy())


def test_crop_keeps_shape() -> None:
    assert random_crop_with_padding(_image(), pad=4).shape == (32, 32, 3)


def test_rotation_zero_is_identity() -> None:
    image = _image()
    np.testing.assert_allclose(rotate(image, 0.0).numpy(), image.numpy(), atol=1e-5)


def test_rotation_changes_image() -> None:
    image = _image()
    assert not np.allclose(rotate(image, 0.5).numpy(), image.numpy(), atol=1e-3)


def test_random_erasing_prob_zero_is_identity() -> None:
    image = _image()
    np.testing.assert_array_equal(random_erasing(image, prob=0.0).numpy(), image.numpy())


def test_random_erasing_prob_one_changes_image() -> None:
    image = tf.zeros((32, 32, 3))
    assert float(tf.reduce_max(random_erasing(image, prob=1.0))) > 0.0


def test_mixup_labels_sum_to_one_and_range_kept() -> None:
    images, labels = _batch()
    mixed_x, mixed_y = mixup(images, labels, alpha=0.4)
    np.testing.assert_allclose(mixed_y.numpy().sum(axis=1), 1.0, atol=1e-5)
    assert mixed_x.shape == images.shape
    assert 0.0 <= float(tf.reduce_min(mixed_x)) and float(tf.reduce_max(mixed_x)) <= 1.0


def test_cutmix_labels_sum_to_one_and_range_kept() -> None:
    images = tf.concat([tf.zeros((4, 32, 32, 3)), tf.ones((4, 32, 32, 3))], axis=0)
    labels = tf.one_hot(tf.range(8) % 10, 10)
    mixed_x, mixed_y = cutmix(images, labels)
    np.testing.assert_allclose(mixed_y.numpy().sum(axis=1), 1.0, atol=1e-5)
    assert mixed_x.shape == images.shape
    assert 0.0 <= float(tf.reduce_min(mixed_x)) and float(tf.reduce_max(mixed_x)) <= 1.0


def test_mix_batch_handles_small_final_batch() -> None:
    images, labels = _batch(3)
    mixed_x, mixed_y = mix_batch(images, labels, mix_prob=1.0)
    assert mixed_x.shape == (3, 32, 32, 3)
    np.testing.assert_allclose(mixed_y.numpy().sum(axis=1), 1.0, atol=1e-5)


def test_augmentations_are_reproducible_with_seed() -> None:
    images, labels = _batch()
    tf.random.set_seed(3)
    first_img, _ = augment_image(_image(), tf.one_hot(1, 10))
    first_mix, _ = mix_batch(images, labels, mix_prob=1.0)
    tf.random.set_seed(3)
    second_img, _ = augment_image(_image(), tf.one_hot(1, 10))
    second_mix, _ = mix_batch(images, labels, mix_prob=1.0)
    np.testing.assert_allclose(first_img.numpy(), second_img.numpy())
    np.testing.assert_allclose(first_mix.numpy(), second_mix.numpy())


def test_pipeline_runs_with_both_augmentation_hooks() -> None:
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(200, 32, 32, 3), dtype=np.uint8)
    labels = np.repeat(np.arange(10), 20).astype(np.int64)
    ds = build_dataset(
        images,
        labels,
        batch_size=32,
        training=True,
        cache=False,
        augment_fn=augment_image,
        batch_augment_fn=mix_batch,
    )
    batch_x, batch_y = next(iter(ds))
    assert batch_x.shape == (32, 32, 32, 3)
    assert batch_y.shape == (32, 10)
    np.testing.assert_allclose(batch_y.numpy().sum(axis=1), 1.0, atol=1e-5)
