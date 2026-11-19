"""Image and batch augmentation for the CIFAR-10 training pipeline."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import tensorflow as tf

Transform = Callable[[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]


def random_crop_with_padding(image: tf.Tensor, pad: int = 4) -> tf.Tensor:
    """Reflect-pad an image by `pad` pixels, then randomly crop back to its size."""
    padded = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    return tf.image.random_crop(padded, tf.shape(image))


def rotate(image: tf.Tensor, angle: tf.Tensor | float) -> tf.Tensor:
    """Rotate an (H, W, C) image by `angle` radians around its center (reflect fill)."""
    angle = tf.cast(angle, tf.float32)
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    cx, cy = (width - 1.0) / 2.0, (height - 1.0) / 2.0
    cos, sin = tf.cos(angle), tf.sin(angle)
    transform = tf.stack(
        [cos, -sin, cx - cx * cos + cy * sin, sin, cos, cy - cx * sin - cy * cos, 0.0, 0.0]
    )
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis],
        transforms=transform[tf.newaxis],
        output_shape=tf.shape(image)[:2],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    rotated.set_shape(image.shape)
    return rotated


def random_erasing(
    image: tf.Tensor,
    prob: float = 0.25,
    area_range: tuple[float, float] = (0.02, 0.25),
    aspect_range: tuple[float, float] = (0.3, 3.3),
) -> tf.Tensor:
    """With probability `prob`, replace a random rectangle with uniform noise."""
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    area = height * width * tf.random.uniform([], area_range[0], area_range[1])
    ratio = tf.exp(tf.random.uniform([], math.log(aspect_range[0]), math.log(aspect_range[1])))
    box_h = tf.minimum(tf.round(tf.sqrt(area * ratio)), height)
    box_w = tf.minimum(tf.round(tf.sqrt(area / ratio)), width)
    top = tf.floor(tf.random.uniform([]) * (height - box_h + 1.0))
    left = tf.floor(tf.random.uniform([]) * (width - box_w + 1.0))
    rows = tf.cast(tf.range(tf.shape(image)[0]), tf.float32)[:, tf.newaxis]
    cols = tf.cast(tf.range(tf.shape(image)[1]), tf.float32)[tf.newaxis, :]
    mask = (rows >= top) & (rows < top + box_h) & (cols >= left) & (cols < left + box_w)
    noise = tf.random.uniform(tf.shape(image))
    erased = tf.where(mask[..., tf.newaxis], noise, image)
    apply = tf.random.uniform([]) < prob
    return tf.where(apply, erased, image)


def augment_image(
    image: tf.Tensor,
    label: tf.Tensor,
    pad: int = 4,
    max_degrees: float = 15.0,
    erase_prob: float = 0.25,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Per-image augmentation: flip, crop, small rotation, random erasing."""
    shape = image.shape
    image = tf.image.random_flip_left_right(image)
    image = random_crop_with_padding(image, pad)
    angle = tf.random.uniform([], -max_degrees, max_degrees) * (math.pi / 180.0)
    image = rotate(image, angle)
    image = random_erasing(image, erase_prob)
    image = tf.clip_by_value(image, 0.0, 1.0)
    image.set_shape(shape)
    return image, label


def _sample_beta(alpha: float) -> tf.Tensor:
    """Sample from Beta(alpha, alpha) using two Gamma draws."""
    first = tf.random.gamma([], alpha)
    second = tf.random.gamma([], alpha)
    return first / (first + second)


def _shuffled(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Return the batch in a random order (the 'partner' images for mixing)."""
    order = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    return tf.gather(images, order), tf.gather(labels, order)


def mixup(images: tf.Tensor, labels: tf.Tensor, alpha: float = 0.2) -> tuple[tf.Tensor, tf.Tensor]:
    """MixUp: blend each image and label with a shuffled partner."""
    lam = _sample_beta(alpha)
    other_x, other_y = _shuffled(images, labels)
    return lam * images + (1.0 - lam) * other_x, lam * labels + (1.0 - lam) * other_y


def cutmix(images: tf.Tensor, labels: tf.Tensor, alpha: float = 1.0) -> tuple[tf.Tensor, tf.Tensor]:
    """CutMix: paste a random box from a shuffled partner; mix labels by pasted area."""
    height = tf.cast(tf.shape(images)[1], tf.float32)
    width = tf.cast(tf.shape(images)[2], tf.float32)
    lam = _sample_beta(alpha)
    cut = tf.sqrt(1.0 - lam)
    box_h, box_w = cut * height, cut * width
    center_y = tf.random.uniform([]) * height
    center_x = tf.random.uniform([]) * width
    y1 = tf.round(tf.clip_by_value(center_y - box_h / 2.0, 0.0, height))
    y2 = tf.round(tf.clip_by_value(center_y + box_h / 2.0, 0.0, height))
    x1 = tf.round(tf.clip_by_value(center_x - box_w / 2.0, 0.0, width))
    x2 = tf.round(tf.clip_by_value(center_x + box_w / 2.0, 0.0, width))
    rows = tf.cast(tf.range(tf.shape(images)[1]), tf.float32)[:, tf.newaxis]
    cols = tf.cast(tf.range(tf.shape(images)[2]), tf.float32)[tf.newaxis, :]
    mask = (rows >= y1) & (rows < y2) & (cols >= x1) & (cols < x2)
    other_x, other_y = _shuffled(images, labels)
    mixed = tf.where(mask[tf.newaxis, :, :, tf.newaxis], other_x, images)
    kept = 1.0 - (y2 - y1) * (x2 - x1) / (height * width)  # label weight of original
    return mixed, kept * labels + (1.0 - kept) * other_y


def mix_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mix_prob: float = 0.5,
) -> tuple[tf.Tensor, tf.Tensor]:
    """With probability `mix_prob` apply CutMix or MixUp (50/50); else pass through."""

    def pick() -> tuple[tf.Tensor, tf.Tensor]:
        return tf.cond(
            tf.random.uniform([]) < 0.5,
            lambda: cutmix(images, labels, cutmix_alpha),
            lambda: mixup(images, labels, mixup_alpha),
        )

    return tf.cond(tf.random.uniform([]) < mix_prob, pick, lambda: (images, labels))


def get_augment_fns(config: dict[str, Any]) -> tuple[Transform, Transform]:
    """Build (per_image_fn, per_batch_fn) from the `augment` section of the config."""
    cfg = config.get("augment", {})

    def per_image(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return augment_image(
            image,
            label,
            pad=cfg.get("pad", 4),
            max_degrees=cfg.get("max_rotation_degrees", 15.0),
            erase_prob=cfg.get("erase_prob", 0.25),
        )

    def per_batch(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return mix_batch(
            images,
            labels,
            mixup_alpha=cfg.get("mixup_alpha", 0.2),
            cutmix_alpha=cfg.get("cutmix_alpha", 1.0),
            mix_prob=cfg.get("batch_mix_prob", 0.5),
        )

    return per_image, per_batch
