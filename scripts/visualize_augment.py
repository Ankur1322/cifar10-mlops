"""Save image grids showing per-image augmentation and MixUp/CutMix."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cifar10.augment import augment_image, cutmix, mixup
from cifar10.data import CLASS_NAMES, load_cifar10
from cifar10.utils import load_config, set_seed, setup_logging

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("visualize_augment")


def _label_text(label: np.ndarray) -> str:
    top = np.argsort(label)[::-1][:2]
    text = f"{CLASS_NAMES[top[0]]} {label[top[0]]:.2f}"
    if label[top[1]] > 0.01:
        text += f"\n{CLASS_NAMES[top[1]]} {label[top[1]]:.2f}"
    return text


def per_image_grid(images: np.ndarray, labels: np.ndarray, out: Path) -> None:
    rows = 5
    fig, axes = plt.subplots(rows, 10, figsize=(15, 8))
    for col in range(10):
        image = tf.constant(images[col], dtype=tf.float32) / 255.0
        onehot = tf.one_hot(labels[col], 10)
        axes[0, col].imshow(image.numpy())
        axes[0, col].set_title(CLASS_NAMES[labels[col]], fontsize=8)
        for row in range(1, rows):
            augmented, _ = augment_image(image, onehot)
            axes[row, col].imshow(augmented.numpy())
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Row 1: original | Rows 2-5: random augmentations", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def mix_grid(images: np.ndarray, labels: np.ndarray, out: Path) -> None:
    x = tf.constant(images[:8], dtype=tf.float32) / 255.0
    y = tf.one_hot(labels[:8], 10)
    mixed = {"original": (x, y), "MixUp": mixup(x, y, 0.4), "CutMix": cutmix(x, y, 1.0)}
    fig, axes = plt.subplots(3, 8, figsize=(16, 7))
    for row, (name, (mx, my)) in enumerate(mixed.items()):
        for col in range(8):
            axes[row, col].imshow(np.clip(mx[col].numpy(), 0, 1))
            axes[row, col].set_title(_label_text(my[col].numpy()), fontsize=7)
            axes[row, col].axis("off")
        axes[row, 0].text(
            -0.1,
            0.5,
            name,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    config = load_config(ROOT / "configs" / "base.yaml")
    setup_logging(config["logging"]["level"])
    set_seed(config["project"]["seed"])
    (x_train, y_train), _ = load_cifar10()
    reports = ROOT / config["paths"]["reports_dir"]
    reports.mkdir(exist_ok=True)

    first_of_each = np.array([np.where(y_train == c)[0][0] for c in range(10)])
    per_image_grid(
        x_train[first_of_each], y_train[first_of_each], reports / "augmentation_grid.png"
    )
    mix_grid(x_train[:8], y_train[:8], reports / "mix_examples.png")
    logger.info("saved reports/augmentation_grid.png and reports/mix_examples.png")


if __name__ == "__main__":
    main()
