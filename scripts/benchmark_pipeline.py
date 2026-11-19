"""Measure input-pipeline throughput (images/second) with and without augmentation."""

import logging
import time
from pathlib import Path

import tensorflow as tf

from cifar10.augment import get_augment_fns
from cifar10.data import build_dataset, load_cifar10
from cifar10.utils import load_config, set_seed, setup_logging

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("benchmark_pipeline")
NUM_IMAGES = 20_000


def measure(images, labels, batch_size: int, **kwargs) -> float:
    """Warm up for one epoch (fills the cache), then time one full epoch."""
    ds = build_dataset(images, labels, batch_size, training=True, **kwargs)
    for _ in ds:
        pass
    start = time.perf_counter()
    count = 0
    for batch_x, _ in ds:
        count += int(batch_x.shape[0])
    return count / (time.perf_counter() - start)


def main() -> None:
    config = load_config(ROOT / "configs" / "base.yaml")
    setup_logging(config["logging"]["level"])
    set_seed(config["project"]["seed"])
    (x_train, y_train), _ = load_cifar10()
    images, labels = x_train[:NUM_IMAGES], y_train[:NUM_IMAGES]
    batch_size = config["data"]["batch_size"]
    per_image, per_batch = get_augment_fns(config)

    runs = {
        "no augmentation": {},
        "per-image augmentation": {"augment_fn": per_image},
        "per-image + MixUp/CutMix": {"augment_fn": per_image, "batch_augment_fn": per_batch},
    }
    results = {name: measure(images, labels, batch_size, **kw) for name, kw in runs.items()}

    lines = [
        "# Input pipeline throughput",
        "",
        f"Measured on this machine (CPU), {NUM_IMAGES} images, batch size {batch_size}, "
        f"TensorFlow {tf.__version__}.",
        "",
        "| Pipeline | Images/second |",
        "|---|---|",
    ]
    for name, speed in results.items():
        logger.info("%-28s %8.0f images/s", name, speed)
        lines.append(f"| {name} | {speed:,.0f} |")
    out = ROOT / config["paths"]["reports_dir"] / "pipeline_benchmark.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("saved %s", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
