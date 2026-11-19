"""Load the real CIFAR-10 data, build splits, and print sanity statistics."""

import logging
from pathlib import Path

import numpy as np

from cifar10.data import build_dataset, load_cifar10, make_splits, stratified_split_indices
from cifar10.utils import load_config, set_seed, setup_logging

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("check_data")


def main() -> None:
    config = load_config(ROOT / "configs" / "base.yaml")
    setup_logging(config["logging"]["level"])
    seed = config["project"]["seed"]
    set_seed(seed)

    train, test = load_cifar10()
    logger.info("raw train %s %s | test %s", train[0].shape, train[0].dtype, test[0].shape)

    val_size = config["data"]["val_size"]
    train_idx, val_idx = stratified_split_indices(train[1], val_size, seed)
    assert set(train_idx).isdisjoint(set(val_idx)), "train/val overlap!"
    logger.info("train/val overlap check passed")

    splits = make_splits(train, test, val_size, seed)
    for name, (x, y) in splits.items():
        logger.info("%-5s %s per-class counts %s", name, x.shape, np.bincount(y).tolist())

    x, y = splits["train"]
    ds = build_dataset(x, y, config["data"]["batch_size"], training=True, seed=seed)
    batch_x, batch_y = next(iter(ds))
    logger.info(
        "batch %s %s | pixel range %.2f-%.2f | labels %s",
        batch_x.shape,
        batch_x.dtype.name,
        float(batch_x.numpy().min()),
        float(batch_x.numpy().max()),
        batch_y.shape,
    )


if __name__ == "__main__":
    main()
