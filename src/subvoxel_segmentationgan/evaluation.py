"""Checkpoint inference and aggregate segmentation evaluation."""

from __future__ import annotations

import csv
import json

import numpy as np

from .config import RunConfig
from .data import create_dataset, paired_volume_paths, split_directory
from .outputs import class_confusion, metric_rows, write_prediction


def evaluate(config: RunConfig) -> list[dict[str, float]]:
    """Run inference on ``Test`` and write predictions plus aggregate metrics."""

    import tensorflow as tf

    from .models import Generator

    config.validate()
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    test_dir = split_directory(config.data_dir, "Test")
    paired_volume_paths(test_dir)
    dataset = create_dataset(test_dir, config, training=False)
    latest = tf.train.latest_checkpoint(str(config.resolved_checkpoint_dir))
    if latest is None:
        raise FileNotFoundError(f"no TensorFlow checkpoint found in {config.resolved_checkpoint_dir}")

    generator = Generator(config.input_channels, config.output_channels, config.patch_size)
    checkpoint = tf.train.Checkpoint(generator=generator)
    restore_status = checkpoint.restore(latest)
    restore_status.expect_partial()
    restore_status.assert_existing_objects_matched()
    prediction_dir = config.resolved_prediction_dir
    prediction_dir.mkdir(parents=True, exist_ok=True)
    totals = [np.zeros(config.output_channels, dtype=np.int64) for _ in range(4)]

    sample_id = 0
    for low_res, target in dataset:
        low_res_cf = tf.transpose(low_res, [0, 4, 1, 2, 3])
        target_cf = tf.transpose(target, [0, 4, 1, 2, 3])
        # Preserve the released evaluator's dropout-active prediction behavior.
        probabilities = generator(low_res_cf, training=True)
        predictions = tf.argmax(probabilities, axis=1).numpy() + 1
        expected_batch = tf.argmax(target_cf, axis=1).numpy() + 1
        inputs = low_res_cf.numpy()
        for input_volume, expected, prediction in zip(inputs, expected_batch, predictions):
            confusion = class_confusion(expected, prediction, config.output_channels)
            for aggregate, sample in zip(totals, confusion):
                aggregate += sample
            write_prediction(
                prediction_dir,
                sample_id,
                input_volume,
                expected,
                prediction,
                validation=True,
            )
            sample_id += 1

    rows = metric_rows(*totals)
    with (prediction_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
    with (prediction_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows
