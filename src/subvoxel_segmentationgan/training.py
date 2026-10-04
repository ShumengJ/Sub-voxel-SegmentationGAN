"""Training workflow for the released 3D conditional GAN."""

from __future__ import annotations

import csv
import datetime
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .config import RunConfig, config_dict
from .data import create_dataset, paired_volume_paths, split_directory


def tensorboard_log_dir(config: RunConfig, timestamp: Optional[str] = None) -> Path:
    """Return the original ``logs/fit/<timestamp>`` location."""

    timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return config.resolved_log_dir / timestamp


def train(config: RunConfig) -> None:
    """Train from ``Train`` volumes and checkpoint after each epoch."""

    import tensorflow as tf

    from .losses import discriminator_loss, generator_loss, get_class2_metrics
    from .models import Discriminator, Generator

    config.validate()
    train_dir = split_directory(config.data_dir, "Train")
    train_paths = paired_volume_paths(train_dir)
    train_dataset = create_dataset(train_dir, config, training=True)
    steps_per_epoch = int(np.ceil(len(train_paths) / config.batch_size))

    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamped_log_dir = tensorboard_log_dir(config)
    with (config.output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config_dict(config), stream, indent=2)

    generator = Generator(config.input_channels, config.output_channels, config.patch_size)
    discriminator = Discriminator(config.input_channels, config.output_channels, config.patch_size)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    checkpoint_prefix = str(config.resolved_checkpoint_dir / "ckpt")
    summary_writer = tf.summary.create_file_writer(str(timestamped_log_dir))
    log_path = config.epoch_log_path
    # ``generator_mse`` is the paper's edge-attentive residual MSE; retain the
    # concise column name for compatibility with existing analysis scripts.
    fields = [
        "epoch", "generator_total", "generator_gan", "generator_mse",
        "generator_bce", "discriminator", "dice_background", "dice_eggshell",
        "dice_pore", "pore_tp", "pore_tn", "pore_fp", "pore_fn",
        "pore_precision", "pore_recall", "pore_specificity",
    ]

    @tf.function
    def train_step(input_image, target):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generated = generator(input_image, training=True)
            real_score = discriminator([input_image, target], training=True)
            generated_score = discriminator([input_image, generated], training=True)
            generator_values = generator_loss(
                generated_score,
                generated,
                target,
                input_image,
                config.lambda_mse,
                config.lambda_bce,
            )
            discriminator_value = discriminator_loss(real_score, generated_score)
        generator_gradients = generator_tape.gradient(
            generator_values[0], generator.trainable_variables
        )
        discriminator_gradients = discriminator_tape.gradient(
            discriminator_value, discriminator.trainable_variables
        )
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )
        confusion = get_class2_metrics(target, generated)
        return (*generator_values, discriminator_value, *confusion)

    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            started = time.time()
            sums = np.zeros(5, dtype=np.float64)
            dice = np.zeros(config.output_channels, dtype=np.float64)
            confusion = np.zeros(4, dtype=np.int64)
            batches = 0
            for low_res, target in train_dataset:
                low_res = tf.transpose(low_res, [0, 4, 1, 2, 3])
                target = tf.transpose(target, [0, 4, 1, 2, 3])
                total, gan, mse, bce, class_dice, disc, tp, tn, fp, fn = train_step(low_res, target)
                sums += [total.numpy(), gan.numpy(), mse.numpy(), bce.numpy(), disc.numpy()]
                dice += class_dice.numpy()
                confusion += [tp.numpy(), tn.numpy(), fp.numpy(), fn.numpy()]
                batches += 1
            if batches != steps_per_epoch:
                raise RuntimeError(f"expected {steps_per_epoch} batches, received {batches}")
            averages = sums / batches
            dice /= batches
            tp, tn, fp, fn = confusion
            epsilon = 1e-7
            row = dict(
                zip(
                    fields,
                    [
                        epoch, *averages, *dice, tp, tn, fp, fn,
                        tp / (tp + fp + epsilon),
                        tp / (tp + fn + epsilon),
                        tn / (tn + fp + epsilon),
                    ],
                )
            )
            writer.writerow(row)
            stream.flush()
            with summary_writer.as_default():
                for name, value in row.items():
                    if name != "epoch":
                        tf.summary.scalar(name, value, step=epoch)
            checkpoint.save(file_prefix=checkpoint_prefix)
            print(
                f"Epoch {epoch}/{config.epochs}: generator={averages[0]:.4f}, "
                f"discriminator={averages[4]:.4f}, pore Dice={dice[2]:.4f}, "
                f"{time.time() - started:.1f}s"
            )
