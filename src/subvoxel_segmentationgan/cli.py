"""Command-line parsing shared by training and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import RunConfig, load_config


def add_config_arguments(parser: argparse.ArgumentParser, *, evaluation: bool = False) -> None:
    parser.add_argument("--config", type=Path, help="JSON configuration file")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="R496_3d dataset root, or its parent, containing Train/ and Test/",
    )
    parser.add_argument("--output-dir", type=Path, help="Training run root")
    parser.add_argument("--checkpoint-dir", type=Path, help="Checkpoint override")
    parser.add_argument("--log-dir", type=Path, help="TensorBoard log-root override")
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        help="Prediction output-root override",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--input-channels", type=int)
    parser.add_argument("--output-channels", type=int)
    parser.add_argument("--seed", type=int)
    if not evaluation:
        parser.add_argument("--epochs", type=int)
        parser.add_argument(
            "--lambda-mse",
            type=float,
            help="Edge-attentive residual MSE weight (released default: 10000)",
        )
        parser.add_argument(
            "--lambda-bce",
            type=float,
            help="Segmentation BCE weight (released default: 100)",
        )


def config_from_args(args: argparse.Namespace, *, evaluation: bool = False) -> RunConfig:
    """Merge CLI overrides over a file or built-in defaults."""

    config = load_config(args.config) if args.config else RunConfig()
    values = {}
    for name in (
        "data_dir",
        "output_dir",
        "checkpoint_dir",
        "log_dir",
        "prediction_dir",
        "batch_size",
        "patch_size",
        "input_channels",
        "output_channels",
        "seed",
        "epochs",
        "lambda_mse",
        "lambda_bce",
    ):
        if hasattr(args, name) and getattr(args, name) is not None:
            values[name] = getattr(args, name)
    config = replace(config, **values).resolved()
    config.validate()
    return config
