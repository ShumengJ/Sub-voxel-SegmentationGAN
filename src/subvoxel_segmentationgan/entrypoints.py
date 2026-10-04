"""Installed console entry points."""

import argparse

from .cli import add_config_arguments, config_from_args


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train the sub-voxel segmentation GAN.")
    add_config_arguments(parser)
    args = parser.parse_args()
    from .training import train

    train(config_from_args(args))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test split.")
    add_config_arguments(parser, evaluation=True)
    args = parser.parse_args()
    from .evaluation import evaluate

    for row in evaluate(config_from_args(args, evaluation=True)):
        print(row)
