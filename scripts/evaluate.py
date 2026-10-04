#!/usr/bin/env python3
"""Evaluate a checkpoint on the test split and write predictions."""

import argparse

from subvoxel_segmentationgan.cli import add_config_arguments, config_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser, evaluation=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from subvoxel_segmentationgan.evaluation import evaluate

    for row in evaluate(config_from_args(args, evaluation=True)):
        print(row)


if __name__ == "__main__":
    main()
