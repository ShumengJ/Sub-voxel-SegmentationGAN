#!/usr/bin/env python3
"""Train the sub-voxel segmentation GAN."""

import argparse

from subvoxel_segmentationgan.cli import add_config_arguments, config_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from subvoxel_segmentationgan.training import train

    train(config_from_args(args))


if __name__ == "__main__":
    main()
