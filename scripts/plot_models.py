#!/usr/bin/env python3
"""Render generator and discriminator architecture diagrams."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--patch-size", type=int, default=256)
    args = parser.parse_args()

    from tensorflow.keras.utils import plot_model

    from subvoxel_segmentationgan.models import Discriminator, Generator

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_model(
        Generator(patch_size=args.patch_size),
        show_shapes=True,
        dpi=200,
        to_file=args.output_dir / "generator.png",
    )
    plot_model(
        Discriminator(patch_size=args.patch_size),
        show_shapes=True,
        dpi=200,
        to_file=args.output_dir / "discriminator.png",
    )


if __name__ == "__main__":
    main()
