"""TIFF loading and output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import tifffile

PathLikeTensor = Union[str, Path, Any]


def _path_from_tensor(value: PathLikeTensor) -> Path:
    """Convert a scalar TensorFlow string tensor or path-like value to ``Path``."""

    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return Path(value)


def load_tiff(low_res_file: PathLikeTensor, high_res_file: PathLikeTensor, seg_file: PathLikeTensor):
    """Load corresponding low-resolution, high-resolution, and label volumes.

    The high-resolution volume is retained in this interface to match the
    published multiscale data organization, although the checked-in training
    loss consumes only the low-resolution input and segmentation labels.
    """

    return (
        tifffile.imread(_path_from_tensor(low_res_file)),
        tifffile.imread(_path_from_tensor(high_res_file)),
        tifffile.imread(_path_from_tensor(seg_file)),
    )


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Min-max normalize an array for preview output, including constant arrays."""

    image = np.asarray(image)
    minimum = image.min()
    maximum = image.max()
    if maximum == minimum:
        return np.zeros(image.shape, dtype=np.uint8)
    return ((image - minimum) / (maximum - minimum) * 255).astype(np.uint8)
