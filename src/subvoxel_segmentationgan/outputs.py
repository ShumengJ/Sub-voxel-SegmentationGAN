"""Prediction serialization and confusion-matrix metrics."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile

from .io import normalize_to_uint8


def class_confusion(target: np.ndarray, prediction: np.ndarray, classes: int = 3):
    """Return TP, TN, FP, and FN arrays for one-indexed label volumes."""

    totals = [np.zeros(classes, dtype=np.int64) for _ in range(4)]
    for index, value in enumerate(range(1, classes + 1)):
        predicted = prediction == value
        expected = target == value
        totals[0][index] = np.sum(predicted & expected)
        totals[1][index] = np.sum(~predicted & ~expected)
        totals[2][index] = np.sum(predicted & ~expected)
        totals[3][index] = np.sum(~predicted & expected)
    return tuple(totals)


def prediction_directories(
    output_dir: Path, *, validation: bool
) -> tuple[Path, Path]:
    """Return the original TIFF and PNG output roots."""

    suffix = "_val" if validation else ""
    output_dir = Path(output_dir)
    return output_dir / f"GAN_TIF{suffix}", output_dir / f"GAN_PNG{suffix}"


def write_prediction(
    output_dir: Path,
    sample_id: int,
    low_res: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    validation: bool = True,
) -> None:
    """Write full TIFF volumes and previews using the original folder names."""

    tif_root, png_root = prediction_directories(output_dir, validation=validation)
    names = {"low_res": low_res, "target": target, "output": prediction}
    for kind, volume in names.items():
        tif_path = tif_root / kind / f"{kind}_{sample_id}.tif"
        png_name = f"low_res_{sample_id}.png" if kind == "low_res" else f"{kind}{sample_id}.png"
        png_path = png_root / kind / png_name
        tif_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(tif_path, volume)
        preview = volume[0]
        if preview.ndim == 3:
            preview = preview[0]
        iio.imwrite(png_path, normalize_to_uint8(preview))


def metric_rows(tp, tn, fp, fn) -> list[dict[str, float]]:
    """Compute per-class and macro metrics from accumulated counts."""

    eps = 1e-8
    metrics = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
    }
    rows = [
        {"class": index + 1, **{name: float(values[index]) for name, values in metrics.items()}}
        for index in range(len(tp))
    ]
    rows.append({"class": "macro", **{name: float(values.mean()) for name, values in metrics.items()}})
    return rows
