from pathlib import Path

import numpy as np
import pytest

from subvoxel_segmentationgan.io import normalize_to_uint8
from subvoxel_segmentationgan.outputs import (
    class_confusion,
    metric_rows,
    prediction_directories,
    write_prediction,
)


def test_constant_preview_is_zero_without_division_warning():
    result = normalize_to_uint8(np.ones((2, 2)))
    assert result.dtype == np.uint8
    assert not result.any()


def test_confusion_and_metrics_for_perfect_prediction():
    labels = np.array([[[1, 2, 3]]])
    confusion = class_confusion(labels, labels)
    rows = metric_rows(*confusion)

    assert all(row["dice"] == pytest.approx(1.0) for row in rows)
    assert all(row["precision"] == pytest.approx(1.0) for row in rows)


def test_original_prediction_directories_and_child_names(tmp_path: Path):
    assert prediction_directories(tmp_path, validation=False) == (
        tmp_path / "GAN_TIF",
        tmp_path / "GAN_PNG",
    )
    assert prediction_directories(tmp_path, validation=True) == (
        tmp_path / "GAN_TIF_val",
        tmp_path / "GAN_PNG_val",
    )

    volume = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    write_prediction(tmp_path, 7, volume[np.newaxis], volume, volume, validation=True)

    assert (tmp_path / "GAN_TIF_val/low_res/low_res_7.tif").is_file()
    assert (tmp_path / "GAN_TIF_val/target/target_7.tif").is_file()
    assert (tmp_path / "GAN_TIF_val/output/output_7.tif").is_file()
    assert (tmp_path / "GAN_PNG_val/low_res/low_res_7.png").is_file()
    assert (tmp_path / "GAN_PNG_val/target/target7.png").is_file()
    assert (tmp_path / "GAN_PNG_val/output/output7.png").is_file()
