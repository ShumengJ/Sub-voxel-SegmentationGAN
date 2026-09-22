from pathlib import Path

import pytest

from subvoxel_segmentationgan.data import paired_volume_paths, split_directory


def _touch(root: Path, directory: str, filename: str) -> Path:
    path = root / directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_paired_volume_paths_matches_numeric_suffix(tmp_path: Path):
    low = _touch(tmp_path, "low_res", "low_res_0002.tif")
    high = _touch(tmp_path, "high_res", "high_res_0002.tif")
    segmentation = _touch(tmp_path, "segmentation", "hr_seg_0002.tif")

    assert paired_volume_paths(tmp_path) == [(low, high, segmentation)]


def test_paired_volume_paths_rejects_missing_volume(tmp_path: Path):
    _touch(tmp_path, "low_res", "low_res_0002.tif")
    _touch(tmp_path, "high_res", "high_res_0002.tif")
    (tmp_path / "segmentation").mkdir()

    with pytest.raises(ValueError, match="segmentation missing"):
        paired_volume_paths(tmp_path)


def test_original_relative_workspace_and_split_layout(tmp_path: Path):
    workspace = tmp_path / "OstrichR496"
    data_root = workspace / "R496_3d"
    split = data_root / "Train"
    low = _touch(split, "low_res", "low_res_0001.tif")
    high = _touch(split, "high_res", "high_res_0001.tif")
    segmentation = _touch(split, "segmentation", "hr_seg_0001.tif")

    assert split_directory(workspace, "Train") == split
    assert split_directory(data_root, "Train") == split
    assert paired_volume_paths(split) == [(low, high, segmentation)]


def test_readme_era_segmentation_prefix_is_an_alias(tmp_path: Path):
    low = _touch(tmp_path, "low_res", "low_res_0002.tif")
    high = _touch(tmp_path, "high_res", "high_res_0002.tif")
    segmentation = _touch(tmp_path, "segmentation", "segmentation_0002.tif")

    assert paired_volume_paths(tmp_path) == [(low, high, segmentation)]


def test_segmentation_prefix_aliases_cannot_be_ambiguous(tmp_path: Path):
    _touch(tmp_path, "low_res", "low_res_0002.tif")
    _touch(tmp_path, "high_res", "high_res_0002.tif")
    _touch(tmp_path, "segmentation", "hr_seg_0002.tif")
    _touch(tmp_path, "segmentation", "segmentation_0002.tif")

    with pytest.raises(ValueError, match="ambiguous files for suffix '0002'"):
        paired_volume_paths(tmp_path)


def test_nested_sample_directories_are_not_guessed(tmp_path: Path):
    _touch(tmp_path, "low_res/specimen-a", "low_res_0002.tif")
    _touch(tmp_path, "high_res/specimen-a", "high_res_0002.tif")
    _touch(tmp_path, "segmentation/specimen-a", "hr_seg_0002.tif")

    with pytest.raises(FileNotFoundError, match="no TIFF volume triplets"):
        paired_volume_paths(tmp_path)
