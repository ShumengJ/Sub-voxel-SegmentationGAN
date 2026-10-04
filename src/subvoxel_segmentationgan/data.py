"""Dataset discovery and TensorFlow input pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

from .config import RunConfig


def _indexed_files(directory: Path, prefixes: Sequence[str]) -> dict[str, Path]:
    """Index direct child TIFFs and reject ambiguous prefix aliases."""

    files: dict[str, Path] = {}
    for prefix in prefixes:
        for path in sorted(directory.glob(f"{prefix}_*.tif")):
            index = path.stem.removeprefix(f"{prefix}_")
            if index in files:
                raise ValueError(
                    f"ambiguous files for suffix {index!r}: {files[index]} and {path}"
                )
            files[index] = path
    return files


def resolve_data_root(data_dir: Path) -> Path:
    """Resolve either an ``R496_3d`` root or its immediate parent.

    The original layout placed ``R496_3d`` directly below ``OstrichR496``.
    No recursive specimen discovery is performed.
    """

    data_dir = Path(data_dir)
    direct = any((data_dir / split).is_dir() for split in ("Train", "Test"))
    legacy_child = data_dir / "R496_3d"
    nested = any((legacy_child / split).is_dir() for split in ("Train", "Test"))
    if direct and nested:
        raise ValueError(
            f"ambiguous dataset root {data_dir}: both it and {legacy_child} contain splits"
        )
    return legacy_child if nested else data_dir


def split_directory(data_dir: Path, split: str) -> Path:
    """Return a split directory from a direct or legacy parent data root."""

    return resolve_data_root(data_dir) / split


def paired_volume_paths(split_dir: Path) -> list[tuple[Path, Path, Path]]:
    """Discover matched volumes and reject incomplete or ambiguous datasets."""

    split_dir = Path(split_dir)
    directories = {
        "low_res": split_dir / "low_res",
        "high_res": split_dir / "high_res",
        "segmentation": split_dir / "segmentation",
    }
    missing_dirs = [str(path) for path in directories.values() if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError("missing dataset director" + ("ies: " if len(missing_dirs) > 1 else "y: ") + ", ".join(missing_dirs))

    groups = {
        "low_res": _indexed_files(directories["low_res"], ("low_res",)),
        "high_res": _indexed_files(directories["high_res"], ("high_res",)),
        "segmentation": _indexed_files(
            directories["segmentation"], ("hr_seg", "segmentation")
        ),
    }
    indices = {name: set(items) for name, items in groups.items()}
    all_indices = set().union(*indices.values())
    if not all_indices:
        raise FileNotFoundError(f"no TIFF volume triplets found under {split_dir}")

    mismatches = {
        name: sorted(all_indices - present)
        for name, present in indices.items()
        if present != all_indices
    }
    if mismatches:
        detail = "; ".join(f"{name} missing {values}" for name, values in mismatches.items())
        raise ValueError(f"unpaired TIFF volumes in {split_dir}: {detail}")

    return [
        (groups["low_res"][index], groups["high_res"][index], groups["segmentation"][index])
        for index in sorted(all_indices)
    ]


def create_dataset(
    split_dir: Path,
    config: RunConfig,
    *,
    training: bool,
):
    """Create the ``tf.data`` pipeline without importing TensorFlow at CLI startup."""

    import tensorflow as tf

    from .augmentation import load_tiff_train, load_tiff_val

    paths = paired_volume_paths(split_dir)
    columns: Iterable[tuple[str, ...]] = zip(*(tuple(str(path) for path in row) for row in paths))
    dataset = tf.data.Dataset.from_tensor_slices(tuple(columns))
    if training:
        dataset = dataset.shuffle(len(paths), seed=config.seed)

    def load_train(low_res, high_res, segmentation):
        return tf.py_function(
            lambda lr, hr, seg: load_tiff_train(
                lr, hr, seg, config.patch_size, config.output_channels
            ),
            [low_res, high_res, segmentation],
            [tf.float32, tf.float32],
        )

    def load_validation(low_res, high_res, segmentation):
        return tf.py_function(
            lambda lr, hr, seg: load_tiff_val(lr, hr, seg, config.output_channels),
            [low_res, high_res, segmentation],
            [tf.float32, tf.float32],
        )

    loader: Callable = load_train if training else load_validation

    def set_shapes(low_res, segmentation):
        low_res.set_shape([config.patch_size] * 3 + [config.input_channels])
        segmentation.set_shape([config.patch_size] * 3 + [config.output_channels])
        return low_res, segmentation

    return (
        dataset.map(loader, num_parallel_calls=tf.data.AUTOTUNE)
        .map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.batch_size)
        .prefetch(1)
    )
