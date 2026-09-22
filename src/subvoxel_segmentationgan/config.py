"""Configuration and repository-independent path handling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class RunConfig:
    """Settings shared by training and evaluation.

    Relative paths are interpreted from the current working directory, not from
    a developer-specific home directory.
    """

    data_dir: Path = Path("OstrichR496/R496_3d")
    output_dir: Path = Path("OstrichR496/Tensorflow/SRSegGAN")
    checkpoint_dir: Optional[Path] = None
    log_dir: Optional[Path] = None
    prediction_dir: Optional[Path] = None
    batch_size: int = 1
    patch_size: int = 256
    epochs: int = 300
    input_channels: int = 1
    output_channels: int = 3
    lambda_mse: float = 10000.0  # Edge-attentive residual MSE weight.
    lambda_bce: float = 100.0  # Segmentation BCE weight.
    seed: int = 1

    @property
    def resolved_checkpoint_dir(self) -> Path:
        return self.checkpoint_dir or self.output_dir / "training_checkpoints"

    @property
    def resolved_log_dir(self) -> Path:
        return self.log_dir or self.output_dir / "logs" / "fit"

    @property
    def resolved_prediction_dir(self) -> Path:
        return self.prediction_dir or self.output_dir / "output"

    @property
    def epoch_log_path(self) -> Path:
        return self.output_dir / "epoch_loss_log.txt"

    def resolved(self, base_dir: Optional[Path] = None) -> "RunConfig":
        """Return a copy with all paths absolute.

        ``base_dir`` defaults to the caller's current working directory.
        """

        base = (base_dir or Path.cwd()).resolve()

        def absolute(path: Optional[Path]) -> Optional[Path]:
            if path is None:
                return None
            return path.expanduser().resolve() if path.is_absolute() else (base / path).resolve()

        return replace(
            self,
            data_dir=absolute(self.data_dir),
            output_dir=absolute(self.output_dir),
            checkpoint_dir=absolute(self.checkpoint_dir),
            log_dir=absolute(self.log_dir),
            prediction_dir=absolute(self.prediction_dir),
        )

    def validate(self) -> None:
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if self.patch_size < 256 or self.patch_size % 256:
            raise ValueError("patch_size must be a positive multiple of 256 for the eight-level U-Net")
        if self.input_channels != 1 or self.output_channels != 3:
            raise ValueError(
                "the released model requires one input channel and three output classes"
            )


def load_config(path: Path) -> RunConfig:
    """Load a JSON configuration, rejecting unknown keys."""

    path = path.expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        values: Mapping[str, Any] = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError("configuration must contain a JSON object")

    allowed = {field.name for field in fields(RunConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown configuration key(s): {', '.join(sorted(unknown))}")

    converted = dict(values)
    for name in ("data_dir", "output_dir", "checkpoint_dir", "log_dir", "prediction_dir"):
        if converted.get(name) is not None:
            converted[name] = Path(converted[name])
    config = RunConfig(**converted).resolved(path.parent)
    config.validate()
    return config


def config_dict(config: RunConfig) -> dict[str, Any]:
    """Return JSON-serializable configuration values."""

    result = asdict(config)
    for name in ("data_dir", "output_dir", "checkpoint_dir", "log_dir", "prediction_dir"):
        if result[name] is not None:
            result[name] = str(result[name])
    return result
