import json
from pathlib import Path

import pytest

from subvoxel_segmentationgan.config import RunConfig, load_config
from subvoxel_segmentationgan.training import tensorboard_log_dir


def test_relative_paths_resolve_from_config_file(tmp_path: Path):
    config_path = tmp_path / "configs" / "run.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "data_dir": "../dataset",
                "output_dir": "../results",
                "log_dir": "../custom-logs",
                "prediction_dir": "../custom-predictions",
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data_dir == (tmp_path / "dataset").resolve()
    assert config.output_dir == (tmp_path / "results").resolve()
    assert config.resolved_checkpoint_dir == config.output_dir / "training_checkpoints"
    assert config.resolved_log_dir == (tmp_path / "custom-logs").resolve()
    assert config.resolved_prediction_dir == (tmp_path / "custom-predictions").resolve()


def test_default_paths_match_original_relative_layout():
    config = RunConfig()

    assert config.data_dir == Path("OstrichR496/R496_3d")
    assert config.output_dir == Path("OstrichR496/Tensorflow/SRSegGAN")
    assert config.resolved_checkpoint_dir == config.output_dir / "training_checkpoints"
    assert config.resolved_log_dir == config.output_dir / "logs" / "fit"
    assert tensorboard_log_dir(config, "20250102-030405") == (
        config.output_dir / "logs" / "fit" / "20250102-030405"
    )
    assert config.epoch_log_path == config.output_dir / "epoch_loss_log.txt"
    assert config.resolved_prediction_dir == config.output_dir / "output"


def test_unknown_config_key_is_rejected(tmp_path: Path):
    config_path = tmp_path / "run.json"
    config_path.write_text('{"unsupported_setting": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="unknown configuration"):
        load_config(config_path)


def test_model_patch_size_constraint():
    with pytest.raises(ValueError, match="multiple of 256"):
        RunConfig(patch_size=128).validate()


def test_released_channel_contract_is_enforced():
    with pytest.raises(ValueError, match="one input channel and three output"):
        RunConfig(output_channels=2).validate()
