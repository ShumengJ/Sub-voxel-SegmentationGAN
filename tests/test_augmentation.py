import numpy as np
import tensorflow as tf

from subvoxel_segmentationgan.augmentation import normalize_3d


def test_zero_volume_normalization_returns_float32():
    normalized = normalize_3d(np.zeros((2, 2, 2), dtype=np.uint16))

    assert normalized.dtype == tf.float32
    assert not normalized.numpy().any()
