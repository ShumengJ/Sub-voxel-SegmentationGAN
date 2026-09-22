import pytest
import tensorflow as tf

from subvoxel_segmentationgan.losses import generator_loss


def test_edge_attentive_residual_matches_released_formula():
    input_image = tf.constant([[[[[0.25, -0.5]]]]])
    prediction = tf.constant(
        [
            [
                [[[0.7, 0.1]]],
                [[[0.2, 0.3]]],
                [[[0.1, 0.6]]],
            ]
        ]
    )
    target = tf.constant(
        [
            [
                [[[1.0, 0.0]]],
                [[[0.0, 0.0]]],
                [[[0.0, 1.0]]],
            ]
        ]
    )

    total, gan, residual, bce, _ = generator_loss(
        tf.zeros((1, 1, 1, 1, 1)),
        prediction,
        target,
        input_image,
    )

    expected_residual = tf.reduce_mean(
        tf.square(input_image * prediction - input_image * target)
    )
    assert float(residual) == pytest.approx(float(expected_residual))
    assert float(total) == pytest.approx(
        float(gan + 10000.0 * expected_residual + 100.0 * bce)
    )
