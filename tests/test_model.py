"""Tests for dyson_cnn.model: architecture invariants and forward pass."""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pytest
import tensorflow as tf

from dyson_cnn.model import build_dysonian_model


# REGRESSION TEST ANCHOR: this number is pinned from the first clean build
# of the refactored architecture (Phase 2 smoke test). If it changes, either
# the architecture was modified or a layer was added/removed. Either way,
# every saved model from before the change becomes incompatible. Update this
# constant only after a deliberate architectural decision.
EXPECTED_PARAM_COUNT = 691_283


def test_model_builds_with_expected_io_shape():
    model = build_dysonian_model(input_shape=(4096, 3), dropout=0.15)
    assert model.input_shape == (None, 4096, 3)
    assert len(model.outputs) == 3
    assert model.name == "DysonCNN_mix"


def test_model_parameter_count_regression():
    """If this fails, the architecture was accidentally modified."""
    model = build_dysonian_model(input_shape=(4096, 3))
    actual = model.count_params()
    assert actual == EXPECTED_PARAM_COUNT, (
        f"Parameter count drifted from {EXPECTED_PARAM_COUNT} to {actual}. "
        f"The CNN architecture was modified — all saved models from before "
        f"this change are now incompatible. Investigate and update "
        f"EXPECTED_PARAM_COUNT only if the change was intentional."
    )


def test_model_output_names_in_fixed_order():
    """Output heads must be exactly [B0, dB, p3] in that order.

    The evaluation and inference code assumes this order when converting
    the model's list/dict predictions back to a (N, 3) array. Swapping
    heads silently swaps the physical meaning of each column.
    """
    model = build_dysonian_model(input_shape=(4096, 3))
    assert model.output_names == ["B0", "dB", "p3"]


def test_model_forward_pass_on_zero_tensor():
    model = build_dysonian_model(input_shape=(4096, 3))
    x = tf.zeros((2, 4096, 3))
    outputs = model(x, training=False)
    assert len(outputs) == 3
    for out in outputs:
        assert out.shape == (2, 1)


def test_model_forward_pass_on_random_tensor():
    model = build_dysonian_model(input_shape=(4096, 3))
    x = tf.random.normal((4, 4096, 3), seed=0)
    outputs = model(x, training=False)
    assert len(outputs) == 3
    for out in outputs:
        assert out.shape == (4, 1)
        # Outputs must be finite (no NaN/Inf from the forward pass)
        assert tf.math.reduce_all(tf.math.is_finite(out))


def test_output_layers_stay_float32_under_mixed_precision():
    """Output heads must remain float32 for numerical stability even when
    the global Keras policy is mixed_float16."""
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        model = build_dysonian_model(input_shape=(4096, 3))
        for out in model.outputs:
            assert out.dtype == tf.float32, (
                f"Output {out.name} has dtype {out.dtype} under mixed_float16 policy; "
                f"must be float32 to avoid underflow in the regression heads."
            )
    finally:
        tf.keras.mixed_precision.set_global_policy("float32")
