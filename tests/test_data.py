"""Tests for dyson_cnn.data: normalization, 3-channel input, splits."""

from __future__ import annotations

import numpy as np
import pytest

from dyson_cnn import data as d
from dyson_cnn.data import NPOINTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_X(rng) -> np.ndarray:
    # Shifted and scaled so normalization is nontrivial
    return (rng.normal(0, 1, (10, NPOINTS)) * 3 + 7).astype(np.float32)


@pytest.fixture
def synthetic_y(rng) -> np.ndarray:
    return rng.uniform(0, 1, (10, 3)).astype(np.float32)


@pytest.fixture
def B_axis() -> np.ndarray:
    return np.linspace(0, 5000, NPOINTS, dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-sample normalization primitives
# ---------------------------------------------------------------------------


def test_standardize_per_sample_zero_mean_unit_std(synthetic_X):
    out = d.standardize_per_sample(synthetic_X)
    assert out.shape == synthetic_X.shape
    assert out.dtype == np.float32
    assert np.allclose(out.mean(axis=1), 0, atol=1e-5)
    assert np.allclose(out.std(axis=1), 1, atol=1e-3)


def test_ptp_norm_per_sample_in_unit_range(synthetic_X):
    out = d.ptp_norm_per_sample(synthetic_X)
    assert out.shape == synthetic_X.shape
    assert out.dtype == np.float32
    assert out.min() >= -1.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6
    # Each row spans exactly [-1, 1] (up to eps)
    for i in range(out.shape[0]):
        assert abs(out[i].min() - (-1.0)) < 1e-5
        assert abs(out[i].max() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 3-channel input tensor
# ---------------------------------------------------------------------------


def test_make_three_channel_input_shape_and_dtype(synthetic_X, B_axis):
    out = d.make_three_channel_input(synthetic_X, B_axis)
    assert out.shape == (synthetic_X.shape[0], NPOINTS, 3)
    assert out.dtype == np.float32


def test_make_three_channel_input_channel_order_invariant(synthetic_X, B_axis):
    """CRITICAL regression test: channel order is [z-score, ptp, B_axis].

    Any trained model assumes this exact order. Reversing or swapping
    channels silently breaks inference without raising any error.
    """
    out = d.make_three_channel_input(synthetic_X, B_axis)

    # ch0 must equal standardize_per_sample(X)
    expected_ch0 = d.standardize_per_sample(synthetic_X)
    assert np.allclose(out[:, :, 0], expected_ch0), "ch0 must be z-score"

    # ch1 must equal ptp_norm_per_sample(X)
    expected_ch1 = d.ptp_norm_per_sample(synthetic_X)
    assert np.allclose(out[:, :, 1], expected_ch1), "ch1 must be ptp-normalized"

    # ch2 must be the B_axis positional channel, identical across samples
    ch2 = out[:, :, 2]
    for i in range(1, ch2.shape[0]):
        assert np.allclose(ch2[0], ch2[i]), "ch2 must be identical across samples"
    # And span [-1, 1]
    assert abs(ch2.min() - (-1.0)) < 1e-6
    assert abs(ch2.max() - 1.0) < 1e-6


def test_make_three_channel_input_rejects_1d_input(B_axis):
    with pytest.raises(ValueError, match="2D"):
        d.make_three_channel_input(np.zeros(NPOINTS, dtype=np.float32), B_axis)


def test_make_three_channel_input_rejects_length_mismatch():
    X = np.zeros((2, 128), dtype=np.float32)
    B = np.arange(256, dtype=np.float32)
    with pytest.raises(ValueError, match="mismatch"):
        d.make_three_channel_input(X, B)


def test_baxis_channel_broadcasts_correctly(B_axis):
    out = d.baxis_channel(B_axis, n_samples=5)
    assert out.shape == (5, NPOINTS)
    assert out.dtype == np.float32
    # All rows identical
    for i in range(1, 5):
        assert np.array_equal(out[0], out[i])


# ---------------------------------------------------------------------------
# Label normalization (min/max)
# ---------------------------------------------------------------------------


def test_compute_y_minmax_shape_and_dtype(synthetic_y):
    y_min, y_max = d.compute_y_minmax(synthetic_y)
    assert y_min.shape == (1, 3)
    assert y_max.shape == (1, 3)
    assert y_min.dtype == np.float32
    assert y_max.dtype == np.float32
    assert (y_max >= y_min).all()


def test_minmax_roundtrip_is_identity():
    rng = np.random.default_rng(0)
    y = rng.uniform(-100, 100, (50, 3)).astype(np.float32)
    y_min, y_max = d.compute_y_minmax(y)
    y_norm = d.minmax_apply(y, y_min, y_max)
    y_back = d.minmax_invert(y_norm, y_min, y_max)
    assert np.allclose(y, y_back, atol=1e-4)


def test_minmax_apply_puts_values_in_unit_range():
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 1, (100, 3)).astype(np.float32) * 1000 + 50
    y_min, y_max = d.compute_y_minmax(y)
    y_norm = d.minmax_apply(y, y_min, y_max)
    # On the training data itself, normalization lands in [0, 1] exactly
    assert y_norm.min() >= 0.0 - 1e-6
    assert y_norm.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Dataset manipulation
# ---------------------------------------------------------------------------


def test_apply_max_samples_slices_to_requested_size(synthetic_X, synthetic_y):
    Xs, ys = d.apply_max_samples(synthetic_X, synthetic_y, 3)
    assert Xs.shape[0] == 3
    assert ys.shape[0] == 3
    # Data identity: slice is the first N rows
    assert np.array_equal(Xs, synthetic_X[:3])


def test_apply_max_samples_none_is_noop(synthetic_X, synthetic_y):
    Xs, ys = d.apply_max_samples(synthetic_X, synthetic_y, None)
    assert Xs.shape == synthetic_X.shape
    assert ys.shape == synthetic_y.shape


def test_apply_max_samples_larger_than_n_returns_all(synthetic_X, synthetic_y):
    Xs, ys = d.apply_max_samples(synthetic_X, synthetic_y, 9999)
    assert Xs.shape[0] == synthetic_X.shape[0]


def test_train_val_test_split_ratios():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (1000, NPOINTS)).astype(np.float32)
    y = rng.uniform(0, 1, (1000, 3)).astype(np.float32)
    X_tr, X_v, X_te, y_tr, y_v, y_te = d.train_val_test_split(X, y, seed=42)
    assert X_tr.shape[0] == 700
    assert X_v.shape[0] == 150
    assert X_te.shape[0] == 150
    # Shapes must match between X and y splits
    assert y_tr.shape[0] == 700
    assert y_v.shape[0] == 150
    assert y_te.shape[0] == 150


def test_train_val_test_split_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (500, NPOINTS)).astype(np.float32)
    y = rng.uniform(0, 1, (500, 3)).astype(np.float32)
    r1 = d.train_val_test_split(X, y, seed=42)
    r2 = d.train_val_test_split(X, y, seed=42)
    for a, b in zip(r1, r2):
        assert np.array_equal(a, b), "Same seed must produce same split"


def test_train_val_test_split_different_seeds_differ():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, NPOINTS)).astype(np.float32)
    y = rng.uniform(0, 1, (200, 3)).astype(np.float32)
    X_tr_a, *_ = d.train_val_test_split(X, y, seed=1)
    X_tr_b, *_ = d.train_val_test_split(X, y, seed=2)
    # Not identical (with very high probability)
    assert not np.array_equal(X_tr_a, X_tr_b)
