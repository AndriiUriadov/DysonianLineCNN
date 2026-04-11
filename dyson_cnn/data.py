"""Data loading, 3-channel input construction, and label normalization.

This module is the SINGLE source of truth for how a synthetic dataset or a
real experimental spectrum is converted into the 3-channel input tensor that
the CNN expects. Any other place in the codebase that rebuilds the input
tensor from scratch is a bug.

Critical invariants (do not change without coordinated edits to MATLAB and
existing trained models):

    - Npoints = 4096
    - Channel order: [z-score, ptp-to-[-1,1], B_axis-to-[-1,1]]
    - Train/val/test split is 70/15/15 with fixed seed
    - y_min / y_max are computed from the training split ONLY
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

EPS = 1e-8
NPOINTS = 4096  # hard invariant across the entire pipeline


# ---------------------------------------------------------------------------
# Per-sample normalization primitives
# ---------------------------------------------------------------------------


def standardize_per_sample(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Z-score normalize each row (sample) independently.

    Args:
        X: `(N, Npoints)` array of spectra.

    Returns:
        `(N, Npoints)` float32 array with per-row mean ≈ 0 and std ≈ 1.
    """
    X = X.astype(np.float32, copy=False)
    mu = np.mean(X, axis=1, keepdims=True)
    sd = np.std(X, axis=1, keepdims=True) + eps
    return (X - mu) / sd


def ptp_norm_per_sample(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Normalize each row to `[-1, 1]` by its own peak-to-peak range.

    Args:
        X: `(N, Npoints)` array of spectra.

    Returns:
        `(N, Npoints)` float32 array in `[-1, 1]` per row.
    """
    X = X.astype(np.float32, copy=False)
    xmin = X.min(axis=1, keepdims=True)
    ptp = (X.max(axis=1, keepdims=True) - xmin) + eps
    return (X - xmin) / ptp * 2.0 - 1.0


def baxis_channel(B_axis: np.ndarray, n_samples: int, eps: float = EPS) -> np.ndarray:
    """Build the positional encoding channel from the magnetic field axis.

    Normalizes `B_axis` to `[-1, 1]` min-max and repeats it for `n_samples`
    rows so it can be stacked as a per-sample channel.

    Args:
        B_axis: 1D array of length `Npoints`.
        n_samples: number of rows in the batch to replicate for.

    Returns:
        `(n_samples, Npoints)` float32 array, all rows identical.
    """
    b = np.asarray(B_axis, dtype=np.float32).reshape(1, -1)
    b = (b - b.min()) / (b.max() - b.min() + eps)
    b = b * 2.0 - 1.0
    return np.repeat(b, n_samples, axis=0)


def make_three_channel_input(X: np.ndarray, B_axis: np.ndarray) -> np.ndarray:
    """Stack `[standardize, ptp, B_axis]` into a 3-channel tensor.

    The channel ORDER is load-bearing — the model was trained on exactly
    `[ch0=z-score, ch1=ptp, ch2=B_axis]`. Swapping them silently breaks any
    saved model.

    Args:
        X: `(N, Npoints)` raw spectra. Can be a single spectrum reshaped as
            `(1, Npoints)`.
        B_axis: 1D magnetic field axis of length `Npoints`.

    Returns:
        `(N, Npoints, 3)` float32 array ready to feed to the CNN.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, Npoints), got shape {X.shape}")
    if X.shape[1] != len(B_axis):
        raise ValueError(
            f"Length mismatch: X has {X.shape[1]} points but B_axis has {len(B_axis)}"
        )

    ch0 = standardize_per_sample(X)
    ch1 = ptp_norm_per_sample(X)
    ch2 = baxis_channel(B_axis, X.shape[0])
    return np.stack([ch0, ch1, ch2], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------


def load_dataset(
    paths: dict[str, Any],
    dataset_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the synthetic dataset from the resolved Drive project directory.

    Args:
        paths: result of `config.load_paths()`, must have `project_dir`.
        dataset_cfg: result of `config.load_dataset_cfg()`, must have `Prefix`.

    Returns:
        `(X, y, B_axis, meta)` where X is `(N, Npoints) float32`, y is
        `(N, 3) float32`, B_axis is `(Npoints,) float32`, and meta is the
        MATLAB-produced `meta_mix_<prefix>.json` dict (generator metadata
        snapshot at generation time).
    """
    project_dir = Path(paths["project_dir"])
    prefix = dataset_cfg["Prefix"]

    x_path = project_dir / f"X_dyson_mix_{prefix}.npy"
    y_path = project_dir / f"y_dyson_mix_{prefix}.npy"
    b_path = project_dir / f"B_axis_mix_{prefix}.npy"
    meta_path = project_dir / f"meta_mix_{prefix}.json"

    for p in (x_path, y_path, b_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)
    B_axis = np.load(b_path).astype(np.float32).squeeze()

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape}")
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError(f"Expected y shape (N, 3), got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y have mismatched N: {X.shape[0]} vs {y.shape[0]}")
    if B_axis.ndim != 1 or B_axis.shape[0] != X.shape[1]:
        raise ValueError(
            f"B_axis shape {B_axis.shape} does not match X.shape[1]={X.shape[1]}"
        )
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN/Inf values")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contains NaN/Inf values")

    return X, y, B_axis, meta


def apply_max_samples(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice the dataset down to `max_samples` rows, or no-op if None.

    Used by the `mac_dev` training profile to shortcut the pipeline for
    debugging without regenerating a smaller dataset.
    """
    if max_samples is None:
        return X, y
    if max_samples >= X.shape[0]:
        return X, y
    return X[:max_samples], y[:max_samples]


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """70/15/15 deterministic split. Matches the legacy notebook exactly.

    Returns:
        `(X_train, X_val, X_test, y_train, y_val, y_test)`.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, shuffle=True
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Target normalization
# ---------------------------------------------------------------------------


def compute_y_minmax(y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-head min/max from the TRAINING split only.

    Returns `(y_min, y_max)` with shape `(1, 3)` float32 each. These are the
    ONLY correct values to use for normalization and denormalization of
    predictions — computing them from the full dataset would leak test
    statistics into training.
    """
    y_min = np.min(y_train, axis=0, keepdims=True).astype(np.float32)
    y_max = np.max(y_train, axis=0, keepdims=True).astype(np.float32)
    return y_min, y_max


def minmax_apply(
    y: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """Normalize `y` to `[0, 1]` using the given min/max. Does NOT clip."""
    return (y.astype(np.float32) - y_min) / (y_max - y_min + eps)


def minmax_invert(
    y_norm: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    """Denormalize `y_norm` back to physical units using `y_min` and `y_max`."""
    return y_norm.astype(np.float32) * (y_max - y_min) + y_min
