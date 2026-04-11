"""End-to-end smoke test for dyson_cnn.infer using a tiny mock run directory."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from dyson_cnn.data import NPOINTS
from dyson_cnn.infer import HEADS, load_run, predict_real_spectrum


@pytest.fixture
def tiny_run_dir(tmp_path) -> Path:
    """Build a minimal run directory: trivial model + y_min/y_max + B_axis.

    The "model" is a 1-layer global average pool + sigmoid dense. Not the
    real architecture — just enough to exercise load_run / predict_real_spectrum
    end to end without a multi-second training run.
    """
    run_dir = tmp_path / "tiny_run"
    run_dir.mkdir()

    # Trivial model with the same [B0, dB, p3] head structure
    inp = keras.Input(shape=(NPOINTS, 3), name="EPR_input")
    x = keras.layers.GlobalAveragePooling1D()(inp)
    outs = [
        keras.layers.Dense(1, name=name, activation="sigmoid", dtype="float32")(x)
        for name in HEADS
    ]
    model = keras.Model(inp, outs, name="TinyDyson")
    model.save(run_dir / "cnn_model.keras")

    # Plausible y_min/y_max matching the dataset.json narrow-mode ranges
    np.save(run_dir / "y_min.npy", np.array([[3400.0, 250.0, 1.3]], dtype=np.float32))
    np.save(run_dir / "y_max.npy", np.array([[4500.0, 350.0, 1.5]], dtype=np.float32))

    # Full-length B_axis
    B = np.linspace(0, 5000, NPOINTS, dtype=np.float32)
    np.save(run_dir / "B_axis.npy", B)

    return run_dir


def test_load_run_returns_all_artifacts(tiny_run_dir):
    model, y_min, y_max, B_axis = load_run(tiny_run_dir)
    assert isinstance(model, keras.Model)
    assert y_min.shape == (1, 3)
    assert y_max.shape == (1, 3)
    assert B_axis.shape == (NPOINTS,)
    assert (y_max > y_min).all()


def test_load_run_rejects_missing_run_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "nonexistent_run")


def test_predict_real_spectrum_finite_outputs(tmp_path, tiny_run_dir):
    # Synthetic "real" spectrum as a single-column CSV
    rng = np.random.default_rng(0)
    spectrum = rng.normal(0, 1, NPOINTS).astype(np.float32)
    spectrum_csv = tmp_path / "mock_spectrum.csv"
    np.savetxt(spectrum_csv, spectrum, delimiter=",")

    out_json = tmp_path / "prediction.json"
    result = predict_real_spectrum(
        run_dir=tiny_run_dir,
        spectrum_csv_path=spectrum_csv,
        out_json_path=out_json,
    )

    # Structure
    assert "y_pred_physical" in result
    assert "y_pred_norm" in result
    phys = result["y_pred_physical"]
    assert set(phys.keys()) == set(HEADS)

    # Finiteness
    for k, v in phys.items():
        assert math.isfinite(v), f"{k} = {v} not finite"

    # The tiny model uses sigmoid heads, so y_pred_norm ∈ (0, 1), and
    # denormalization lands inside [y_min, y_max] matching dataset.json ranges.
    assert 3400 <= phys["B0"] <= 4500
    assert 250 <= phys["dB"] <= 350
    assert 1.3 <= phys["p3"] <= 1.5

    # JSON was written and is readable
    assert out_json.exists()
    loaded = json.loads(out_json.read_text())
    assert "y_pred_physical" in loaded
    assert loaded["y_pred_physical"] == phys


def test_predict_refuses_wrong_spectrum_length(tmp_path, tiny_run_dir):
    bad_spectrum = np.zeros(1024, dtype=np.float32)
    bad_csv = tmp_path / "wrong_length.csv"
    np.savetxt(bad_csv, bad_spectrum, delimiter=",")

    with pytest.raises(ValueError, match="points"):
        predict_real_spectrum(run_dir=tiny_run_dir, spectrum_csv_path=bad_csv)


def test_predict_without_output_paths_returns_result_only(tmp_path, tiny_run_dir):
    """When out_json_path and out_preview_png_path are None, no files are
    written but the result dict is still returned. Used by pure-function
    tests that only care about the prediction."""
    spectrum = np.zeros(NPOINTS, dtype=np.float32)
    spectrum_csv = tmp_path / "zero.csv"
    np.savetxt(spectrum_csv, spectrum, delimiter=",")

    result = predict_real_spectrum(
        run_dir=tiny_run_dir,
        spectrum_csv_path=spectrum_csv,
        out_json_path=None,
        out_preview_png_path=None,
    )

    assert "y_pred_physical" in result
    assert not (tmp_path / "prediction.json").exists()
    assert not (tmp_path / "preview.png").exists()
