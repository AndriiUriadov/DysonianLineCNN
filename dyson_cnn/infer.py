"""Inference on a single real experimental EPR spectrum.

The real spectrum is expected as a 1D CSV (one value per line, `Npoints`
rows) produced by `matlab/PrepareOneSpectrumForCNN.m`, which resamples the
Bruker `.DTA` onto the run's `B_axis`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import data as data_mod

HEADS = ["B0", "dB", "p3"]


def load_run(
    run_dir: str | Path,
) -> tuple[tf.keras.Model, np.ndarray, np.ndarray, np.ndarray]:
    """Load a saved run's model, y_min, y_max, and B_axis.

    Returns:
        `(model, y_min, y_max, B_axis)`. `y_min` and `y_max` have shape
        `(1, 3)`; `B_axis` is 1D of length `Npoints`.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    model_path = run_dir / "cnn_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    y_min = np.load(run_dir / "y_min.npy").astype(np.float32).reshape(1, -1)
    y_max = np.load(run_dir / "y_max.npy").astype(np.float32).reshape(1, -1)
    B_axis = np.load(run_dir / "B_axis.npy").astype(np.float32).squeeze()

    if y_min.shape != (1, 3) or y_max.shape != (1, 3):
        raise ValueError(
            f"Expected y_min/y_max shape (1,3), got {y_min.shape}/{y_max.shape}"
        )

    return model, y_min, y_max, B_axis


def predict_real_spectrum(
    run_dir: str | Path,
    spectrum_csv_path: str | Path,
    out_json_path: str | Path | None = None,
    out_preview_png_path: str | Path | None = None,
) -> dict[str, Any]:
    """Predict Dysonian parameters for a single real experimental spectrum.

    Args:
        run_dir: path to the run directory containing the saved model and
            normalization constants.
        spectrum_csv_path: path to a 1-column CSV with `Npoints` values,
            produced by `PrepareOneSpectrumForCNN.m`.
        out_json_path: optional path to write the prediction result JSON.
            If None, nothing is written.
        out_preview_png_path: optional path to write a preview of the three
            input channels fed to the model. If None, no plot is saved.

    Returns:
        A result dict with keys:
            - runName: directory basename
            - spectrum_file: path to the input CSV
            - y_pred_norm: normalized predictions (list of 3 floats)
            - y_pred_physical: denormalized predictions keyed by head name
    """
    run_dir = Path(run_dir)
    spectrum_csv_path = Path(spectrum_csv_path)

    model, y_min, y_max, B_axis = load_run(run_dir)

    x = np.loadtxt(spectrum_csv_path, delimiter=",").astype(np.float32).reshape(-1)
    if x.shape[0] != B_axis.shape[0]:
        raise ValueError(
            f"Spectrum has {x.shape[0]} points but run's B_axis has {B_axis.shape[0]}. "
            f"Run PrepareOneSpectrumForCNN.m against this run's B_axis.csv first."
        )

    # Shape (1, Npoints) so make_three_channel_input sees a proper batch
    X = x.reshape(1, -1)
    X_in = data_mod.make_three_channel_input(X, B_axis)  # (1, Npoints, 3)

    pred = model.predict(X_in, verbose=0)
    out_names = list(model.output_names)

    if isinstance(pred, list):
        y_pred_norm = np.concatenate(
            [p.reshape(1, 1) for p in pred], axis=1
        ).astype(np.float32)
    elif isinstance(pred, dict):
        y_pred_norm = np.column_stack(
            [pred[k].reshape(-1) for k in out_names]
        ).astype(np.float32)
    else:
        y_pred_norm = np.asarray(pred, dtype=np.float32).reshape(1, -1)

    y_pred_norm = np.clip(y_pred_norm, 0.0, 1.0)
    y_pred = data_mod.minmax_invert(y_pred_norm, y_min, y_max)

    result: dict[str, Any] = {
        "runName": run_dir.name,
        "spectrum_file": str(spectrum_csv_path),
        "output_names": out_names,
        "y_pred_norm": y_pred_norm.flatten().tolist(),
        "y_pred_physical": {
            HEADS[i]: float(y_pred[0, i]) for i in range(3)
        },
        "notes": "Input channels: [standardize, ptp(-1..1), B_axis(-1..1)].",
    }

    print("[INFO] Predicted parameters (physical units):")
    for k, v in result["y_pred_physical"].items():
        print(f"[INFO] {k}: {v:.6g}")

    if out_json_path is not None:
        out_json_path = Path(out_json_path)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved results to: {out_json_path}")

    if out_preview_png_path is not None:
        out_png = Path(out_preview_png_path)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 3.5))
        plt.plot(X_in[0, :, 0], label="ch0 (z-score)")
        plt.plot(X_in[0, :, 1], label="ch1 (ptp [-1,1])")
        plt.plot(X_in[0, :, 2], label="ch2 (B-axis [-1,1])")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved input channel preview to: {out_png}")

    return result


def predict_for_set(
    config_dir: str | Path,
    set_name: str,
    spectrum_basename: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """High-level convenience: predict one real spectrum for a given set.

    Resolves all paths from ``config/inference.json`` and
    ``config/sets/<set_name>.json``, then delegates to
    ``predict_real_spectrum``.

    Args:
        config_dir: path to the config/ directory.
        set_name: e.g. ``"set-1"``.
        spectrum_basename: override inference.json ``spectrum_basename``.
        run_name: override inference.json ``runName``.

    Returns:
        The result dict from ``predict_real_spectrum``.
    """
    from . import config as cfg_mod

    paths = cfg_mod.load_paths(config_dir, set_name=set_name)
    inf_cfg = cfg_mod.load_inference_cfg(config_dir)

    run_name = run_name or inf_cfg["runName"]
    basename = spectrum_basename or inf_cfg["spectrum_basename"]

    set_project_dir = Path(paths["set_project_dir"])
    run_dir = Path(paths["set_runs_dir"]) / run_name
    spectrum_csv = set_project_dir / f"{basename}_spectrum.csv"
    out_json = set_project_dir / f"{basename}_real_predicted_params.json"
    out_png = set_project_dir / f"{basename}_spectrum_preview.png"

    return predict_real_spectrum(
        run_dir=run_dir,
        spectrum_csv_path=spectrum_csv,
        out_json_path=out_json,
        out_preview_png_path=out_png,
    )
