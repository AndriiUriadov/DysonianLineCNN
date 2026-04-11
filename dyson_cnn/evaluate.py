"""Test-set evaluation: metrics, parity plots, predictions CSV.

`train_and_save_run` saves the test split inside the run directory as
`_X_test.npy` / `_y_test.npy` so that `evaluate_run` can be called later
from a different notebook/session without re-splitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras

from . import data as data_mod

HEADS = ["B0", "dB", "p3"]


def _predict_and_stack(
    model: tf.keras.Model,
    X: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Predict with the model and return a `(N, 3)` array in `HEADS` order.

    Tolerates list, dict, and plain array returns from `model.predict`
    because Keras' return type depends on the model's output structure.
    """
    pred = model.predict(X, batch_size=batch_size, verbose=0)
    out_names = list(model.output_names)

    if isinstance(pred, list):
        pred_dict = {name: arr for name, arr in zip(out_names, pred)}
        return np.column_stack([pred_dict[k] for k in HEADS]).astype(np.float32)
    if isinstance(pred, dict):
        return np.column_stack([pred[k] for k in HEADS]).astype(np.float32)
    return np.asarray(pred, dtype=np.float32)


def evaluate_run(
    run_dir: str | Path,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    B_axis: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate a saved run on the test set and produce parity plots.

    If `X_test` / `y_test` are not provided, they are loaded from
    `run_dir/_X_test.npy` and `run_dir/_y_test.npy` (saved by
    `train_and_save_run`). If `B_axis` is not provided, it is loaded from
    `run_dir/B_axis.npy`.

    Returns a metrics dict with per-head MAE and RMSE in physical units.
    Also writes `parity_test.png` and `dysonian_test_predictions.csv`.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Load artifacts
    model_path = run_dir / "cnn_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    y_min = np.load(run_dir / "y_min.npy").astype(np.float32).reshape(1, -1)
    y_max = np.load(run_dir / "y_max.npy").astype(np.float32).reshape(1, -1)

    if B_axis is None:
        B_axis = np.load(run_dir / "B_axis.npy").astype(np.float32).squeeze()

    if X_test is None:
        X_test = np.load(run_dir / "_X_test.npy").astype(np.float32)
    if y_test is None:
        y_test = np.load(run_dir / "_y_test.npy").astype(np.float32)

    # Build 3-channel input
    X_test_in = data_mod.make_three_channel_input(X_test, B_axis)

    # Predict and denormalize
    y_pred_norm = _predict_and_stack(model, X_test_in)
    y_pred_norm = np.clip(y_pred_norm, 0.0, 1.0)

    y_true = y_test.astype(np.float32)
    y_pred = data_mod.minmax_invert(y_pred_norm, y_min, y_max)

    # Metrics
    metrics: dict[str, dict[str, float]] = {}
    print("[INFO] Test metrics (denormalized, physical units):")
    for i, name in enumerate(HEADS):
        mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        metrics[name] = {"mae": mae, "rmse": rmse}
        print(f"[INFO] {name:>2}: MAE={mae:.6g}, RMSE={rmse:.6g}")

    # Parity plot
    fig = plt.figure(figsize=(15, 4))
    for i, name in enumerate(HEADS):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.scatter(y_true[:, i], y_pred[:, i], s=10, alpha=0.35)
        lo = float(min(y_true[:, i].min(), y_pred[:, i].min()))
        hi = float(max(y_true[:, i].max(), y_pred[:, i].max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{name}: True vs Predicted")
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(run_dir / "parity_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Predictions CSV
    out_csv = run_dir / "dysonian_test_predictions.csv"
    np.savetxt(
        out_csv,
        np.column_stack([y_true, y_pred]),
        delimiter=",",
        header="B0_true,dB_true,p3_true,B0_pred,dB_pred,p3_pred",
        comments="",
        fmt="%.8f",
    )
    print(f"[INFO] Predictions saved to: {out_csv}")
    print(f"[INFO] Parity plot saved to: {run_dir / 'parity_test.png'}")

    return {"metrics": metrics, "n_samples": int(y_true.shape[0])}
