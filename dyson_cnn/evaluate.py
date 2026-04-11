"""Test-set evaluation: metrics, parity plots, residual distributions, predictions CSV.

`train_and_save_run` saves the test split inside the run directory as
`_X_test.npy` / `_y_test.npy` so that `evaluate_run` can be called later
from a different notebook/session without re-splitting.

Produces the following visualization artifacts in the run directory:

    parity_test.{png,pdf}       — 3-panel combined parity (density-colored)
    parity_B0.{png,pdf}         — full-size B0 parity with residual inset
    parity_dB.{png,pdf}         — full-size dB parity with residual inset
    parity_p3.{png,pdf}         — full-size p3 parity with residual inset
    residuals_test.{png,pdf}    — 3-panel residual histograms

PDF versions are for LaTeX article figures; PNG versions are for quick
preview in a file browser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

from . import data as data_mod

HEADS = ["B0", "dB", "p3"]

# Physical units for axis labels. p3 is dimensionless.
UNITS: dict[str, str] = {"B0": "G", "dB": "G", "p3": ""}

# Default formats for every figure. PDF is for LaTeX article figures;
# PNG is for quick preview in file browsers and on GitHub.
DEFAULT_FORMATS: list[str] = ["png", "pdf"]


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _density_colors(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute point density via Gaussian KDE for colored scatter.

    Returns a flat array of per-point density values. Falls back to zeros
    on degenerate input (e.g., all points identical) so the plot still
    draws without crashing.
    """
    try:
        xy = np.vstack([x, y])
        return gaussian_kde(xy)(xy)
    except Exception:
        return np.zeros(len(x))


def _save_fig(fig: plt.Figure, base_path: Path, formats: list[str]) -> None:
    """Save `fig` to `base_path.<ext>` for each requested format."""
    for ext in formats:
        fig.savefig(f"{base_path}.{ext}", dpi=150, bbox_inches="tight")


def _plot_parity_per_head(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    head_name: str,
    ax: Axes,
    show_inset: bool = True,
) -> None:
    """Draw a per-head true-vs-predicted scatter into an existing axes.

    Features:
        - Density-colored scatter via Gaussian KDE (viridis colormap)
        - y = x diagonal in red dashed
        - Per-head R² and MAE in the subplot title
        - Optional residual-histogram inset in the top-left corner
    """
    density = _density_colors(y_true, y_pred)
    order = density.argsort()
    x_s = y_true[order]
    y_s = y_pred[order]
    d_s = density[order]

    ax.scatter(
        x_s,
        y_s,
        c=d_s,
        s=12,
        alpha=0.75,
        cmap="viridis",
        edgecolors="none",
        rasterized=True,
    )

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    span = hi - lo
    pad = 0.02 * span
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")

    # Metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    unit = UNITS[head_name]
    unit_label = f" ({unit})" if unit else ""

    ax.set_xlabel(f"True {head_name}{unit_label}")
    ax.set_ylabel(f"Predicted {head_name}{unit_label}")
    ax.set_title(f"{head_name}: R² = {r2:.5f},  MAE = {mae:.4g}{unit_label}")
    ax.grid(True, alpha=0.3)

    if show_inset:
        residuals = y_pred - y_true
        axins = inset_axes(
            ax,
            width="38%",
            height="28%",
            loc="upper left",
            borderpad=1.0,
        )
        axins.hist(
            residuals,
            bins=30,
            color="tab:blue",
            alpha=0.75,
            edgecolor="black",
            linewidth=0.3,
        )
        axins.axvline(0, color="r", linestyle="--", linewidth=0.8)
        axins.set_title(
            f"Residuals  μ={float(residuals.mean()):.2g}, σ={float(residuals.std()):.2g}",
            fontsize=7,
        )
        axins.tick_params(axis="both", labelsize=6)
        axins.grid(False)


def _plot_residual_hist_per_head(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    head_name: str,
    ax: Axes,
) -> None:
    """Draw a per-head residual histogram with mean/std annotations."""
    residuals = y_pred - y_true
    mu = float(residuals.mean())
    std = float(residuals.std())

    ax.hist(
        residuals,
        bins=40,
        color="tab:blue",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(0, color="r", linestyle="--", linewidth=1, label="zero")
    ax.axvline(mu, color="k", linestyle="-", linewidth=1, label=f"μ = {mu:.3g}")

    unit = UNITS[head_name]
    unit_label = f" ({unit})" if unit else ""

    ax.set_xlabel(f"Residual: predicted − true {head_name}{unit_label}")
    ax.set_ylabel("Count")
    ax.set_title(f"{head_name}: μ = {mu:.3g},  σ = {std:.3g}{unit_label}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def evaluate_run(
    run_dir: str | Path,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    B_axis: np.ndarray | None = None,
    formats: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a saved run on the test set and produce parity/residual plots.

    If `X_test` / `y_test` are not provided, they are loaded from
    `run_dir/_X_test.npy` and `run_dir/_y_test.npy` (saved by
    `train_and_save_run`). If `B_axis` is not provided, it is loaded from
    `run_dir/B_axis.npy`.

    Args:
        run_dir: path to the training run directory.
        X_test, y_test, B_axis: optional explicit overrides.
        formats: list of file extensions to save figures to. Defaults to
            ["png", "pdf"] — PNG for quick preview, PDF for LaTeX article
            figures. Pass ["png"] to skip PDF, or ["png", "pdf", "svg"]
            for web-ready vector output too.

    Returns a metrics dict with per-head MAE, RMSE, R², mean residual,
    and std of residuals in physical units. Also writes:

        parity_test.{ext}       — 3-panel combined parity
        parity_<head>.{ext}     — full-size per-head parity with inset
        residuals_test.{ext}    — 3-panel residual histograms
        dysonian_test_predictions.csv — raw per-sample predictions
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if formats is None:
        formats = DEFAULT_FORMATS

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

    # --- Metrics ---
    metrics: dict[str, dict[str, float]] = {}
    print("[INFO] Test metrics (denormalized, physical units):")
    for i, name in enumerate(HEADS):
        mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        r2 = float(r2_score(y_true[:, i], y_pred[:, i]))
        residuals = y_pred[:, i] - y_true[:, i]
        metrics[name] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "residual_mean": float(residuals.mean()),
            "residual_std": float(residuals.std()),
        }
        print(
            f"[INFO] {name:>2}: MAE={mae:.6g}, RMSE={rmse:.6g}, "
            f"R²={r2:.6f}, residual μ={residuals.mean():.3g}, σ={residuals.std():.3g}"
        )

    # --- Combined 3-panel parity (density-colored, R² in title) ---
    fig = plt.figure(figsize=(18, 6))
    for i, name in enumerate(HEADS):
        ax = fig.add_subplot(1, 3, i + 1)
        _plot_parity_per_head(
            y_true[:, i], y_pred[:, i], name, ax=ax, show_inset=False
        )
    fig.tight_layout()
    _save_fig(fig, run_dir / "parity_test", formats)
    plt.close(fig)

    # --- Per-head full-size parity with residual histogram inset ---
    for i, name in enumerate(HEADS):
        fig, ax = plt.subplots(figsize=(7, 7))
        _plot_parity_per_head(
            y_true[:, i], y_pred[:, i], name, ax=ax, show_inset=True
        )
        fig.tight_layout()
        _save_fig(fig, run_dir / f"parity_{name}", formats)
        plt.close(fig)

    # --- Combined 3-panel residual histograms ---
    fig = plt.figure(figsize=(15, 4))
    for i, name in enumerate(HEADS):
        ax = fig.add_subplot(1, 3, i + 1)
        _plot_residual_hist_per_head(y_true[:, i], y_pred[:, i], name, ax=ax)
    fig.tight_layout()
    _save_fig(fig, run_dir / "residuals_test", formats)
    plt.close(fig)

    # --- Predictions CSV ---
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
    print(
        f"[INFO] Figures saved to: {run_dir}/{{parity_test,parity_B0,parity_dB,parity_p3,residuals_test}}.{{{','.join(formats)}}}"
    )

    return {"metrics": metrics, "n_samples": int(y_true.shape[0])}
