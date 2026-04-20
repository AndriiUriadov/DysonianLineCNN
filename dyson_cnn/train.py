"""End-to-end training: load dataset, build model, fit, save run directory.

A single call to `train_and_save_run` does everything the legacy monolith
notebook did in its training section, producing an atomic run directory
that downstream evaluation and inference code reads from.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from . import data as data_mod
from .model import build_dysonian_model


# ---------------------------------------------------------------------------
# Precision policy
# ---------------------------------------------------------------------------


def _has_nvidia_gpu() -> bool:
    """Detect whether an NVIDIA GPU is present and visible to TensorFlow."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception:
        return False
    if not gpus:
        return False
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu) or {}
        name = (details.get("device_name") or "").upper()
        if "NVIDIA" in name or any(tag in name for tag in ("TESLA", "RTX", "GTX", "A100", "V100", "T4")):
            return True
    # If we can't tell, assume non-NVIDIA to be safe under Apple Metal.
    return False


def configure_precision(training_cfg: dict[str, Any]) -> str:
    """Apply the mixed-precision policy requested by the training profile.

    Falls back to `float32` with a warning if mixed precision was requested
    but no NVIDIA GPU is available (e.g. Apple Silicon via tensorflow-metal).

    Returns the policy string that was actually applied (`"mixed_float16"`
    or `"float32"`).
    """
    want_mixed = bool(training_cfg.get("mixed_precision", False))
    if not want_mixed:
        return "float32"

    if not _has_nvidia_gpu():
        warnings.warn(
            "mixed_precision=True requested but no NVIDIA GPU detected. "
            "Falling back to float32. (Apple Silicon via tensorflow-metal "
            "does not reliably support mixed_float16.)",
            RuntimeWarning,
        )
        return "float32"

    try:
        keras.mixed_precision.set_global_policy("mixed_float16")
    except Exception as e:
        warnings.warn(f"Failed to enable mixed_float16: {e}. Falling back to float32.")
        return "float32"
    return "mixed_float16"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_and_save_run(
    paths: dict[str, Any],
    dataset_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    generator_meta: dict[str, Any] | None = None,
) -> Path:
    """Load dataset, train the CNN, save the run directory, return its path.

    All artifacts needed to reload and apply the model later are saved into
    a single timestamped subdirectory under `paths["runs_dir"]`:

        cnn_model.keras, y_min.npy, y_max.npy, B_axis.npy, B_axis.csv,
        model_meta.json, history.csv, loss.png, loss_full_and_zoom.png

    Args:
        paths: result of `config.load_paths()`.
        dataset_cfg: result of `config.load_dataset_cfg()`.
        training_cfg: result of `config.load_training_cfg(profile=...)`.
        generator_meta: optional pre-loaded generator metadata dict. If None,
            `data.load_dataset` will try to read it from the Drive-side
            `meta_mix_<prefix>.json` file and pass it through to model_meta.

    Returns:
        `Path` to the created run directory.
    """
    seed = int(training_cfg.get("random_seed", 42))
    tf.keras.utils.set_random_seed(seed)

    precision = configure_precision(training_cfg)

    print(f"[INFO] Profile: {training_cfg.get('profile_name', '<unnamed>')}")
    print(f"[INFO] Precision policy: {precision}")
    print(f"[INFO] Random seed: {seed}")

    # 1) Load dataset from Drive
    X, y, B_axis, meta = data_mod.load_dataset(paths, dataset_cfg)
    if generator_meta is None:
        generator_meta = meta
    print(f"[INFO] Loaded dataset: X {X.shape}, y {y.shape}, B_axis {B_axis.shape}")

    # 2) Optional dev-time slicing
    max_samples = training_cfg.get("max_samples")
    X, y = data_mod.apply_max_samples(X, y, max_samples)
    if max_samples is not None:
        print(f"[INFO] Sliced dataset to {X.shape[0]} samples (max_samples={max_samples})")

    # 3) Train/val/test split (deterministic)
    X_train, X_val, X_test, y_train, y_val, y_test = data_mod.train_val_test_split(
        X, y, seed=seed
    )

    # 4) Target normalization using TRAIN ONLY
    y_min, y_max = data_mod.compute_y_minmax(y_train)
    y_train_norm = np.clip(data_mod.minmax_apply(y_train, y_min, y_max), 0.0, 1.0)
    y_val_norm = np.clip(data_mod.minmax_apply(y_val, y_min, y_max), 0.0, 1.0)

    print(f"[INFO] y_min (train): {y_min.flatten()}")
    print(f"[INFO] y_max (train): {y_max.flatten()}")

    # 5) 3-channel inputs
    X_train_in = data_mod.make_three_channel_input(X_train, B_axis)
    X_val_in = data_mod.make_three_channel_input(X_val, B_axis)

    print(
        f"[INFO] Input tensors: train {X_train_in.shape}, val {X_val_in.shape}"
    )

    # 6) Build model
    Npoints = X.shape[1]
    model = build_dysonian_model(
        input_shape=(Npoints, 3),
        dropout=float(training_cfg.get("dropout", 0.15)),
    )

    # 7) Compile with loss weights from profile
    loss_weights = training_cfg.get("loss_weights", {"B0": 1.0, "dB": 1.0, "p3": 2.0})
    opt = tf.keras.optimizers.Adam(
        learning_rate=float(training_cfg.get("learning_rate", 1e-3)),
        clipnorm=float(training_cfg.get("clipnorm", 1.0)),
    )
    model.compile(
        optimizer=opt,
        loss={"B0": "mse", "dB": "mse", "p3": "mse"},
        loss_weights={
            "B0": float(loss_weights.get("B0", 1.0)),
            "dB": float(loss_weights.get("dB", 1.0)),
            "p3": float(loss_weights.get("p3", 2.0)),
        },
        metrics={"B0": "mae", "dB": "mae", "p3": "mae"},
    )
    print("[INFO] Model compiled. Summary:")
    model.summary()

    # 8) Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(training_cfg.get("early_stopping_patience", 40)),
        min_delta=1e-6,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=float(training_cfg.get("reduce_lr_factor", 0.5)),
        patience=int(training_cfg.get("reduce_lr_patience", 10)),
        min_lr=float(training_cfg.get("reduce_lr_min", 1e-6)),
        verbose=1,
    )

    # 9) Target dicts for named outputs
    y_train_dict = {
        "B0": y_train_norm[:, 0],
        "dB": y_train_norm[:, 1],
        "p3": y_train_norm[:, 2],
    }
    y_val_dict = {
        "B0": y_val_norm[:, 0],
        "dB": y_val_norm[:, 1],
        "p3": y_val_norm[:, 2],
    }

    # 10) Train
    history = model.fit(
        X_train_in,
        y_train_dict,
        validation_data=(X_val_in, y_val_dict),
        epochs=int(training_cfg.get("epochs", 300)),
        batch_size=int(training_cfg.get("batch_size", 64)),
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # 11) Create timestamped run directory
    stamp = time.strftime("%Y%m%d_%H%M%S")
    runs_base = Path(paths.get("set_runs_dir", paths["runs_dir"]))
    run_dir = runs_base / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Run directory: {run_dir}")

    # 12) Save artifacts
    model.save(run_dir / "cnn_model.keras")
    np.save(run_dir / "y_min.npy", y_min.astype(np.float32))
    np.save(run_dir / "y_max.npy", y_max.astype(np.float32))
    np.save(run_dir / "B_axis.npy", B_axis.astype(np.float32))
    np.savetxt(run_dir / "B_axis.csv", B_axis.reshape(-1), delimiter=",")

    meta_out = {
        "created": stamp,
        "set_name": paths.get("set_name"),
        "geometry": dataset_cfg.get("geometry", "plate"),
        "dataset_prefix": dataset_cfg.get("Prefix", "unknown"),
        "profile_name": training_cfg.get("profile_name", "unknown"),
        "precision_policy": precision,
        "input_shape": list(X_train_in.shape[1:]),
        "heads": ["B0", "dB", "p3"],
        "loss_weights": {
            "B0": float(loss_weights.get("B0", 1.0)),
            "dB": float(loss_weights.get("dB", 1.0)),
            "p3": float(loss_weights.get("p3", 2.0)),
        },
        "y_min": y_min.flatten().tolist(),
        "y_max": y_max.flatten().tolist(),
        "random_seed": seed,
        "max_samples": max_samples,
        "generator_meta": generator_meta,
    }
    with open(run_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    pd.DataFrame(history.history).to_csv(run_dir / "history.csv", index=False)

    # 13) Loss plots
    _plot_losses(history, run_dir)

    # 14) Save test split for use by evaluate_run without re-splitting
    np.save(run_dir / "_X_test.npy", X_test.astype(np.float32))
    np.save(run_dir / "_y_test.npy", y_test.astype(np.float32))

    print(f"[INFO] Artifacts saved to: {run_dir}")
    return run_dir


def _plot_losses(history: keras.callbacks.History, run_dir: Path) -> None:
    """Save both single and dual (full + zoomed) loss curves."""
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # Single plot
    plt.figure(figsize=(7.5, 5))
    plt.plot(loss, label="Training")
    plt.plot(val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training / Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(run_dir / "loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Dual: full + zoomed tail
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(loss, label="Training")
    ax[0].plot(val_loss, label="Validation")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss (MSE)")
    ax[0].set_title("Training / Validation Loss (full)")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(loss, label="Training")
    ax[1].plot(val_loss, label="Validation")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss (MSE)")
    ax[1].set_title("Training / Validation Loss (zoomed tail)")
    ax[1].set_ylim(0.0, 0.01)
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(run_dir / "loss_full_and_zoom.png", dpi=150, bbox_inches="tight")
    plt.close()
