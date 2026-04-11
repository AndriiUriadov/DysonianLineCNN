"""1D residual CNN architecture for Dysonian EPR parameter regression.

The architecture is copied verbatim from the legacy monolithic notebook.
It must not be modified during the migration — any change would invalidate
the equivalence check in Phase 3 and break existing trained models.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import models


def _res_block(
    x: tf.Tensor,
    filters: int,
    k: int = 3,
    dilation: int = 1,
) -> tf.Tensor:
    """Residual 1D convolutional block with optional dilation.

    Structure:
        Conv1D(filters, k, ReLU) -> Conv1D(filters, k, linear) -> Add -> ReLU

    If the input channel count does not match `filters`, a 1x1 Conv1D
    projection is applied to the skip connection.
    """
    y = L.Conv1D(
        filters, k, padding="same", dilation_rate=dilation, activation="relu"
    )(x)
    y = L.Conv1D(
        filters, k, padding="same", dilation_rate=dilation, activation=None
    )(y)
    if x.shape[-1] != filters:
        x = L.Conv1D(filters, 1, padding="same", activation=None)(x)
    y = L.Add()([x, y])
    y = L.Activation("relu")(y)
    return y


def build_dysonian_model(
    input_shape: tuple[int, int],
    dropout: float = 0.15,
) -> tf.keras.Model:
    """Build the multi-output 1D CNN for Dysonian EPR spectra regression.

    Outputs three named heads in this order: `B0`, `dB`, `p3`. The output
    layers are explicitly cast to `float32` so they remain numerically stable
    under mixed-precision training.

    Compilation is NOT done here — `dyson_cnn.train.train_and_save_run` calls
    `model.compile` after construction so that loss weights and optimizer
    hyperparameters come from `training.json`.

    Args:
        input_shape: `(Npoints, 3)` — time-series length and channel count.
        dropout: dropout rate before the per-head dense stack.

    Returns:
        Uncompiled `tf.keras.Model` with name `"DysonCNN_mix"`.
    """
    inp = L.Input(shape=input_shape, name="EPR_input")

    x = L.Conv1D(32, 9, padding="same", activation="relu")(inp)
    x = L.Conv1D(48, 7, padding="same", activation="relu")(x)

    x = L.MaxPooling1D(2)(x)
    x = _res_block(x, 64, k=5, dilation=1)
    x = _res_block(x, 64, k=5, dilation=2)

    x = L.MaxPooling1D(2)(x)
    x = _res_block(x, 96, k=5, dilation=2)
    x = _res_block(x, 96, k=3, dilation=4)

    x = L.MaxPooling1D(2)(x)
    x = _res_block(x, 128, k=3, dilation=4)
    x = _res_block(x, 128, k=3, dilation=8)

    x = L.GlobalAveragePooling1D()(x)
    x = L.Dense(512, activation="relu")(x)
    x = L.Dropout(dropout)(x)

    def head(name: str) -> tf.Tensor:
        h = L.Dense(128, activation="relu")(x)
        return L.Dense(1, name=name, activation="linear", dtype="float32")(h)

    out_B0 = head("B0")
    out_dB = head("dB")
    out_p3 = head("p3")

    return models.Model(
        inp,
        [out_B0, out_dB, out_p3],
        name="DysonCNN_mix",
    )
