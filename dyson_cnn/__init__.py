"""DysonianLineCNN — CNN-based regression of Dysonian EPR line parameters.

Public entry points:
    dyson_cnn.config     — JSON config loading and profile resolution
    dyson_cnn.data       — dataset loading, 3-channel input, normalization
    dyson_cnn.model      — 1D residual CNN architecture
    dyson_cnn.train      — end-to-end training and run-directory saving
    dyson_cnn.evaluate   — test-set evaluation and parity plotting
    dyson_cnn.infer      — inference on real experimental spectra

The package is designed to be imported identically from Google Colab and a
local Mac environment. All paths and hyperparameters come from `config/*.json`
so that notebooks and MATLAB share a single source of truth.
"""

__version__ = "0.1.0"
