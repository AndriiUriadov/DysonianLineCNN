# DysonianLineCNN

A hybrid MATLAB + Python pipeline for regressing Dysonian EPR line parameters
(`B0`, `dB`, `p3`) from first-derivative EPR spectra using a 1D CNN, and for
applying the trained model to real experimental Bruker spectra (`.DTA` / `.DSC`).

**Authors:** A.V. Uriadov, D.V. Savchenko — National Technical University of
Ukraine "Igor Sikorsky Kyiv Polytechnic Institute".

## Pipeline overview

1. **Synthetic dataset generation (MATLAB, local Mac)** —
   [matlab/DysonGeneratorMix.m](matlab/DysonGeneratorMix.m) produces
   `X_dyson_mix_*.npy` / `y_dyson_mix_*.npy` / `B_axis_mix_*.npy` on Google Drive.
2. **CNN training and evaluation (Python, Google Colab)** —
   [notebooks/01_train_and_eval.ipynb](notebooks/01_train_and_eval.ipynb) is a
   thin wrapper around the [dyson_cnn/](dyson_cnn/) package.
3. **Inference on real spectra (MATLAB + Python)** —
   [matlab/PrepareOneSpectrumForCNN.m](matlab/PrepareOneSpectrumForCNN.m)
   resamples a real spectrum onto the run's `B_axis`, then
   [notebooks/02_infer_real.ipynb](notebooks/02_infer_real.ipynb) predicts
   parameters, and [matlab/Validator.m](matlab/Validator.m) reconstructs
   the fitted curve for visual comparison.

## Repository layout

| Path | Contents |
| --- | --- |
| [config/](config/) | JSON configuration files (paths, dataset, training, inference) |
| [dyson_cnn/](dyson_cnn/) | Python package: data, model, training, evaluation, inference |
| [matlab/](matlab/) | MATLAB scripts: dataset generator, spectrum preparation, validator |
| [notebooks/](notebooks/) | Thin Colab notebooks that import from `dyson_cnn/` |
| [tests/](tests/) | `pytest` suite |
| [data/](data/) | Raw experimental Bruker spectra (`.DTA`/`.DSC`) and baseline MATLAB fitters |
| [Documentation/](Documentation/) | Thesis PDFs, architecture diagram, BibTeX |

Data (`.npy`, `.DTA`, `.DSC`) and training artifacts (`runs/`) are **not**
stored in git — they live on Google Drive and are accessed via Drive for
Desktop on Mac or `google.colab.drive.mount` in Colab.

## Getting started

### First-time setup

```bash
# 1) Clone and enter the repo
git clone git@github.com:uriadov/DysonianLineCNN.git
cd DysonianLineCNN

# 2) Create a virtualenv and install the package in editable mode
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3) Copy the paths template and edit for your machine
cp config/paths.example.json config/paths.json
$EDITOR config/paths.json  # set drive_root_mac to your Google Drive mount

# 4) Install the nbstripout git filter (one-time, per clone)
nbstripout --install --attributes .gitattributes
```

The `[dev]` extra pulls in `pytest` and `nbstripout`. See
[pyproject.toml](pyproject.toml) for the full dependency list.

### Running tests

```bash
pytest tests/ -v
```

40 tests cover config loading, normalization invariants, channel order,
model architecture, and inference. All should pass in a few seconds.

### Further reading

- [config/README.md](config/README.md) — every tunable parameter documented
- [MigrationPlan.md](MigrationPlan.md) — the full migration design
- [redacted.md](redacted.md) — context and critical invariants
