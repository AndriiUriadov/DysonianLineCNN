# DysonianLineCNN

A hybrid MATLAB + Python pipeline for extracting Dysonian EPR line parameters
(`B0`, `dB`, `p`) from first-derivative EPR spectra. Three methods are
compared on 180 experimental spectra across 5 independent datasets:

1. **MATLAB optimizer** — `lsqnonlin` with Feher-Kip Dysonian model
2. **EasySpin esfit** — Levenberg-Marquardt, two-step approach
3. **1D Residual CNN** — trained on synthetic spectra, predicts parameters
   directly from the normalized spectrum shape

All three methods use the same physical model with the full two-term A/D
coefficients (Holiatkina et al., *J. Appl. Phys.* **134**, 145702, 2023).

**Authors:** A.V. Uriadov, D.V. Savchenko — National Technical University of
Ukraine "Igor Sikorsky Kyiv Polytechnic Institute".

## Pipeline overview (per-set)

Each experimental set has its own parameter ranges and magnetic field window,
requiring an independent CNN model. The workflow for each set:

1. **Classical baselines (MATLAB)** —
   [matlab/baseline/FitAll.m](matlab/baseline/FitAll.m) fits all spectra in
   `data/set-N/` with both MATLAB lsqnonlin and EasySpin esfit, saving
   results to `results/set-N/{matlab,easyspin}/`.

2. **Synthetic dataset generation (MATLAB)** —
   [matlab/DysonGeneratorMix.m](matlab/DysonGeneratorMix.m) reads per-set
   config from [config/sets/set-N.json](config/sets/) and writes
   `.npy` files to Google Drive (`DysonianLineCNN/set-N/`).

3. **CNN training (Google Colab)** —
   [notebooks/01_train_and_eval.ipynb](notebooks/01_train_and_eval.ipynb)
   trains the CNN and saves the run to `Drive/DysonianLineCNN/set-N/runs/`.

4. **Inference on real spectra** —
   [matlab/PrepareOneSpectrumForCNN.m](matlab/PrepareOneSpectrumForCNN.m)
   resamples spectra onto the run's `B_axis`, then
   [dyson_cnn.infer.predict_for_set](dyson_cnn/infer.py) predicts parameters,
   and [matlab/Validator.m](matlab/Validator.m) overlays the reconstruction.

5. **Comparison** — Side-by-side results from all three methods in
   `results/set-N/comparison.csv`.

## Repository layout

| Path | Contents |
| --- | --- |
| [config/](config/) | JSON configs: paths, training profiles, inference target |
| [config/sets/](config/sets/) | Per-set dataset generation parameters (B0/dB/p ranges, BWindow) |
| [dyson_cnn/](dyson_cnn/) | Python package: data, model, training, evaluation, inference |
| [matlab/](matlab/) | MATLAB scripts: dataset generator, spectrum prep, validator |
| [matlab/baseline/](matlab/baseline/) | Classical fitters: FitMatlab.m, FitEasyspin.m, FitAll.m |
| [notebooks/](notebooks/) | Thin Colab/Mac notebooks with `SET_NAME` variable |
| [tests/](tests/) | 49 pytest tests |
| [results/](results/) | Summary CSVs and comparison tables (committed); per-spectrum JSON/PNG (local) |
| [scripts/](scripts/) | Figure generation scripts (e.g. `figure6_comparison.py`) |
| [figures/](figures/) | Generated publication figures (final only; `temp/` gitignored) |
| [data/](data/) | Raw experimental Bruker spectra in `set-1/` through `set-5/` (not in git) |

Data (`.npy`, `.DTA`, `.DSC`) and training artifacts (trained CNN models
under `results/set-N/cnn/runs/<stamp>/`) are **not** stored in git — they
live on Google Drive and are accessed via Drive for Desktop on Mac or
`google.colab.drive.mount` in Colab.

### Google Drive layout

The Drive mirror contains exactly the same `results/` tree as the local
repo, plus trained CNN models inside `results/set-N/cnn/runs/<stamp>/` and
a `data/` mirror:

```text
<Drive>/Python/DysonianLineCNN/
  config/, dyson_cnn/, matlab/, notebooks/     — code mirror (read-only from Colab)
  data/set-N/                                   — raw Bruker .DTA/.DSC mirror
  results/
    set-N/
      matlab/, easyspin/   — classical fit JSONs + overlay PNGs
      cnn/
        <id>_predicted.json, <id>_cnn_fit.png   — per-spectrum CNN outputs
        summary.csv
        runs/<YYYYMMDD_HHMMSS>/                 — trained CNN model
          cnn_model.keras, y_min.npy, y_max.npy
          B_axis.csv, B_axis.npy, model_meta.json
          parity_*.png, loss.png, history.csv
  report.pdf                                    — compiled LaTeX report
```

## Example results

Sample outputs from processing all 180 experimental spectra across 5 sets
(MATLAB, EasySpin, and CNN fit overlays, parity plots, loss curves):

[**View results on Google Drive**](https://drive.google.com/drive/folders/1wRbYo90H6e9iGxK2dmwY1a97YAnGb8OC?usp=share_link)

The folder contains per-set subdirectories with:
- **Trained CNN models** (`runs/<stamp>/cnn_model.keras`, `model_meta.json`,
  `y_min.npy`, `y_max.npy`, `B_axis.csv`)
- **Parity plots** — true vs. predicted scatter with R² and MAE for each
  output head (B0, dB, p)
- **Loss curves** — training/validation loss with zoomed tail
- **Fit overlays** — per-spectrum PNG comparing experimental (green) and
  reconstructed (red) curves for each method
- **Prediction JSONs** — CNN-predicted B0, dB, p for each real spectrum

## Run directory contents

Each training run (`notebooks/01_train_and_eval.ipynb` →
`dyson_cnn.train.train_and_save_run`) writes a timestamped subfolder
under `<project>/runs/<YYYYMMDD_HHMMSS>/` on Google Drive. A run
directory is the atomic unit of reproducibility — the model file alone
is useless without the normalization statistics and the axis that were
used at training time. Never copy `cnn_model.keras` out of its run dir.

Contents of a typical `colab_full` run (`N=10000`, `Npoints=4096`):

| File | Approx. size | Purpose |
| --- | --- | --- |
| `cnn_model.keras` | 13 MB | Trained Keras model with weights |
| `model_meta.json` | 1 KB | Generator parameters snapshot, training profile, seed, head names, loss weights, `y_min`/`y_max` |
| `y_min.npy` / `y_max.npy` | 140 B each | Per-head min/max computed from the training split only; used to denormalize predictions back to physical units |
| `B_axis.npy` / `B_axis.csv` | 16 KB / 100 KB | Magnetic field axis consumed by both inference and [matlab/PrepareOneSpectrumForCNN.m](matlab/PrepareOneSpectrumForCNN.m) to resample real spectra onto the exact axis the model was trained on |
| `history.csv` | 35 KB | Per-epoch training and validation metrics (one row per epoch) for custom plots |
| `loss.png` | 40 KB | Training / validation loss curve |
| `loss_full_and_zoom.png` | 100 KB | Same curve plus a zoomed tail region for inspecting late-epoch convergence |
| `parity_test.png` | 260 KB | Combined 3-panel true-vs-predicted scatter (`B0`, `dB`, `p3`) on the test set, density-colored, with `R²` and MAE in each subplot title |
| `parity_B0.png` | 120 KB | Full-size B0 parity for article figures with residual histogram inset |
| `parity_dB.png` | 110 KB | Full-size dB parity, same format |
| `parity_p3.png` | 130 KB | Full-size p3 parity, same format |
| `residuals_test.png` | 70 KB | Combined 3-panel residual histograms (`y_pred − y_true`) with μ and σ annotated |
| `dysonian_test_predictions.csv` | 110 KB | Per-sample `(B0, dB, p3)` true and predicted values in physical units for ad-hoc analysis |
| `_X_test.npy` / `_y_test.npy` | 23 MB / 18 KB | Cached test split so `evaluate_run` can be replayed in a separate Jupyter session without re-splitting the dataset |

Figures are saved as PNG by default. For LaTeX article embeddings you
can opt into vector output via
`evaluate_run(run_dir, formats=["png", "pdf"])` (or add `"svg"` too).

The leading underscore on `_X_test.npy` / `_y_test.npy` marks them as
internal implementation files — they let you call `evaluate_run(run_dir)`
weeks later from a different notebook session without loading the full
10 000-sample dataset back into memory. Safe to delete if you need the
disk space back; `evaluate_run` will just fail with a clear error.

### Current production runs

Each set has an independent CNN model trained on set-specific synthetic data.

| Set | Run | Spectra | B0 MAE (G) | dB MAE (G) | p MAE |
| --- | --- | --- | --- | --- | --- |
| set-1 | `20260412_122613` | 6 | 1.14 | 0.27 | 0.012 |
| set-2 | `20260412_125230` | 2 | 5.37 | 5.14 | 0.012 |
| set-3 | `20260412_135545` | 53 | 0.67 | 0.51 | 0.021 |
| set-4 | `20260412_141449` | 80 | 0.56 | 0.77 | 0.022 |
| set-5 | `20260412_143406` | 39 | 0.80 | 0.33 | 0.015 |

All models: 691,283 parameters, `colab_full` profile, R² > 0.998 on all heads.
See [Analysis.md](Analysis.md) for the full three-method comparison.

## Getting started

### First-time setup (local development)

These instructions cover running the pipeline on your own machine for
dataset generation (MATLAB), Python unit tests, and local model training.
For Google Colab setup see the [Colab setup](#colab-setup) section — no
local setup is needed for Colab-only workflows.

The Python side is platform-independent and tested on macOS, Windows, and
Linux. MATLAB is available on all three platforms as well.

#### Step 1. Clone the repository and create a virtualenv

```bash
git clone git@github.com:AndriiUriadov/DysonianLineCNN.git
cd DysonianLineCNN
python3 -m venv .venv
```

#### Step 2. Activate the virtualenv

The command depends on your OS and shell:

| OS / shell | Activation command |
| --- | --- |
| macOS and Linux (bash, zsh) | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| Windows Git Bash or WSL | `source .venv/Scripts/activate` |

On Windows PowerShell, first-time activation may require allowing local
scripts once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

#### Step 3. Install the package and the git filter

With the venv active:

```bash
pip install -e ".[dev]"
nbstripout --install --attributes .gitattributes
```

The `[dev]` extra pulls in `pytest`, `nbstripout`, and `nbformat`. See
[pyproject.toml](pyproject.toml) for the full dependency list.

#### Step 4. Create `config/paths.json` for your machine

This file is gitignored because the local Drive mount path is per-user
and per-OS. Copy the template first:

| OS / shell | Copy command |
| --- | --- |
| macOS, Linux, Git Bash, WSL | `cp config/paths.example.json config/paths.json` |
| Windows PowerShell | `Copy-Item config\paths.example.json config\paths.json` |
| Windows Command Prompt | `copy config\paths.example.json config\paths.json` |

Then open `config/paths.json` in any text editor and set `drive_root_mac`
to the absolute path of your Google Drive mount root (see the next
subsection for platform-specific paths).

> The field is named `drive_root_mac` for historical reasons, but it is
> used by `load_paths()` on every non-Colab runtime (macOS, Windows,
> Linux). Treat it as "local Drive root" regardless of your OS.

##### Drive mount paths by OS

**macOS** — install [Google Drive for Desktop](https://www.google.com/drive/download/).
The mount path is typically:

```text
/Users/<username>/Library/CloudStorage/GoogleDrive-<email>/My Drive
```

If your system locale is set to e.g. Ukrainian, the last segment appears
localized as `Мій диск` instead of `My Drive`. Use the localized name
that you actually see in Finder.

**Windows** — install [Google Drive for Desktop](https://www.google.com/drive/download/).
Modern installations mount Drive as a virtual drive letter (commonly `G:`):

```text
G:\My Drive
```

Older installations may instead use a folder under the user profile:

```text
C:\Users\<username>\Google Drive\My Drive
```

In the JSON file you can write backslashes either as forward slashes
(simplest) or as escaped pairs:

```json
"drive_root_mac": "G:/My Drive"
```

```json
"drive_root_mac": "G:\\My Drive"
```

**Linux** — Google does not ship an official Drive client for Linux. The
working options are:

- **[rclone](https://rclone.org/drive/)** — recommended for scripting-heavy
  workflows. After `rclone config` to authorize access:

  ```bash
  mkdir -p ~/gdrive
  rclone mount gdrive: ~/gdrive --daemon
  ```

  Then set `"drive_root_mac": "/home/<username>/gdrive"` in `paths.json`.

- **[google-drive-ocamlfuse](https://github.com/astrada/google-drive-ocamlfuse)** —
  a FUSE-based userland filesystem with similar semantics.

- **Manual sync via the Drive web UI** — slower but dependency-free.
  In that case `drive_root_mac` points at any local directory where you
  mirror the project files, and you copy datasets and trained runs back
  and forth by hand.

#### Colab first-time setup

On Google Colab there is no local install step beyond adding the
`GITHUB_DEPLOY_KEY` secret (see [Colab setup](#colab-setup) below). The
notebooks bootstrap themselves on a fresh runtime and `load_paths`
automatically falls back to `paths.example.json` because `drive_root_colab`
is universal across users (`/content/drive/MyDrive`).

### Running tests

```bash
pytest tests/ -v
```

49 tests cover config loading (including per-set), normalization invariants,
channel order, model architecture, and inference. All should pass in a few seconds.

### Colab setup

The training notebooks ([notebooks/01_train_and_eval.ipynb](notebooks/01_train_and_eval.ipynb)
and [notebooks/02_infer_real.ipynb](notebooks/02_infer_real.ipynb)) are
thin wrappers around `dyson_cnn/*` and run on both Google Colab and local
Mac. On a fresh Colab runtime, the first code cell does:

1. Reads an SSH deploy key from Colab Secrets.
2. Writes it to `~/.ssh/id_ed25519`, `chmod 600`, trusts `github.com`.
3. `git clone git@github.com:AndriiUriadov/DysonianLineCNN.git` into `/content`.
4. `pip install -e ".[dev]"` editable install.
5. `drive.mount('/content/drive')`.

**One-time Colab setup** (per Colab account):

1. On your Mac, generate a dedicated SSH keypair for Colab (do not reuse
   your personal key):

   ```bash
   ssh-keygen -t ed25519 -f /tmp/colab_deploy_key -N "" -C "colab-deploy"
   ```

2. Add the **public** key (`/tmp/colab_deploy_key.pub`) to the GitHub repo
   as a **read-only** deploy key:

   ```text
   GitHub → DysonianLineCNN → Settings → Deploy keys → Add deploy key
   Title: Colab (read-only)
   Key: <paste contents of /tmp/colab_deploy_key.pub>
   Allow write access: UNCHECKED
   ```

3. Copy the **private** key contents into Colab Secrets:

   ```text
   Colab → any notebook → Secrets (key icon in left sidebar) → New secret
   Name: GITHUB_DEPLOY_KEY
   Value: <paste contents of /tmp/colab_deploy_key — the whole file>
   Notebook access: enabled
   ```

4. Delete the local copy so it never ends up in git or Drive:

   ```bash
   rm /tmp/colab_deploy_key /tmp/colab_deploy_key.pub
   ```

Read-only is important: it means even accidental `git push` from a Colab
notebook fails with a clear error, so writes stay local to your Mac.

After the first successful `git pull` in Colab, the notebooks are cached
in `/content/DysonianLineCNN` for the session.

## Reproducing results for a specific set

After completing the setup above, reproduce the full pipeline for any set
(e.g. `set-1`):

```bash
# 1. Classical baselines (requires MATLAB + EasySpin + Optimization Toolbox)
matlab -batch "addpath('matlab'); addpath('matlab/baseline'); FitAll('set-1')"

# 2. Generate synthetic training data (requires MATLAB + npy-matlab)
matlab -batch "addpath('matlab'); SetName='set-1'; DysonGeneratorMix"

# 3. Train CNN on Google Colab
#    Open notebooks/01_train_and_eval.ipynb, set SET_NAME = "set-1", run all cells

# 4. Update config/inference.json with the new runName and set_name

# 5. Prepare real spectra for CNN (requires MATLAB + EasySpin)
#    Loop PrepareOneSpectrumForCNN for each spectrum_basename in the set

# 6. CNN inference (Python)
python -c "
from dyson_cnn.infer import predict_for_set
for i in range(1, 7):  # adjust range for other sets
    predict_for_set('config', 'set-1', spectrum_basename=str(i))
"

# 7. Validate: overlay CNN reconstruction on experimental spectra
#    Loop Validator for each spectrum_basename in the set

# 8. Compile LaTeX report
cd latex && pdflatex report.tex && pdflatex report.tex
```

Per-set configuration files (`config/sets/set-N.json`) contain the
parameter ranges determined from the classical baseline fits. See
[Analysis.md](Analysis.md) for the rationale behind each range choice.

### Key configuration notes

- **BWindow_G** must match the intersection of all Bruker sweeps in
  the set. Use `eprload` to check B ranges before generation.
- **BOffsetRange_G** and **BScaleRange** should be `[0,0]` and `[1,1]`
  respectively — field calibration augmentation degrades B0 accuracy.
- All Dyson model formulas use the full two-term A/D coefficients from
  Holiatkina et al. (2023). Verify consistency if modifying any fitter.

### Further reading

- [config/README.md](config/README.md) — every tunable parameter documented

## License

This project is licensed under the [MIT License](LICENSE).

If you use this software in your research, please cite it:

> A.V. Uriadov, D.V. Savchenko, "DysonianLineCNN: 1D CNN for Dysonian EPR
> Line Parameter Extraction," 2026. https://github.com/AndriiUriadov/DysonianLineCNN

See [CITATION.cff](CITATION.cff) for the machine-readable citation format
(GitHub renders a "Cite this repository" button automatically).
