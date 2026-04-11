# DysonianLineCNN

A hybrid MATLAB + Python pipeline for regressing Dysonian EPR line parameters
(`B0`, `dB`, `p3`) from first-derivative EPR spectra using a 1D CNN, and for
applying the trained model to real experimental Bruker spectra (`.DTA` / `.DSC`).

**Authors:** A.V. Uriadov, D.V. Savchenko — National Technical University of
Ukraine "Igor Sikorsky Kyiv Polytechnic Institute".

## Pipeline overview

1. **Synthetic dataset generation (MATLAB)** —
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
| `parity_test.png` | 150 KB | Three-panel true-vs-predicted scatter (`B0`, `dB`, `p3`) on the test set, with a `y=x` diagonal |
| `dysonian_test_predictions.csv` | 110 KB | Per-sample `(B0, dB, p3)` true and predicted values in physical units for ad-hoc analysis |
| `_X_test.npy` / `_y_test.npy` | 23 MB / 18 KB | Cached test split so `evaluate_run` can be replayed in a separate Jupyter session without re-splitting the dataset |

The leading underscore on `_X_test.npy` / `_y_test.npy` marks them as
internal implementation files — they let you call `evaluate_run(run_dir)`
weeks later from a different notebook session without loading the full
10 000-sample dataset back into memory. Safe to delete if you need the
disk space back; `evaluate_run` will just fail with a clear error.

The current production run referenced by
[config/inference.json](config/inference.json) is **`20260411_204438`**
(`colab_full` profile, early-stopped at epoch 110 with best weights
restored from epoch 70). Test metrics on 1500 held-out samples:

| Parameter | Range | MAE | RMSE | Relative error |
| --- | --- | --- | --- | --- |
| `B0` | 3400–4500 G | 14.88 G | 19.02 G | 1.35 % |
| `dB` | 250–350 G | 0.80 G | 0.98 G | 0.80 % |
| `p3` | 1.30–1.50 | 0.0022 | 0.0028 | 1.10 % |

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

42 tests cover config loading, normalization invariants, channel order,
model architecture, and inference. All should pass in a few seconds.

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

### Further reading

- [config/README.md](config/README.md) — every tunable parameter documented
- [Documentation/](Documentation/) — thesis PDFs, BibTeX, architecture diagram
