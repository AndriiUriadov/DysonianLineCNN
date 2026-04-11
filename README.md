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
- [MigrationPlan.md](MigrationPlan.md) — the full migration design
- [redacted.md](redacted.md) — context and critical invariants
