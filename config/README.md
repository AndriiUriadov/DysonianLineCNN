# config/

All tunable parameters of the DysonianLineCNN pipeline live here in four JSON
files. Both MATLAB (`jsondecode`) and Python (`json.load`) read these files,
so there is a single source of truth for paths, dataset generator inputs,
training hyperparameters, and inference targets. Code in `matlab/` and
`dyson_cnn/` must not contain hardcoded duplicates of anything listed below.

## Files

| File | Tracked by git? | Read by |
|---|---|---|
| [paths.example.json](paths.example.json) | yes (template) | nobody directly — users copy it to `paths.json` |
| [paths.json](paths.json) | **no** (gitignored, user-specific) | MATLAB + Python |
| [dataset.json](dataset.json) | yes | MATLAB (`DysonGeneratorMix.m`) and Python (prefix only) |
| [training.json](training.json) | yes | Python (`dyson_cnn.train`) |
| [inference.json](inference.json) | yes | Python (`dyson_cnn.infer`) and MATLAB (`Validator.m`, `PrepareOneSpectrumForCNN.m`) |

## First-time setup

```bash
cp config/paths.example.json config/paths.json
$EDITOR config/paths.json
```

Edit `drive_root_mac` to match your Google Drive for Desktop mount location.
On macOS Sonoma and later, it is of the form
`/Users/<username>/Library/CloudStorage/GoogleDrive-<email>/My Drive`. Note
that if your Google Drive for Desktop is localized (for example, Ukrainian),
the last path segment may be `Мій диск` instead of `My Drive`.

`paths.json` is gitignored — your local values never end up in the public
history.

## paths.json — paths on your machine

| Field | Meaning |
|---|---|
| `drive_root_mac` | Absolute path to the root of your Google Drive mount on macOS. Used by MATLAB and by Python when running on Mac. |
| `drive_root_colab` | Absolute path to `MyDrive` in Google Colab after `drive.mount(...)`. Constant across Colab sessions: `/content/drive/MyDrive`. |
| `project_subdir` | Path to the project folder inside Drive, relative to the Drive root. Default: `Python/DysonianLineCNN`. |
| `runs_subdir` | Subfolder inside the project folder where training runs are written. Each run is a timestamped directory. |
| `data_subdir` | Subfolder where raw Bruker spectra (`.DTA`/`.DSC`) live. Matches the repo's top-level `data/` folder by convention. |

Python-side code resolves the drive root by detecting the runtime: if
`/content` exists it picks `drive_root_colab`, otherwise `drive_root_mac`.
MATLAB always uses `drive_root_mac`.

## dataset.json — inputs to the MATLAB generator

Field names match `matlab/DysonGeneratorMix.m`'s `inputParser` parameters
exactly, so MATLAB can `jsondecode` the file and feed the result as defaults
into the parser with no renaming.

### Core

| Field | Meaning |
|---|---|
| `N` | Number of synthetic spectra to generate. |
| `Npoints` | Number of points on the visible (B > 0) magnetic field axis. **Must stay 4096** — the CNN architecture, real-spectrum preparation, and all saved models assume this length. |
| `Seed` | RNG seed passed to `rng()` at the start of generation. Keep fixed for reproducibility. |
| `Prefix` | Filename prefix for outputs: `X_dyson_mix_<Prefix>.npy`, `y_dyson_mix_<Prefix>.npy`, etc. Also referenced by `inference.json.dataset_prefix` so Python knows which dataset a trained model was fit on. |

### Physics ranges (all in Gauss)

| Field | Meaning |
|---|---|
| `B0Range_G` | `[min, max]` range of the resonance field `B0`. |
| `dBRange_G` | `[min, max]` range of the peak-to-peak linewidth `dB` (also written `ΔB_pp`). |
| `pRange` | `[min, max]` range of the 3rd label `p3`. **Mode-dependent**: in narrow mode (`dBRange_G[1] < dB_thr_G`) this is the physical Dyson skin-effect parameter `p`; in wide mode (`dBRange_G[0] > dB_thr_G`) this is the Joshi asymmetry parameter `alpha`, which must be non-negative. |
| `BWindow_G` | `[Bmin, Bmax]` visible magnetic field window. Only `B > 0` is used. |
| `dB_thr_G` | Threshold separating narrow and wide Dysonian regimes. Default 1000 G = 100 mT. The generator refuses `dBRange_G` that straddles this threshold. |

### Augmentation (on top of pure physics)

All augmentations are applied only if `DoAugment` is `true`.

| Field | Meaning |
|---|---|
| `SNRdBRange` | Signal-to-noise ratio in dB for additive Gaussian noise. |
| `BaselineOffsetRange` | Constant baseline offset (relative units). |
| `BaselineSlopeRange` | Linear baseline slope across `BWindow_G`. |
| `BaselineQuadRange` | Quadratic baseline curvature. |
| `BScaleRange` | Global magnetic field scale factor (simulates calibration errors). |
| `BOffsetRange_G` | Global magnetic field offset in Gauss. |
| `SpikeProb` | Per-spectrum probability of injecting spike artifacts. |
| `SpikeCountRange` | `[min, max]` number of spikes per affected spectrum. |
| `SpikeAmpRange` | `[min, max]` relative amplitude of spikes (fraction of signal scale). |
| `ModSmoothProb` | Per-spectrum probability of modulation-induced Gaussian smoothing (simulates lock-in detection). |
| `ModSmoothSigmaPts` | `[min, max]` Gaussian smoothing width in units of data points. |

### Debug

| Field | Meaning |
|---|---|
| `DoVisualCheck` | If `true`, MATLAB plots `Nvisual` random spectra at the end of generation for sanity checking. |
| `Nvisual` | Number of spectra plotted when `DoVisualCheck` is enabled. |

## training.json — CNN hyperparameters

Profiles are fully self-contained dictionaries under `profiles.<name>`. Select
one via:

1. Setting `active_profile` at the top of the JSON file. This is the default.
2. Setting the `DYSON_PROFILE` environment variable before running Python,
   which overrides `active_profile` for that process.

Three profiles are shipped:

| Profile | Where | Dataset size | Epochs | `mixed_precision` | Purpose |
|---|---|---|---|---|---|
| `colab_full` | Google Colab (NVIDIA GPU) | full | 300 | `true` | Final training runs for the article |
| `mac_full` | Mac Apple Silicon (`tensorflow-metal`) | full | 300 | `false` | Full training locally when Colab is unavailable |
| `mac_dev` | Mac | 500 (sliced) | 10 | `false` | Fast dev iterations for debugging the pipeline |

`mixed_precision: true` is only honored on NVIDIA GPUs. `dyson_cnn.train`
detects Apple Silicon at runtime and forces `float32` even when a profile
requests mixed precision, so `colab_full` and `mac_full` can be used
interchangeably without editing the JSON.

### Per-profile fields

| Field | Meaning |
|---|---|
| `epochs` | Upper bound on training epochs. Early stopping usually terminates earlier. |
| `batch_size` | Mini-batch size. Lower on Mac because Metal memory pressure differs from CUDA. |
| `dropout` | Dropout rate applied once before the output heads. |
| `learning_rate` | Initial Adam learning rate. `ReduceLROnPlateau` scales it down during training. |
| `clipnorm` | Global gradient norm clipping for Adam. |
| `mixed_precision` | Request `mixed_float16` policy. Ignored on Apple Silicon (forced to `float32`). |
| `loss_weights` | Per-head MSE loss weights. `p3` is weighted `2.0` because it is harder to learn than `B0`/`dB` — underweighting it makes training collapse onto the easy heads. |
| `early_stopping_patience` | Epochs with no `val_loss` improvement before early stop. |
| `reduce_lr_patience` | Epochs with no `val_loss` improvement before LR is halved. |
| `reduce_lr_factor` | Multiplicative factor applied to LR when `ReduceLROnPlateau` fires. |
| `reduce_lr_min` | Lower bound on learning rate. |
| `max_samples` | If not `null`, the loaded dataset is sliced to `X[:max_samples]` before splitting. Used by `mac_dev` to shortcut the pipeline for debugging. |
| `random_seed` | Seed passed to `tf.keras.utils.set_random_seed` and the `train_test_split` call. Distinct from `dataset.json.Seed` (which is the MATLAB generator seed). |

## inference.json — which run, which spectrum

| Field | Meaning |
|---|---|
| `runName` | Timestamped directory name inside `<project>/runs/` that holds the trained model to use for inference. Set this to `TO_BE_SET_AFTER_FIRST_TRAINING` on a fresh checkout; update after your first successful training run. To find a valid value, list your Drive runs folder after training and pick the newest timestamp directory. |
| `spectrum_basename` | Basename (without `.DTA`/`.DSC` extension) of the real Bruker spectrum to predict on. The file must live under `data/` on Drive. |
| `dataset_prefix` | Prefix of the synthetic dataset the model was trained on. Must match `dataset.json.Prefix` at training time. Used so `Validator.m` can find the right `meta_mix_<prefix>.json` for physical-mode reconstruction (narrow vs wide Dyson). |

## Why four files instead of one

Different lifecycles. `paths.json` varies per-user and never gets committed.
`dataset.json` changes rarely (you regenerate datasets maybe once a month).
`training.json` is tuned every few runs. `inference.json` changes several
times per session as you validate the model against different spectra.
Keeping them separate means you do not churn unrelated parameters when you
only wanted to change one thing, and git diffs stay narrow.
