# Analysis: Three-Method Comparison of Dysonian EPR Line Parameter Extraction

## Overview

This analysis compares three methods for extracting Dysonian EPR line parameters
(B0, dB, p) from experimental first-derivative spectra recorded on Bruker
spectrometers:

1. **MATLAB optimizer** — `lsqnonlin` (9 parameters: B0, dB, p, A, offset,
   slope, quad, Bscale, Bshift) with `fminsearch` fallback
2. **EasySpin esfit** — Levenberg-Marquardt, two-step masked approach
   (step 1: fix p, fit B0/dB/A/offset; step 2: fix baseline, fit B0/dB/p/A)
3. **1D Residual CNN** — trained on synthetic Dysonian spectra, predicts
   B0/dB/p directly from the normalized spectrum shape

All three use the same Feher-Kip Dysonian model with full two-term A/D
coefficients (Holiatkina et al., J. Appl. Phys. 134, 145702, 2023).

Total: **180 experimental spectra** across 5 independent sets.

## Data Summary

| Set | Spectra | B0 range (G) | dB range (G) | p range | BWindow (G) | Notes |
|-----|---------|-------------|-------------|---------|-------------|-------|
| set-1 | 6 | 3331–3382 | 17–31 | 1.0–1.3 | 3257.85–3455.90 | Narrow lines, initial dev set |
| set-2 | 2 | 3434–3994 | 300–824 | 1.2–1.6 | 0–6998.29 | Wide lines, different sweeps |
| set-3 | 53 | 3347–3357 | 5–52 | 0.4–1.4 | 3256–3455.90 | Large set, wide dB range |
| set-4 | 80 | 3349–3361 | 5–86 | 0.1–1.3 | 3256–3455.90 | Largest set |
| set-5 | 39 | 3348–3355 | 5–31 | 0.1–1.3 | 3257.85–3457.75 | Overlaps with set-1 (5/6 shared) |

Note: set-1 spectra 1,2,4,5,6 are identical to set-5 spectra with the same
names (verified by MD5). Set-5 additionally contains 33 unique spectra.

## CNN Test Metrics (Synthetic Data)

Performance on held-out synthetic test split (15% of 10,000 generated spectra):

| Set | B0 MAE (G) | B0 RMSE (G) | dB MAE (G) | dB RMSE (G) | p3 MAE | p3 RMSE | R² (B0) |
|-----|-----------|------------|-----------|------------|--------|---------|---------|
| set-1 | 1.14 | 1.48 | 0.27 | 0.36 | 0.012 | 0.018 | 0.999 |
| set-2 | 5.37 | 6.80 | 5.14 | 6.74 | 0.012 | 0.017 | 0.999 |
| set-3 | 0.67 | 0.90 | 0.51 | 0.70 | 0.021 | 0.029 | 0.999 |
| set-4 | 0.56 | 0.74 | 0.77 | 1.00 | 0.022 | 0.030 | 0.999 |
| set-5 | 0.80 | 1.04 | 0.33 | 0.41 | 0.015 | 0.021 | 0.999 |

Set-2 has larger absolute MAE because the parameter ranges are much wider
(B0 range 800 G, dB range 700 G vs ~100 G for other sets). Relative accuracy
is comparable (~0.7% of range for all sets).

## Method Agreement on Real Spectra

### CNN vs EasySpin (mean absolute difference per set)

| Set | ΔB0 (G) | ΔdB (G) | Δp | Comment |
|-----|---------|---------|-----|---------|
| set-1 | 1.15 | 0.59 | 0.040 | Excellent agreement |
| set-2 | 54.9 | 77.9 | 0.088 | Wide lines, ill-conditioned fit |
| set-3 | 5.3 | 4.3 | 0.34 | Includes narrow lines where both methods struggle |
| set-4 | 2.1 | 4.0 | 0.23 | Good agreement on most spectra |
| set-5 | ~1–3 | ~1–2 | ~0.04 | Similar to set-1 (shared spectra) |

### CNN vs MATLAB (mean |ΔB0|)

MATLAB lsqnonlin systematically reports B0 ~15–25 G lower than CNN/EasySpin
across all narrow-line sets (set-1, set-3, set-4, set-5). This is because
MATLAB fits Bscale and Bshift calibration parameters that absorb field
offsets into the B0 estimate, while CNN and EasySpin both predict the
apparent B0 on the spectrometer axis.

## Key Findings

### 1. Formula Consistency is Critical

The original EasySpin fitter (`data/DysonFitEasyspin.m`) used a single-term
approximation for the Dyson A/D coefficients, while the MATLAB fitter and
the CNN training generator used the full two-term formula from Holiatkina
et al. (2023). This caused a ~0.5 systematic bias in fitted p values.
After unifying all components to use the full formula, the p discrepancy
between methods dropped to < 0.1.

**Paper reference:** Holiatkina, M. et al., J. Appl. Phys. 134, 145702 (2023)

Full formula (Eq. 3):
```
A = [sinh(p) + sin(p)] / [2p(cosh(p) + cos(p))] + [1 + cosh(p)cos(p)] / [(cosh(p) + cos(p))²]
D = [sinh(p) - sin(p)] / [2p(cosh(p) + cos(p))] + [sinh(p)sin(p)] / [(cosh(p) + cos(p))²]
```

### 2. BWindow Must Match Experimental Sweep

Setting `BWindow_G` to the exact intersection of all Bruker sweeps in a set
eliminates zero-filled tails that degrade CNN predictions. With BWindow
[0, 5000] (5000 G span, 4096 points), a line with dB=25 G occupies only
~20 points. Narrowing to the actual sweep (~200 G) gives ~340 points per
line — 17× better resolution.

### 3. BOffset Augmentation Hurts B0 Accuracy

The CNN training augmentation `BOffsetRange_G=[-5,5]` teaches the network
to undo field calibration offsets. This makes the predicted B0 differ from
the apparent resonance position by up to ~5 G, causing visible misalignment
in Validator overlays. Disabling this augmentation (`BOffsetRange=[0,0]`)
improved B0 MAE from 3.89 G to 1.14 G (3.4× improvement on set-1).

### 4. CNN Speed Advantage

| Method | Time per spectrum | Notes |
|--------|------------------|-------|
| MATLAB lsqnonlin | 1–5 s | 9 parameters, bounded optimization |
| EasySpin esfit | 5–30 s | Two-step LM with mask computation |
| **CNN** | **~0.1 s** | Single forward pass (after model load) |

The CNN is 10–300× faster than classical methods, making it suitable for
high-throughput screening of large spectral datasets.

### 5. CNN Limitations

- Requires a pre-trained model per parameter regime (separate model per set)
- Cannot handle spectra outside the training parameter ranges
- No uncertainty estimates (classical methods provide confidence intervals)
- B0 prediction is sensitive to BWindow alignment and augmentation choices

### 6. Classical Method Limitations

- MATLAB's 9-parameter fit can get trapped in local minima (several spectra
  in set-3 gave p=0.2 at the lower bound, indicating convergence failure)
- EasySpin's 2-step approach can miss the global optimum when p and B0 are
  strongly correlated
- Both methods produce unphysical results (p=5.0 upper bound) on some
  narrow-line spectra in set-3/set-4 where the signal-to-noise ratio is low

## Production Models

| Set | Run directory | Epochs (best) | Early stop |
|-----|--------------|---------------|------------|
| set-1 | `set-1/runs/20260412_122613` | 76 | 116 |
| set-2 | `set-2/runs/20260412_125230` | 39 | 79 |
| set-3 | `set-3/runs/20260412_135545` | 68 | 108 |
| set-4 | `set-4/runs/20260412_141449` | 128 | 168 |
| set-5 | `set-5/runs/20260412_143406` | 49 | 89 |

All models: 691,283 parameters, 3-channel input (z-score, ptp, B_axis),
colab_full profile (300 max epochs, batch=64, lr=1e-3, EarlyStopping
patience=40).

## Files

- Per-set summaries: `results/set-N/{matlab,easyspin,cnn}/summary.csv`
- Side-by-side comparison: `results/set-N/comparison.csv`
- Per-spectrum fit overlays: `results/set-N/{matlab,easyspin,cnn}/<id>_*.png`
  (gitignored, regenerated by FitAll/Validator)
- CNN predictions: `results/set-N/cnn/<id>_predicted.json` (gitignored)
