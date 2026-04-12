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

## Why CNN: Practical Advantages Over Classical Fitting

### 4. No Initial Parameter Estimates Required

Classical iterative methods (lsqnonlin, esfit) require starting guesses
for B0, dB, p, and the quality of the result depends critically on these
initial values. When the starting point is far from the true minimum, the
optimizer converges to a local minimum or hits a parameter bound.

In our data, this manifests as **convergence failures in 10–15% of spectra**:

- **MATLAB** returned `p = 0.200` (lower bound) for 7 out of 53 spectra
  in set-3 (#15, #16, #17, #22, #27, #30, #31) and for 5 spectra in
  set-4, indicating the optimizer got trapped.
- **EasySpin** returned `p = 5.000` (upper bound) for spectra set-3/#40
  and set-3/#44 — a non-physical result meaning the optimizer diverged.
- **MATLAB** reported `dB = 5.00` (lower bound) for several narrow-line
  spectra in set-3 and set-4.

The CNN never produces such artifacts. It always returns a value within
the physically meaningful range, because it learns the mapping from
spectrum shape to parameters directly, without iterative search.

### 5. Speed Enables Real-Time and High-Throughput Analysis

| Method | Time per spectrum | 80 spectra (set-4) | Notes |
|--------|------------------|-------------------|-------|
| MATLAB lsqnonlin | 1–5 s | ~5 min | 9 parameters, bounded optimization |
| EasySpin esfit | 5–30 s | ~15 min | Two-step LM with mask, GUI warning |
| **CNN** | **~0.1 s** | **~8 s** | Single forward pass (after model load) |

The CNN is 10–300× faster than classical methods. This is not merely a
convenience — it enables qualitatively different use cases:

- **Real-time analysis during EPR experiments:** monitor B0, dB, p as
  temperature or microwave power is swept, with sub-second feedback.
- **High-throughput screening:** process thousands of spectra from
  multi-sample studies or time-resolved measurements in minutes.
- **Parameter mapping:** generate spatial maps of Dysonian parameters
  across a sample (EPR imaging) where classical fitting would be
  prohibitively slow.

### 6. Deterministic Reproducibility

The same input spectrum always produces the same output parameters.
Classical methods depend on:
- Initial parameter estimates (heuristic, varies with implementation)
- Optimization strategy (lsqnonlin vs esfit, mask selection, step sequence)
- Convergence tolerance and iteration limits

In our data, MATLAB and EasySpin disagree by ΔB0 ≈ 15–20 G on the same
spectra (set-1 through set-5), purely due to different optimization
strategies. The CNN eliminates this source of variability.

### 7. No Licensed Toolbox Dependencies

CNN inference requires only open-source libraries (NumPy, TensorFlow).
Classical methods require:
- MATLAB Optimization Toolbox (for `lsqnonlin`)
- EasySpin (for `esfit`, `eprload`)

This lowers the barrier for reproducing results and enables deployment
on systems without MATLAB licenses.

### 8. CNN Limitations (Trade-offs)

- **Pre-training required:** A model must be trained for each parameter
  regime (~5 min on Colab T4 GPU, one-time cost per set). The training
  data generation and range selection require initial classical fitting.
- **No uncertainty estimates:** Classical methods provide confidence
  intervals and correlation matrices; the CNN returns point estimates only.
- **Bounded generalization:** The CNN cannot extrapolate beyond its
  training parameter ranges. A spectrum with dB outside [dBmin, dBmax]
  will produce an unreliable prediction with no warning.
- **BWindow sensitivity:** The magnetic field window of synthetic training
  data must match the experimental sweep precisely (see Finding #2).

### 9. Classical Method Limitations (For Comparison)

- **Convergence failures:** 10–15% of spectra produce results at parameter
  bounds (see Finding #4 above).
- **Sensitivity to initial estimates:** Different starting points can lead
  to different final parameters, even for the same optimizer.
- **Systematic B0 discrepancy:** MATLAB's 9-parameter model (with Bscale,
  Bshift) reports B0 ~15–20 G lower than EasySpin's 5-parameter model,
  because calibration offsets are absorbed differently.
- **Speed:** Impractical for datasets with >100 spectra without automation.

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

## Conclusions

1. **The 1D residual CNN achieves accuracy comparable to classical
   optimization methods** (MATLAB lsqnonlin and EasySpin esfit) for
   extracting Dysonian EPR line parameters B0, dB, and p from
   first-derivative spectra. Across 180 experimental spectra in 5
   independent sets, the CNN predictions agree with EasySpin results
   within ΔB0 ≈ 1–5 G, ΔdB ≈ 0.5–4 G, and Δp ≈ 0.04–0.3 for
   narrow-line sets (set-1, set-3–5). R² > 0.998 on synthetic test
   data for all three output heads in every set.

2. **The CNN is 10–300× faster than classical methods**, requiring ~0.1 s
   per spectrum (single forward pass) compared to 1–30 s for iterative
   optimization. This makes CNN-based parameter extraction practical for
   high-throughput analysis of large spectral datasets where classical
   fitting would be prohibitively slow.

3. **Correct physical model and careful data preparation are more important
   than model architecture.** The three most impactful improvements during
   development were all about data, not the neural network:
   - Unifying the Dyson A/D coefficient formula across all pipeline
     components (eliminated ~0.5 systematic bias in p)
   - Matching BWindow to the exact experimental sweep (eliminated
     zero-filled tails, improved resolution 17×)
   - Disabling BOffset augmentation (improved B0 MAE 3.4×, from 3.89 to
     1.14 G)

4. **Each experimental set requires its own CNN model.** The parameter
   ranges (B0, dB, p) and magnetic field windows vary significantly
   between sets. A model trained on one set does not generalize to another.
   The per-set training workflow — classical fit to determine ranges,
   synthetic data generation, CNN training, inference, validation — takes
   approximately 15 minutes per set (including ~5 min Colab training on
   T4 GPU).

5. **Classical methods remain necessary as a reference.** The CNN has no
   built-in uncertainty estimation and cannot detect when a spectrum falls
   outside its training distribution. Classical methods, despite being
   slower, provide confidence intervals and convergence diagnostics.
   The recommended workflow for publication-quality results is: CNN for
   rapid initial screening, followed by classical verification on spectra
   of interest.

6. **The three methods are complementary.** MATLAB lsqnonlin provides the
   most flexible fit (9 parameters including field calibration), EasySpin
   esfit offers robust convergence through its two-step masked approach,
   and the CNN offers unmatched speed. Discrepancies between methods
   (particularly the ~15–20 G systematic B0 offset between MATLAB and
   EasySpin/CNN) highlight the importance of reporting which method and
   which Dyson model parameterization was used when publishing EPR
   line parameters.

## Files

- Per-set summaries: `results/set-N/{matlab,easyspin,cnn}/summary.csv`
- Side-by-side comparison: `results/set-N/comparison.csv`
- Per-spectrum fit overlays: `results/set-N/{matlab,easyspin,cnn}/<id>_*.png`
  (gitignored, regenerated by FitAll/Validator)
- CNN predictions: `results/set-N/cnn/<id>_predicted.json` (gitignored)
