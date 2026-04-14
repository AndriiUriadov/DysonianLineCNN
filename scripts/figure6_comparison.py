"""
Figure 6. Three-method comparison on spectra of increasing complexity.

3x3 subplot grid plotted directly from raw data.
  Columns: MATLAB | EasySpin | CNN
  Rows:    (a) High-quality | (b) Moderate | (c) Challenging

Usage:
    python scripts/figure6_comparison.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
TEMP = REPO / "figures" / "temp"


# --- Dysonian physics (Holiatkina et al. 2023, Eq. 2-3) -------------------

def dyson_ad(p):
    p = max(p, 1e-9)
    ch, sh = np.cosh(p), np.sinh(p)
    c, s = np.cos(p), np.sin(p)
    den = ch + c
    d1, d2 = 2 * p * den, den ** 2
    A = (sh + s) / d1 + (1 + ch * c) / d2
    D = (sh - s) / d1 + (sh * s) / d2
    return A, D


def dyson_narrow(B, B0, dB, p):
    dB = max(dB, 1e-12)
    x = 2 * (B - B0) / (np.sqrt(3) * dB)
    A, D_ = dyson_ad(p)
    den = (1 + x ** 2) ** 2
    sig = (-A * 2 * x) / den + D_ * (1 - x ** 2) / den
    if np.argmin(sig) < np.argmax(sig):
        sig = -sig
    return sig


def scale_to_exp(sim, exp):
    A = np.column_stack([sim, np.ones_like(sim)])
    coef, *_ = np.linalg.lstsq(A, exp, rcond=None)
    return coef[0] * sim + coef[1]


# --- Spectra selection -----------------------------------------------------

SPECTRA = [
    # (set, id, row_label)          SNR  ΔB0_MC  ΔB0_EC  — rationale
    ("set-5", "13", "(a)", ""),   # 281    0.6     0.6   all three excellent
    ("set-3", "10", "(b)", ""),   #  56   14.5     2.6   MATLAB shifts, ES slight, CNN best
    ("set-3", "53", "(c)", ""),   #  17   13.3     4.8   MATLAB bad, ES deviates, CNN stable
]


def load_data(set_name, sid):
    # Try per-spectrum baxis first, then shared per-set
    b_file = TEMP / f"{set_name}_{sid}_baxis.csv"
    if not b_file.exists():
        b_file = TEMP / f"{set_name}_baxis.csv"
    if not b_file.exists():
        candidates = sorted(TEMP.glob(f"{set_name}_*_baxis.csv"))
        b_file = candidates[0] if candidates else b_file
    s_file = TEMP / f"{set_name}_{sid}_spectrum.csv"
    B = np.loadtxt(b_file, delimiter=",").flatten()
    spc = np.loadtxt(s_file, delimiter=",").flatten()
    mask = B > 0
    return B[mask], spc[mask]


def load_params(set_name, sid):
    cmp = RESULTS / set_name / "comparison.csv"
    with open(cmp) as f:
        for row in csv.DictReader(f):
            if row["spectrum_id"] == sid:
                return {
                    "matlab":   (float(row["B0_matlab"]),   float(row["dB_matlab"]),   float(row["p_matlab"])),
                    "easyspin": (float(row["B0_easyspin"]), float(row["dB_easyspin"]), float(row["p_easyspin"])),
                    "cnn":      (float(row["B0_cnn"]),      float(row["dB_cnn"]),      float(row["p_cnn"])),
                }
    return None


# --- Main ------------------------------------------------------------------

def main():
    fig, axes = plt.subplots(
        3, 3, figsize=(14, 10),
        sharex="row", sharey="row",
    )

    methods = ["matlab", "easyspin", "cnn"]
    method_titles = ["MATLAB (lsqnonlin)", "EasySpin (esfit)", "CNN (this work)"]

    for row_idx, (set_name, sid, row_label, desc) in enumerate(SPECTRA):
        B, spc = load_data(set_name, sid)
        params = load_params(set_name, sid)

        # Compute all fits
        fits = {}
        for m in methods:
            B0, dB, p = params[m]
            sim = dyson_narrow(B, B0, dB, p)
            fits[m] = scale_to_exp(sim, spc)

        for col_idx, m in enumerate(methods):
            ax = axes[row_idx, col_idx]
            B0, dB, p = params[m]

            ax.plot(B, spc, color="#2ca02c", lw=0.6, alpha=0.8,
                    label="Experimental" if row_idx == 0 and col_idx == 0 else None)
            ax.plot(B, fits[m], color="#d62728", lw=1.2,
                    label="Fit" if row_idx == 0 and col_idx == 0 else None)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.25, linewidth=0.5)

            # Parameter box
            ax.text(
                0.97, 0.95,
                f"$B_0$={B0:.1f} G\n$\\Delta B$={dB:.1f} G\n$p$={p:.3f}",
                transform=ax.transAxes, fontsize=8.5,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="gray", lw=0.5),
            )

            # Column header
            if row_idx == 0:
                ax.set_title(method_titles[col_idx], fontsize=13, fontweight="bold", pad=10)

            # Row label (left column only)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}  $dI/dB$ (a.u.)", fontsize=10)

            # x-axis label (bottom row only)
            if row_idx == 2:
                ax.set_xlabel("Magnetic field $B$ (G)", fontsize=10)

    # Single legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
               frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0.0, 0.03, 1.0, 1.0], h_pad=2.0, w_pad=1.0)

    out_dir = REPO / "figures"
    out_dir.mkdir(exist_ok=True)
    for fmt in ["png", "pdf"]:
        p = out_dir / f"figure6_comparison.{fmt}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {p}")
    plt.close()


if __name__ == "__main__":
    main()
