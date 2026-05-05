# sim_to_real.py
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Sim-to-Real Noise Module for Time-of-Flight / SPAD Sensors        ║
║                                                                              ║
║  THEORY:                                                                     ║
║  Real SPAD (Single-Photon Avalanche Diode) sensors do not produce perfect   ║
║  transient renders. Three physical phenomena corrupt the ideal signal:       ║
║                                                                              ║
║  1. IRF (Instrument Response Function)                                       ║
║     The laser pulse is NOT a perfect delta function. It has finite width,    ║
║     asymmetry (skew), and the detector has timing jitter. The measured       ║
║     signal is the TRUE scene response convolved with this pulse shape:       ║
║         measured(t) = true(t) ⊛ irf(t)                                      ║
║                                                                              ║
║  2. Shot Noise (Photon noise)                                                ║
║     Light is quantized into photons. In any time bin, the number of         ║
║     detected photons follows a Poisson distribution:                         ║
║         k ~ Poisson(λ)   where λ = expected_photon_count                    ║
║     This means SNR = √λ — weak signals are disproportionately noisy.        ║
║                                                                              ║
║  3. Dark Count Rate (DCR)                                                    ║
║     Thermal electrons in the detector spontaneously trigger the SPAD,       ║
║     producing false counts even in complete darkness. These are uniformly   ║
║     distributed across all time bins:                                        ║
║         dcr_counts_per_bin = DCR_cps × bin_width_seconds                    ║
║                                                                              ║
║  PIPELINE:                                                                   ║
║  clean render → IRF convolution → Shot noise → DCR → SPAD output            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy import signal
from scipy.stats import skewnorm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as patches

from matplotlib.colors import LogNorm
import os
import cv2

# ── global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0f0f1a',
    'axes.facecolor':    '#1a1a2e',
    'axes.edgecolor':    '#444466',
    'axes.labelcolor':   '#ccccee',
    'axes.titlesize':    11,
    'axes.labelsize':    9,
    'xtick.color':       '#aaaacc',
    'ytick.color':       '#aaaacc',
    'text.color':        '#ccccee',
    'grid.color':        '#2a2a4a',
    'grid.linewidth':    0.6,
    'legend.facecolor':  '#1a1a2e',
    'legend.edgecolor':  '#444466',
    'legend.fontsize':   8,
    'figure.dpi':        150,
})

PALETTE = {
    'clean':    '#4fc3f7',   # light blue
    'irf':      '#ffb74d',   # amber
    'shot':     '#81c784',   # green
    'dcr':      '#ce93d8',   # lavender
    'noisy':    '#ef5350',   # red
    'gaussian': '#4fc3f7',
    'skewed':   '#ffb74d',
    'dexp':     '#81c784',
}

os.makedirs("output_spad", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. IRF  ─  Instrument Response Function
# ══════════════════════════════════════════════════════════════════════════════

class IRF:
    """
    The IRF encodes EVERYTHING that temporally smears the ideal signal:
      • Laser pulse width  (finite duration of the emitted pulse)
      • Detector jitter    (timing uncertainty in the SPAD avalanche trigger)
      • Electronics jitter (TDC quantisation, cable delays)

    All three effects combine into a single convolution kernel. In calibration,
    one illuminates a perfect mirror and measures the return — that measured
    waveform IS the IRF.
    """

    @staticmethod
    def gaussian(num_bins: int, sigma_bins: float) -> np.ndarray:
        """
        Symmetric Gaussian — simplest model, good for narrow-linewidth lasers
        with low detector jitter. Rarely seen in practice due to asymmetry.

        f(t) = exp(−t² / 2σ²)
        """
        t = np.arange(num_bins) - num_bins // 2
        pulse = np.exp(-0.5 * (t / max(sigma_bins, 1e-6)) ** 2)
        return pulse / pulse.sum()

    @staticmethod
    def skewed_gaussian(num_bins: int,
                        sigma_bins: float,
                        skewness: float = 3.0) -> np.ndarray:
        """
        Skewed Gaussian — the most common real-world IRF shape.

        Physical origin: The laser pulse has a fast rise time but a slow
        exponential decay (from gain switching). The SPAD also has a slightly
        asymmetric timing response. Result: right-skewed pulse.

        skewness > 0 → trailing tail (photons appear to arrive late)
        skewness < 0 → leading tail  (rare, seen in some pulsed diode lasers)
        """
        t = np.arange(num_bins) - num_bins // 2
        pulse = skewnorm.pdf(t, a=skewness, loc=0, scale=sigma_bins)
        if pulse.sum() < 1e-12:
            pulse = np.zeros(num_bins)
            pulse[num_bins // 2] = 1.0
        return pulse / pulse.sum()

    @staticmethod
    def double_exponential(num_bins: int,
                           rise_bins: float = 2.0,
                           fall_bins: float = 8.0) -> np.ndarray:
        """
        Double-exponential (RC model) IRF.

        Physical origin: Models the detector + TDC electronics as an RC
        circuit. The pulse charges quickly (rise_bins) and discharges slowly
        (fall_bins). This is also a good model for afterpulsing tails in
        high-count-rate scenarios.

        f(t) = (1 − exp(−t/τ_rise)) × exp(−t/τ_fall),  t ≥ 0
        """
        t = np.arange(num_bins, dtype=float)
        c = num_bins // 4
        ts = t - c
        pulse = np.where(
            ts >= 0,
            (1 - np.exp(-ts / max(rise_bins,  1e-6))) *
             np.exp(-ts / max(fall_bins, 1e-6)),
            0.0
        )
        if pulse.sum() < 1e-12:
            pulse[c] = 1.0
        return pulse / pulse.sum()

    @staticmethod
    def convolve(transient: np.ndarray,
                 irf_kernel: np.ndarray) -> np.ndarray:
        """
        Apply IRF along the time axis (axis=2) via FFT convolution.
        mode='same' preserves the number of time bins.
        """
        H, W, T, C = transient.shape
        out = np.zeros_like(transient)
        for c in range(C):
            for h in range(H):
                row  = transient[h, :, :, c]          # (W, T)
                conv = np.array([
                    signal.fftconvolve(row[w], irf_kernel, mode='same')
                    for w in range(W)
                ])
                out[h, :, :, c] = conv
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. SHOT NOISE  ─  Poisson Photon Statistics
# ══════════════════════════════════════════════════════════════════════════════

class ShotNoise:
    """
    THEORY:
    Light is made of discrete photons. In a time bin of width Δt, if the
    expected number of photons is λ, the ACTUAL count follows:

        k ~ Poisson(λ)
        E[k]   = λ
        Var[k] = λ          ← variance equals the mean!
        SNR    = λ / √λ = √λ

    This means:
      • Bright signals (large λ) → high SNR, small relative noise
      • Weak signals  (small λ) → low SNR, dominant noise
      • SNR improves as √(integration_time) — doubling time → √2 better SNR

    The photons_per_unit parameter converts mitransient radiance units
    (arbitrary) into physical photon counts. It encodes:
      • Laser power
      • Collection aperture
      • Detector quantum efficiency (QE)
      • Integration / averaging time
    """

    def __init__(self, photons_per_unit: float = 1000.0):
        self.photons_per_unit = photons_per_unit

    def apply(self, transient: np.ndarray,
              return_counts: bool = False) -> np.ndarray:
        expected = np.clip(transient, 0, None) * self.photons_per_unit
        counts   = np.random.poisson(expected).astype(np.float64)
        if return_counts:
            return counts
        return counts / self.photons_per_unit


# ══════════════════════════════════════════════════════════════════════════════
# 3. DARK COUNT RATE  ─  Thermal Background
# ══════════════════════════════════════════════════════════════════════════════

class DCR:
    """
    THEORY:
    SPADs can fire spontaneously due to thermally generated electron-hole pairs.
    These 'dark counts' are:
      • Completely INDEPENDENT of the optical signal
      • Uniformly distributed across all time bins
      • Poisson distributed with rate = DCR_cps × bin_width_seconds
      • Temperature-dependent (cooling the sensor dramatically reduces DCR)

    Typical values:
      • Room-temperature SPAD:   100 – 100,000 cps
      • Cooled SPAD (-20°C):     10  –  1,000 cps
      • Best research SPADs:     < 10 cps

    At long ranges, the signal is weak and DCR becomes the dominant
    noise source — it sets the fundamental sensitivity limit.
    """

    def __init__(self,
                 dcr_cps: float          = 1000.0,
                 bin_width_seconds: float = 67e-12):
        self.dark_counts_per_bin = dcr_cps * bin_width_seconds
        self.dcr_cps             = dcr_cps
        self.bin_width_seconds   = bin_width_seconds

    def apply(self, transient: np.ndarray,
              photons_per_unit: float = 1000.0) -> np.ndarray:
        dark = np.random.poisson(
            self.dark_counts_per_bin,
            size=transient.shape
        ).astype(np.float64)
        return transient + dark / photons_per_unit


# ══════════════════════════════════════════════════════════════════════════════
# 4. FULL SPAD SENSOR MODEL
# ══════════════════════════════════════════════════════════════════════════════

class SPADSensorModel:
    """
    Composable physical sensor model:

        clean render  ──►  IRF  ──►  Shot noise  ──►  DCR  ──►  SPAD output
    """

    def __init__(self,
                 irf_type        : str   = 'skewed_gaussian',
                 irf_sigma_bins  : float = 2.0,
                 irf_skewness    : float = 3.0,
                 irf_rise_bins   : float = 2.0,
                 irf_fall_bins   : float = 8.0,
                 irf_kernel_size : int   = 31,
                 photons_per_unit: float = 1000.0,
                 dcr_cps         : float = 1000.0,
                 bin_width_opl   : float = 0.02,
                 seed            : int   = 42):

        np.random.seed(seed)
        self.photons_per_unit = photons_per_unit
        self.irf_type         = irf_type
        self.irf_sigma_bins   = irf_sigma_bins
        self.irf_skewness     = irf_skewness
        self.dcr_cps          = dcr_cps
        self.bin_width_opl    = bin_width_opl

        # build IRF kernel
        if irf_type == 'gaussian':
            self.irf_kernel = IRF.gaussian(irf_kernel_size, irf_sigma_bins)
        elif irf_type == 'skewed_gaussian':
            self.irf_kernel = IRF.skewed_gaussian(
                irf_kernel_size, irf_sigma_bins, irf_skewness)
        elif irf_type == 'double_exponential':
            self.irf_kernel = IRF.double_exponential(
                irf_kernel_size, irf_rise_bins, irf_fall_bins)
        else:
            raise ValueError(f"Unknown irf_type: {irf_type}")

        c                  = 299_792_458.0
        bin_width_seconds  = bin_width_opl / c
        self.shot_noise    = ShotNoise(photons_per_unit)
        self.dcr           = DCR(dcr_cps, bin_width_seconds)

    def apply(self, transient   : np.ndarray,
              apply_irf  : bool = True,
              apply_shot : bool = True,
              apply_dcr  : bool = True) -> np.ndarray:
        r = transient.copy().astype(np.float64)
        if apply_irf:
            r = IRF.convolve(r, self.irf_kernel)
        if apply_shot:
            r = self.shot_noise.apply(r)
        if apply_dcr:
            r = self.dcr.apply(r, self.photons_per_unit)
        return r

    def get_photon_counts(self, transient: np.ndarray) -> np.ndarray:
        r = self.apply(transient)
        return np.round(r * self.photons_per_unit).astype(np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# 5. METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(clean: np.ndarray, noisy: np.ndarray,
                    photons_per_unit: float) -> dict:
    signal_power = np.mean(clean ** 2)
    noise_power  = np.mean((noisy - clean) ** 2)
    snr_db       = 10 * np.log10(signal_power / (noise_power + 1e-12))

    # PSNR over active bins only
    active       = clean > clean.max() * 0.01
    mse          = np.mean((noisy[active] - clean[active]) ** 2)
    psnr         = 10 * np.log10((clean.max() ** 2) / (mse + 1e-12))

    # Mean photon count in bright bins
    bright_bins  = clean[active] * photons_per_unit
    mean_photons = bright_bins.mean() if active.any() else 0.0

    return dict(
        snr_db           = snr_db,
        psnr_db          = psnr,
        peak_clean       = float(clean.max()),
        peak_noisy       = float(noisy.max()),
        peak_ratio       = float(noisy.max() / (clean.max() + 1e-12)),
        noise_std        = float((noisy - clean).std()),
        noise_mean       = float((noisy - clean).mean()),
        mean_bright_phot = float(mean_photons),
        expected_snr_sqrt= float(np.sqrt(mean_photons)),  # theoretical √N
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _title_box(fig, title: str, subtitle: str = ''):
    """Add a centered suptitle with optional subtitle."""
    full = title if not subtitle else f"{title}\n{subtitle}"
    fig.suptitle(full, fontsize=13, fontweight='bold',
                 color='white', y=0.98)


# ── 6-A: Pipeline diagram ────────────────────────────────────────────────────
def plot_pipeline_diagram(save='output_spad/00_pipeline.png'):
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis('off')

    stages = [
        (1.0,  'mitransient\nclean render', '#1565c0', '(H,W,T,C)\nfloat64'),
        (3.5,  'IRF\nconvolution',          '#e65100', 'pulse smearing\n⊛ kernel'),
        (6.0,  'Shot Noise\n(Poisson)',      '#2e7d32', 'k ~ Poisson(λ)\nλ=I·scale'),
        (8.5,  'Dark Count\nRate (DCR)',     '#6a1b9a', 'uniform floor\ncps × Δt'),
        (11.0, 'SPAD\noutput',              '#b71c1c', 'photon counts\nint64'),
    ]

    for x, label, color, sub in stages:
        ax.add_patch(patches.FancyBboxPatch(
            (x - 0.85, 0.6), 1.7, 1.8,
            boxstyle='round,pad=0.1',
            facecolor=color, edgecolor='white',
            linewidth=1.5, alpha=0.85, zorder=2))
        ax.text(x, 1.75, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white', zorder=3)
        ax.text(x, 0.9,  sub,   ha='center', va='center',
                fontsize=7, color='#dddddd',  zorder=3)

    # arrows
    for x in [1.85, 4.35, 6.85, 9.35]:
        ax.annotate('', xy=(x + 0.3, 1.5), xytext=(x, 1.5),
                    arrowprops=dict(arrowstyle='->', color='#aaaacc',
                                   lw=2.0))

    ax.set_facecolor('#0f0f1a')
    fig.patch.set_facecolor('#0f0f1a')
    _title_box(fig, 'Sim-to-Real SPAD Noise Pipeline',
               'Physical degradations applied sequentially to the ideal render')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save}")


# ── 6-B: IRF kernel shapes ───────────────────────────────────────────────────
def plot_irf_kernels(save='output_spad/01_irf_kernels.png'):
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    kernels = {
        'Gaussian\n(σ=2 bins)':
            ('gaussian',  PALETTE['gaussian'],
             IRF.gaussian(127, 2.0)),
        'Skewed Gaussian\n(σ=2, skew=3)':
            ('skewed',    PALETTE['skewed'],
             IRF.skewed_gaussian(127, 2.0, 3.0)),
        'Double Exponential\n(rise=2, fall=8)':
            ('dexp',      PALETTE['dexp'],
             IRF.double_exponential(127, 2.0, 8.0)),
    }

    t = np.arange(127) - 63
    theory_texts = {
        'gaussian':
            "f(t) = exp(−t²/2σ²)\n\n"
            "• Symmetric around peak\n"
            "• Good for narrow-linewidth lasers\n"
            "• Rarely seen in practice\n"
            "• Sets the diffraction limit\n"
            "  of timing resolution",
        'skewed':
            "f(t) = 2Φ(αt/σ)·φ(t/σ)\n\n"
            "• Fast rise, slow trailing edge\n"
            "• Most realistic SPAD IRF\n"
            "• Tail from afterpulsing\n"
            "• α controls asymmetry\n"
            "• Calibrated from mirror return",
        'dexp':
            "f(t) = (1−e^{−t/τ_r})·e^{−t/τ_f}\n\n"
            "• Models RC circuit response\n"
            "• τ_r: detector rise time\n"
            "• τ_f: electronics decay\n"
            "• Common in TCSPC systems\n"
            "• Good for afterpulsing",
    }

    for col, (label, (key, color, kern)) in enumerate(kernels.items()):
        # top row: kernel shape
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t, kern, color=color, linewidth=2.5, zorder=3)
        ax.fill_between(t, kern, alpha=0.25, color=color)

        # FWHM annotation
        half = kern.max() / 2
        fwhm = np.sum(kern > half)
        above = np.where(kern > half)[0]
        if len(above) > 1:
            ax.axhspan(0, half, alpha=0.08, color=color)
            ax.axvline(t[above[0]],  color=color, linewidth=0.8,
                       linestyle='--', alpha=0.6)
            ax.axvline(t[above[-1]], color=color, linewidth=0.8,
                       linestyle='--', alpha=0.6)
            ax.text(0, half * 1.1,
                    f'FWHM ≈ {fwhm} bins',
                    ha='center', fontsize=7.5, color=color)

        ax.set_title(label, color='white', fontsize=10)
        ax.set_xlabel('Bins from center')
        ax.set_ylabel('Normalized amplitude')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-40, 60)

        # bottom row: theory text
        ax2 = fig.add_subplot(gs[1, col])
        ax2.axis('off')
        ax2.text(0.5, 0.95, theory_texts[key],
                 transform=ax2.transAxes,
                 ha='center', va='top',
                 fontsize=8.5, color='#ccccee',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.6',
                           facecolor='#1a1a2e',
                           edgecolor=color, linewidth=1.5))

    _title_box(fig, 'IRF Kernel Shapes — Physical Models',
               'Each kernel represents a different laser + detector system')
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save}")


# ── 6-C: Noise theory visual ─────────────────────────────────────────────────
def plot_noise_theory(save='output_spad/02_noise_theory.png'):
    """
    Visual explanation of Poisson shot noise and DCR.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Poisson distributions for different λ ---
    ax = axes[0]
    lambdas = [1, 5, 20, 100]
    colors  = ['#ef5350', '#ffb74d', '#81c784', '#4fc3f7']
    k_max   = 140
    k       = np.arange(k_max)
    for lam, col in zip(lambdas, colors):
        from scipy.stats import poisson
        pmf = poisson.pmf(k, lam)
        ax.plot(k[:int(lam*3)+10],
                pmf[:int(lam*3)+10],
                color=col, linewidth=2,
                label=f'λ={lam}  SNR={np.sqrt(lam):.1f}')
        ax.fill_between(k[:int(lam*3)+10],
                        pmf[:int(lam*3)+10],
                        alpha=0.15, color=col)
    ax.set_title('Shot Noise: Poisson Distributions\nk ~ Poisson(λ),  SNR = √λ')
    ax.set_xlabel('Photon count k')
    ax.set_ylabel('P(k | λ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, k_max)

    # --- Panel 2: SNR vs photon count curve ---
    ax = axes[1]
    lam_range = np.logspace(0, 5, 500)
    snr_range = np.sqrt(lam_range)
    ax.loglog(lam_range, snr_range,
              color=PALETTE['shot'], linewidth=2.5)
    ax.fill_between(lam_range, 1, snr_range, alpha=0.15,
                    color=PALETTE['shot'])

    # annotate regimes
    for lam, label, col in [
        (2,    'Very noisy\n(few photons)',   '#ef5350'),
        (50,   'Moderate SNR\n(usable)',      '#ffb74d'),
        (5000, 'High SNR\n(excellent)',       '#81c784'),
    ]:
        snr = np.sqrt(lam)
        ax.scatter([lam], [snr], color=col, s=80, zorder=5)
        ax.text(lam * 1.3, snr * 0.7, label, fontsize=7.5,
                color=col, va='top')

    ax.set_xlabel('Expected photon count λ')
    ax.set_ylabel('SNR = √λ')
    ax.set_title('SNR vs Photon Count\n(log-log, fundamental quantum limit)')
    ax.grid(True, alpha=0.3, which='both')
    ax.axhline(1, color='#ef5350', linestyle='--',
               linewidth=1, alpha=0.5, label='SNR=1 (noise floor)')
    ax.legend()

    # --- Panel 3: DCR illustration ---
    ax = axes[2]
    T_bins = 200
    bins   = np.arange(T_bins)

    # Gaussian signal peak
    true_signal = 5.0 * np.exp(-0.5 * ((bins - 80) / 5) ** 2)
    dcr_level   = 0.3 * np.ones(T_bins)
    combined    = true_signal + dcr_level + np.random.poisson(
        true_signal * 10 + 3, T_bins) / 10 - (true_signal + 0.3)

    ax.fill_between(bins, 0, true_signal,
                    color=PALETTE['clean'], alpha=0.5,
                    label='True signal')
    ax.fill_between(bins, 0, dcr_level,
                    color=PALETTE['dcr'], alpha=0.4,
                    label=f'DCR floor\n(uniform, all bins)')
    ax.plot(bins, true_signal + dcr_level + np.random.normal(0, 0.1, T_bins),
            color=PALETTE['noisy'], linewidth=0.8, alpha=0.7,
            label='Measured (with noise)')

    ax.axhline(dcr_level[0], color=PALETTE['dcr'],
               linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(T_bins * 0.6, dcr_level[0] + 0.05,
            'DCR floor', color=PALETTE['dcr'], fontsize=8)
    ax.set_title('Dark Count Rate (DCR)\nUniform thermal background noise')
    ax.set_xlabel('Time bin')
    ax.set_ylabel('Counts / radiance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    _title_box(fig, 'Noise Theory — Shot Noise & DCR',
               'Fundamental physical limits of SPAD detectors')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save}")


# ── 6-D: Single pixel profile ────────────────────────────────────────────────
def plot_pixel_profile(clean: np.ndarray,
                       noisy: np.ndarray,
                       ph: int, pw: int,
                       bin_width_opl: float = 0.02,
                       start_opl    : float = 3.5,
                       save='output_spad/03_pixel_profile.png'):
    T          = clean.shape[2]
    c_px       = clean[ph, pw, :, :].mean(-1)
    n_px       = noisy[ph, pw, :, :].mean(-1)
    opl        = start_opl + bin_width_opl * (np.arange(T) + 0.5)
    c_light    = 299_792_458.0
    time_ns    = opl / c_light * 1e9

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. OPL domain
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(opl, c_px, color=PALETTE['clean'],  lw=2,   label='Clean')
    ax.plot(opl, n_px, color=PALETTE['noisy'],  lw=0.9, alpha=0.8, label='SPAD')
    ax.set_xlabel('Optical Path Length (m)')
    ax.set_ylabel('Radiance')
    ax.set_title('Signal in OPL Domain')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Time domain
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(time_ns, c_px, color=PALETTE['clean'], lw=2,   label='Clean')
    ax.plot(time_ns, n_px, color=PALETTE['noisy'], lw=0.9, alpha=0.8, label='SPAD')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Radiance')
    ax.set_title('Signal in Time Domain')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Noise difference
    ax = fig.add_subplot(gs[0, 2])
    diff = n_px - c_px
    ax.plot(opl, diff, color='#aaaacc', lw=0.8, alpha=0.7)
    ax.fill_between(opl, diff, 0,
                    where=(diff > 0), color='#ef5350', alpha=0.35,
                    label='Added noise')
    ax.fill_between(opl, diff, 0,
                    where=(diff < 0), color='#4fc3f7', alpha=0.35,
                    label='IRF smear (sub)')
    ax.axhline(0, color='white', lw=0.5)
    ax.set_xlabel('Optical Path Length (m)')
    ax.set_ylabel('Δ Radiance')
    ax.set_title('Noise Contribution\n(noisy − clean)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. RGB channels separately
    ax = fig.add_subplot(gs[1, 0])
    ch_colors = ['#ef5350', '#81c784', '#4fc3f7']
    ch_names  = ['R', 'G', 'B']
    for ci, (cc, cn) in enumerate(zip(ch_colors, ch_names)):
        ax.plot(opl, clean[ph, pw, :, ci],
                color=cc, lw=1.5, label=f'Clean {cn}')
        ax.plot(opl, noisy[ph, pw, :, ci],
                color=cc, lw=0.7, linestyle='--', alpha=0.7,
                label=f'SPAD {cn}')
    ax.set_xlabel('Optical Path Length (m)')
    ax.set_ylabel('Radiance')
    ax.set_title('Per-Channel Analysis (RGB)')
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(True, alpha=0.3)

    # 5. Histogram of noise values
    ax = fig.add_subplot(gs[1, 1])
    noise_vals = (noisy - clean).flatten()
    ax.hist(noise_vals, bins=120,
            color=PALETTE['noisy'], alpha=0.7, edgecolor='none',
            density=True)
    mu, sigma = noise_vals.mean(), noise_vals.std()
    x  = np.linspace(noise_vals.min(), noise_vals.max(), 300)
    ax.plot(x, np.exp(-0.5*((x-mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi)),
            color='white', lw=1.5, linestyle='--',
            label=f'Gaussian fit\nμ={mu:.4f}, σ={sigma:.4f}')
    ax.set_xlabel('Noise value')
    ax.set_ylabel('Density')
    ax.set_title('Noise Distribution\n(across full volume)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Instantaneous SNR vs bin
    ax = fig.add_subplot(gs[1, 2])
    eps      = 1e-8
    inst_snr = 20 * np.log10(np.abs(c_px) / (np.abs(diff) + eps))
    active   = c_px > c_px.max() * 0.005
    ax.plot(opl[active], inst_snr[active],
            color=PALETTE['shot'], lw=1.5, alpha=0.85)
    ax.axhline(0,  color='#ef5350', lw=1, linestyle='--',
               alpha=0.7, label='SNR = 0 dB')
    ax.axhline(10, color='#ffb74d', lw=1, linestyle='--',
               alpha=0.7, label='SNR = 10 dB')
    ax.axhline(20, color='#81c784', lw=1, linestyle='--',
               alpha=0.7, label='SNR = 20 dB')
    ax.set_xlabel('Optical Path Length (m)')
    ax.set_ylabel('Instantaneous SNR (dB)')
    ax.set_title('SNR per Time Bin\n(signal-present bins only)')
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

    _title_box(fig,
               f'Pixel ({ph}, {pw}) — Full Noise Analysis',
               'OPL domain · time domain · per-channel · noise histogram · SNR')
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save}")


# ── 6-E: Ablation study ──────────────────────────────────────────────────────
def plot_ablation(clean: np.ndarray,
                  sensor: SPADSensorModel,
                  ph: int, pw: int,
                  save='output_spad/04_ablation.png'):

    np.random.seed(42)
    stages = {
        'Clean\n(baseline)':       sensor.apply(clean, False, False, False),
        'IRF only\n(pulse smear)': sensor.apply(clean, True,  False, False),
        'Shot only\n(Poisson)':    sensor.apply(clean, False, True,  False),
        'DCR only\n(dark floor)':  sensor.apply(clean, False, False, True),
        'IRF + Shot':              sensor.apply(clean, True,  True,  False),
        'All stages\n(full SPAD)': sensor.apply(clean, True,  True,  True),
    }
    colors = [PALETTE['clean'], PALETTE['irf'], PALETTE['shot'],
              PALETTE['dcr'],   '#ffd54f',       PALETTE['noisy']]

    T    = clean.shape[2]
    opl  = 3.5 + 0.02 * (np.arange(T) + 0.5)

    def px(arr):
        return arr[ph, pw, :, :].mean(-1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (label, arr), col in zip(axes, stages.items(), colors):
        y = px(arr)
        ax.plot(opl, y, color=col, lw=1.5)
        ax.fill_between(opl, y, alpha=0.2, color=col)

        # stats annotation
        peak = y.max()
        mean = y[y > y.max() * 0.01].mean() if (y > y.max() * 0.01).any() else 0
        std  = y.std()
        ax.text(0.97, 0.95,
                f'peak={peak:.3f}\nmean={mean:.4f}\nstd={std:.4f}',
                transform=ax.transAxes,
                ha='right', va='top', fontsize=7,
                color='#dddddd',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#0f0f1a', alpha=0.7))
        ax.set_title(label, color='white')
        ax.set_xlabel('OPL (m)')
        ax.set_ylabel('Radiance')
        ax.grid(True, alpha=0.3)

    _title_box(fig, 'Ablation Study — Contribution of Each Noise Stage',
               'Each panel isolates or combines one or more physical effects')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save}")


# ── 6-F: Spatial heatmaps ────────────────────────────────────────────────────
def plot_spatial_heatmaps(clean : np.ndarray,
                          noisy : np.ndarray,
                          counts: np.ndarray,
                          save  = 'output_spad/05_spatial_heatmaps.png'):
    """
    Spatial summaries collapsed over the time axis.
    """
    # collapse time → integrated photon image
    clean_int  = clean.sum(axis=2).mean(axis=-1)   # (H, W)
    noisy_int  = noisy.sum(axis=2).mean(axis=-1)
    count_int  = counts.sum(axis=2).mean(axis=-1)
    noise_map  = (noisy - clean).std(axis=2).mean(axis=-1)

    # first-arrival map
    thresh        = clean.max() * 0.02
    active_mask   = clean.sum(axis=(-1, -2)) > thresh
    first_arr     = np.argmax(clean.sum(axis=-1) > thresh, axis=-1).astype(float)
    first_arr[~active_mask] = np.nan

    # peak-time map
    peak_time     = np.argmax(clean.sum(axis=-1), axis=-1).astype(float)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes      = axes.flatten()

    maps = [
        (clean_int,  'Integrated Radiance\n(clean)',         'plasma',  False),
        (noisy_int,  'Integrated Radiance\n(SPAD noisy)',    'plasma',  False),
        (count_int,  'Mean Photon Counts\n(per bin)',         'inferno', False),
        (noise_map,  'Noise Std Dev\n(spatial distribution)', 'hot',     False),
        (first_arr,  'First Photon Arrival\n(time bin)',      'viridis', True),
        (peak_time,  'Peak Radiance Time Bin\n(per pixel)',   'cool',    False),
    ]

    for ax, (data, title, cmap, use_nan) in zip(axes, maps):
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, color='white')
        ax.axis('off')

    _title_box(fig, 'Spatial Analysis — Sensor Output Heatmaps',
               'Time-integrated views of clean render vs SPAD output')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save}")


# ── 6-G: Parameter sensitivity ───────────────────────────────────────────────
def plot_parameter_sensitivity(clean: np.ndarray,
                               ph: int, pw: int,
                               save='output_spad/06_sensitivity.png'):
    """
    Show how SNR and peak shape change as we sweep sensor parameters.
    """
    c_px = clean[ph, pw, :, :].mean(-1)
    T    = len(c_px)
    opl  = 3.5 + 0.02 * (np.arange(T) + 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ① sweep photons_per_unit
    ax = axes[0]
    for ppu, col in [(50, '#ef5350'), (200, '#ffb74d'),
                     (1000, '#81c784'), (5000, '#4fc3f7')]:
        np.random.seed(0)
        sn = ShotNoise(ppu)
        noisy_px = sn.apply(clean[ph:ph+1, pw:pw+1])[0, 0, :, :].mean(-1)
        ax.plot(opl, noisy_px, color=col, lw=1.2,
                label=f'{ppu} ph/unit  SNR≈{np.sqrt(ppu*c_px.max()):.0f}')
    ax.plot(opl, c_px, color='white', lw=2.5, linestyle='--',
            label='Clean (truth)', zorder=5)
    ax.set_title('Shot Noise Sweep\n(photons_per_unit)')
    ax.set_xlabel('OPL (m)')
    ax.set_ylabel('Radiance')
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # ② sweep IRF sigma
    ax = axes[1]
    for sig, col in [(0.5, '#ef5350'), (1.5, '#ffb74d'),
                     (3.0, '#81c784'), (6.0, '#4fc3f7')]:
        kern  = IRF.skewed_gaussian(63, sig, 3.0)
        smear = signal.fftconvolve(c_px, kern, mode='same')
        ax.plot(opl, smear, color=col, lw=1.5,
                label=f'σ={sig} bins')
    ax.plot(opl, c_px, color='white', lw=2.5,
            linestyle='--', label='Clean', zorder=5)
    ax.set_title('IRF Width Sweep\n(skewed gaussian σ)')
    ax.set_xlabel('OPL (m)')
    ax.set_ylabel('Radiance')
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # ③ sweep DCR
    ax = axes[2]
    c_light = 299_792_458.0
    bws = 0.02 / c_light
    for dcr, col in [(100, '#ef5350'), (1000, '#ffb74d'),
                     (5000, '#81c784'), (20000, '#4fc3f7')]:
        np.random.seed(0)
        d     = DCR(dcr, bws)
        dpx   = d.apply(
            clean[ph:ph+1, pw:pw+1],
            photons_per_unit=1000.0
        )[0, 0, :, :].mean(-1)
        floor = dcr * bws / 1000.0
        ax.plot(opl, dpx, color=col, lw=1.2,
                label=f'{dcr} cps  floor≈{floor:.4f}')
    ax.plot(opl, c_px, color='white', lw=2.5,
            linestyle='--', label='Clean', zorder=5)
    ax.set_title('DCR Sweep\n(dark count rate in cps)')
    ax.set_xlabel('OPL (m)')
    ax.set_ylabel('Radiance')
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

    _title_box(fig, 'Parameter Sensitivity Analysis',
               'How each sensor parameter independently affects the output')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save}")


# ── 6-H: Metrics dashboard ───────────────────────────────────────────────────
def plot_metrics_dashboard(metrics: dict,
                           sensor : SPADSensorModel,
                           save   = 'output_spad/07_metrics.png'):
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

    # Left: bar chart of metrics
    ax = fig.add_subplot(gs[0])
    names  = ['SNR (dB)', 'PSNR (dB)', 'Peak ratio ×100',
              'Noise std ×1000', 'Bright phot √ (SNR theory)']
    values = [
        metrics['snr_db'],
        metrics['psnr_db'],
        metrics['peak_ratio'] * 100,
        metrics['noise_std']  * 1000,
        metrics['expected_snr_sqrt'],
    ]
    colors_bar = [PALETTE['clean'], PALETTE['shot'], PALETTE['irf'],
                  PALETTE['noisy'], PALETTE['dcr']]
    bars = ax.barh(names, values, color=colors_bar, alpha=0.8, edgecolor='none')
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=9, color='white')
    ax.set_xlabel('Value (see labels for units)')
    ax.set_title('Quantitative Metrics')
    ax.grid(True, alpha=0.3, axis='x')

    # Right: sensor configuration table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    config = [
        ['Parameter',            'Value'],
        ['IRF type',             sensor.irf_type],
        ['IRF σ (bins)',         f'{sensor.irf_sigma_bins:.1f}'],
        ['IRF skewness',         f'{sensor.irf_skewness:.1f}'],
        ['Photons / unit',       f'{sensor.photons_per_unit:.0f}'],
        ['DCR (cps)',            f'{sensor.dcr_cps:.0f}'],
        ['Bin width OPL (m)',    f'{sensor.bin_width_opl:.3f}'],
        ['Bin width (ps)',       f'{sensor.bin_width_opl/299792458*1e12:.1f}'],
        ['SNR (dB)',             f'{metrics["snr_db"]:.2f}'],
        ['PSNR (dB)',            f'{metrics["psnr_db"]:.2f}'],
    ]
    table = ax2.table(
        cellText    = config[1:],
        colLabels   = config[0],
        cellLoc     = 'center',
        loc         = 'center',
        bbox        = [0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor('#1a1a2e' if r > 0 else '#2a2a4e')
        cell.set_edgecolor('#444466')
        cell.set_text_props(color='white')
    ax2.set_title('Sensor Configuration', pad=15)

    _title_box(fig, 'Metrics & Configuration Dashboard')
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save}")


# ── 6-I: Comparison video ────────────────────────────────────────────────────
def save_comparison_video(clean : np.ndarray,
                          noisy : np.ndarray,
                          counts: np.ndarray,
                          save  = 'output_spad/08_comparison.mp4'):
    """
    3-panel side-by-side video: clean | SPAD noisy | photon counts (log scale).
    """
    def tm(arr):
        v = np.quantile(arr, 0.995)
        return np.clip(arr / (v + 1e-12), 0, 1)

    clean_tm  = tm(clean)
    noisy_tm  = tm(noisy)
    counts_f  = counts.astype(float)
    counts_tm = tm(counts_f)

    H, W, T, C = clean.shape
    W3 = W * 3 + 4   # 2-pixel dividers

    writer = cv2.VideoWriter(
        save,
        cv2.VideoWriter_fourcc(*'mp4v'),
        24, (W3, H + 30)    # +30 for info bar
    )

    total_signal = clean.sum()

    for t in range(T):
        canvas = np.zeros((H + 30, W3, 3), dtype=np.uint8)

        for col_idx, (arr, label, tint) in enumerate([
            (clean_tm,  'CLEAN',        np.array([0.3, 0.7, 1.0])),
            (noisy_tm,  'SPAD (noisy)', np.array([1.0, 0.3, 0.3])),
            (counts_tm, 'PHOTON CNT',   np.array([0.6, 1.0, 0.6])),
        ]):
            frame = np.clip(arr[:, :, t, :3], 0, 1)
            # apply subtle tint
            frame = frame * 0.85 + frame * tint * 0.15
            frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            x0 = col_idx * (W + 2)
            canvas[30:, x0:x0 + W] = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)

        # info bar
        opl_val  = 3.5 + 0.02 * (t + 0.5)
        time_ps  = opl_val / 299792458.0 * 1e12
        sig_frac = clean[:, :, t, :].sum() / (total_signal + 1e-12) * 100

        cv2.putText(canvas,
                    f't={t:03d}  OPL={opl_val:.2f}m  '
                    f'time={time_ps:.0f}ps  signal={sig_frac:.2f}%',
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 255), 1)

        for col_idx, label in enumerate(['CLEAN', 'SPAD (noisy)', 'PHOTON CNT']):
            cv2.putText(canvas, label,
                        (col_idx * (W + 2) + 5, H + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (180, 180, 255), 1)

        writer.write(canvas)

    writer.release()
    mb = os.path.getsize(save) / 1024 / 1024
    print(f"Saved: {save}  ({mb:.2f} MB, {T} frames @ 24 fps)")


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import mitransient as mitr

    RULER = '═' * 55

    # ── A. Render ─────────────────────────────────────────────────────────────
    print(f'\n{RULER}')
    print('A. RENDERING CLEAN SCENE (spp=512)')
    print(RULER)
    scene = mi.load_dict(mitr.cornell_box())
    _, img_transient = mi.render(scene, spp=512)
    clean = np.array(img_transient, dtype=np.float64)
    print(f'Shape : {clean.shape}   (H, W, T, C)')
    print(f'Range : [{clean.min():.4f}, {clean.max():.4f}]')

    # ── B. Sensor model ───────────────────────────────────────────────────────
    print(f'\n{RULER}')
    print('B. BUILDING SPAD SENSOR MODEL')
    print(RULER)
    sensor_model = SPADSensorModel(
        irf_type         = 'skewed_gaussian',
        irf_sigma_bins   = 2.0,
        irf_skewness     = 3.0,
        irf_kernel_size  = 31,
        photons_per_unit = 1000.0,
        dcr_cps          = 1000.0,
        bin_width_opl    = 0.02,
        seed             = 42,
    )
    print('Running full pipeline...')
    noisy  = sensor_model.apply(clean)
    counts = sensor_model.get_photon_counts(clean)
    print(f'Noisy range  : [{noisy.min():.4f},  {noisy.max():.4f}]')
    print(f'Count range  : [{counts.min()},  {counts.max()}]')

    # ── C. Metrics ────────────────────────────────────────────────────────────
    print(f'\n{RULER}')
    print('C. METRICS')
    print(RULER)
    metrics = compute_metrics(clean, noisy, sensor_model.photons_per_unit)
    col_w   = 22
    for k, v in metrics.items():
        print(f'  {k:<{col_w}}: {v:.4f}')

    # ── D. Bright pixel ───────────────────────────────────────────────────────
    bright_px = np.unravel_index(
        clean.sum(axis=(2, 3)).argmax(), clean.shape[:2])
    ph, pw = int(bright_px[0]), int(bright_px[1])
    print(f'\nBrightest pixel: ({ph}, {pw})')

    # ── E. All plots ──────────────────────────────────────────────────────────
    print(f'\n{RULER}')
    print('D. GENERATING ALL PLOTS')
    print(RULER)

    plot_pipeline_diagram()
    plot_irf_kernels()
    plot_noise_theory()
    plot_pixel_profile(clean, noisy, ph, pw)
    plot_ablation(clean, sensor_model, ph, pw)
    plot_spatial_heatmaps(clean, noisy, counts)
    plot_parameter_sensitivity(clean, ph, pw)
    plot_metrics_dashboard(metrics, sensor_model)

    # ── F. Video ──────────────────────────────────────────────────────────────
    print(f'\n{RULER}')
    print('E. SAVING COMPARISON VIDEO')
    print(RULER)
    save_comparison_video(clean, noisy, counts)

    # ── G. Save arrays ────────────────────────────────────────────────────────
    print(f'\n{RULER}')
    print('F. SAVING DATA ARRAYS')
    print(RULER)
    np.save('output_spad/clean.npy',        clean.astype(np.float32))
    np.save('output_spad/noisy.npy',        noisy.astype(np.float32))
    np.save('output_spad/photon_counts.npy', counts)
    print('Saved: clean.npy, noisy.npy, photon_counts.npy')

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{RULER}')
    print('SUMMARY — ALL OUTPUTS')
    print(RULER)
    outputs = [
        ('00_pipeline.png',       'Pipeline diagram with theory'),
        ('01_irf_kernels.png',    'All 3 IRF shapes + theory text'),
        ('02_noise_theory.png',   'Poisson distributions + SNR curve + DCR'),
        ('03_pixel_profile.png',  'Full single-pixel analysis (6 panels)'),
        ('04_ablation.png',       'Per-stage ablation study (6 panels)'),
        ('05_spatial_heatmaps.png','Spatial maps: radiance, counts, arrival'),
        ('06_sensitivity.png',    'Parameter sweep: ppu, σ_IRF, DCR'),
        ('07_metrics.png',        'Metrics bar chart + config table'),
        ('08_comparison.mp4',     '3-panel video: clean | SPAD | counts'),
        ('clean.npy',             'Clean transient float32'),
        ('noisy.npy',             'SPAD output float32'),
        ('photon_counts.npy',     'Integer photon counts int64'),
    ]
    for fname, desc in outputs:
        size = os.path.getsize(f'output_spad/{fname}')
        unit = 'MB' if size > 1e6 else 'KB'
        sval = size/1e6 if size > 1e6 else size/1e3
        print(f'  {fname:<30} {sval:6.1f} {unit}   {desc}')

    print(f'\n  SNR  = {metrics["snr_db"]:.2f} dB')
    print(f'  PSNR = {metrics["psnr_db"]:.2f} dB')
    print('\nDone.')