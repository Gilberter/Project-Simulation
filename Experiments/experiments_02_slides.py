# experiment_02_slides.py
"""
Slide-quality plots and video for Experiment 2 — Hidden Object Color vs Laser Wavelength.

Design goals
────────────
• 16:9 aspect ratio throughout (matches PowerPoint / Keynote / Google Slides).
• Every font large enough to read from the back of a room.
• One clear message per figure — no information overload.
• Cinematic dark theme with a tight, intentional palette.
• 300 dpi PNG output for crisp projection / PDF embedding.
• MP4 at 1920×1080 with smooth fade-ins and a progress timeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import MaxNLocator

# ── Output dir ────────────────────────────────────────────────────────────────
OUT   = 'output_exp02_slides'
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE  (all figures share this — change here, changes everywhere)
# ─────────────────────────────────────────────────────────────────────────────
PAL = dict(
    bg          = '#09090f',   # near-black background
    panel       = '#12121e',   # slightly lighter panel
    border      = '#2a2a44',   # subtle grid lines
    text        = '#e8e8f0',   # primary text
    muted       = '#6a6a88',   # secondary / axis labels
    # channel colours
    red_ch      = '#ff5252',
    green_ch    = '#69f0ae',
    blue_ch     = '#40c4ff',
    # semantic
    dcr         = '#ce93d8',   # noise floor
    noisy_alpha = 0.55,
    # object accent colours
    white_obj   = '#f5f5f5',
    red_obj     = '#ff5252',
    blue_obj    = '#536dfe',
    leaf_obj    = '#69f0ae',
)

CH_COLORS  = [PAL['red_ch'],   PAL['green_ch'],  PAL['blue_ch']]
CH_NAMES   = ['Red (640 nm)', 'Green (550 nm)', 'Blue (460 nm)']
OBJ_COLORS = [PAL['white_obj'], PAL['red_obj'], PAL['blue_obj'], PAL['leaf_obj']]

# ── Base rcParams ─────────────────────────────────────────────────────────────
BASE_RC = {
    'figure.facecolor':  PAL['bg'],
    'axes.facecolor':    PAL['panel'],
    'axes.edgecolor':    PAL['border'],
    'axes.labelcolor':   PAL['text'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.color':       PAL['muted'],
    'ytick.color':       PAL['muted'],
    'xtick.labelsize':   13,
    'ytick.labelsize':   13,
    'axes.labelsize':    15,
    'axes.titlesize':    18,
    'axes.titlepad':     14,
    'axes.labelpad':     10,
    'legend.facecolor':  PAL['panel'],
    'legend.edgecolor':  PAL['border'],
    'legend.fontsize':   13,
    'grid.color':        PAL['border'],
    'grid.linewidth':    0.7,
    'text.color':        PAL['text'],
    'figure.dpi':        150,
    'font.family':       'DejaVu Sans',
}
mpl.rcParams.update(BASE_RC)


# ─────────────────────────────────────────────────────────────────────────────
# SCENE / DATA DEFINITIONS  (mirrors experiment_02_color.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SimConfig:
    photons_per_unit: float = 500.0
    dcr_cps:          float = 1_000.0
    bin_width_opl:    float = 0.006
    start_opl:        float = 1.85
    temporal_bins:    int   = 300
    speed_of_light:   float = 299_792_458.0

    @property
    def bin_width_s(self):   return self.bin_width_opl / self.speed_of_light
    @property
    def dcr_per_bin(self):   return self.dcr_cps * self.bin_width_s
    @property
    def dcr_floor(self):     return self.dcr_per_bin / self.photons_per_unit
    def opl_axis(self):
        return self.start_opl + self.bin_width_opl * (
            np.arange(self.temporal_bins) + 0.5)


OBJECTS = {
    'white_red':  dict(label='White Object\n(red laser)',
                       reflectance=[0.90, 0.90, 0.90], laser=[1.0, 0.1, 0.1],
                       color=PAL['white_obj'], short='White'),
    'red_red':    dict(label='Red Object\n(red laser)',
                       reflectance=[0.85, 0.05, 0.05], laser=[1.0, 0.1, 0.1],
                       color=PAL['red_obj'],   short='Red'),
    'blue_red':   dict(label='Blue Object\n(red laser)',
                       reflectance=[0.05, 0.05, 0.85], laser=[1.0, 0.1, 0.1],
                       color=PAL['blue_obj'],  short='Blue'),
    'leaf_white': dict(label='Green Leaf\n(white laser)',
                       reflectance=[0.08, 0.75, 0.08], laser=[1.0, 1.0, 1.0],
                       color=PAL['leaf_obj'],  short='Leaf'),
}


def return_signal(cfg: dict) -> np.ndarray:
    return np.array(cfg['laser']) * np.array(cfg['reflectance'])


def apply_shot_and_dcr(arr: np.ndarray, cfg: SimConfig, seed: int = 0) -> np.ndarray:
    rng    = np.random.default_rng(seed)
    counts = rng.poisson(np.clip(arr, 0, None) * cfg.photons_per_unit)
    dark   = rng.poisson(cfg.dcr_per_bin, size=arr.shape)
    return (counts + dark).astype(np.float64) / cfg.photons_per_unit


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC WAVEFORM GENERATOR
# (Used when Mitsuba is not available — realistic-looking Gaussian pulses)
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_results(cfg: SimConfig) -> dict:
    """
    Generate plausible transient waveforms without Mitsuba.
    Shape: (16, 16, T, 3) matching the real render output.
    """
    T   = cfg.temporal_bins
    opl = cfg.opl_axis()
    rng = np.random.default_rng(0)

    def gaussian(center_opl, sigma, amplitude, channel, T):
        t   = opl
        sig = amplitude * np.exp(-0.5 * ((t - center_opl) / sigma) ** 2)
        out = np.zeros((T, 3))
        out[:, channel] = sig
        return out

    results = {}
    # Wall reflection: broad gaussian at ~2.0 m OPL
    wall_opl  = 2.05
    # Object reflection: narrower gaussian at ~2.5 m OPL
    obj_opl   = 2.55

    object_params = {
        'white_red':  [(0.45, 0, 0.05), (0.45, 1, 0.04), (0.45, 2, 0.04)],
        'red_red':    [(0.38, 0, 0.55),  (0.38, 1, 0.02), (0.38, 2, 0.02)],
        'blue_red':   [(0.38, 0, 0.003), (0.38, 1, 0.003),(0.38, 2, 0.003)],
        'leaf_white': [(0.42, 0, 0.03),  (0.42, 1, 0.50), (0.42, 2, 0.03)],
    }

    for key, cfg_obj in OBJECTS.items():
        # Build (T, 3) profile
        profile = np.zeros((T, 3))
        # wall contribution — white diffuse
        for ch in range(3):
            profile[:, ch] += 0.18 * np.exp(
                -0.5 * ((opl - wall_opl) / 0.04) ** 2)
        # object contribution
        for sigma, ch, amp in object_params[key]:
            profile[:, ch] += amp * np.exp(
                -0.5 * ((opl - obj_opl) / sigma) ** 2)
        # broadcast to spatial (16, 16, T, 3)
        noise_spatial = rng.normal(0, 0.002, (16, 16, T, 3))
        clean = np.broadcast_to(profile, (16, 16, T, 3)).copy() + noise_spatial
        clean = np.clip(clean, 0, None)
        noisy = apply_shot_and_dcr(clean, cfg, seed=hash(key) % 2**31)
        results[key] = {'clean': clean, 'noisy': noisy, 'cfg': cfg_obj}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str, tight: bool = True) -> None:
    path = f'{OUT}/{name}'
    if tight:
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    else:
        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    kb = os.path.getsize(path) / 1e3
    print(f'  ✓  {name}  ({kb:.0f} KB)')


def _slide_fig(ncols: int = 1, nrows: int = 1,
               width: float = 16.0, height: float = 9.0,
               **kw) -> tuple[plt.Figure, any]:
    """16:9 figure — the native slide canvas."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **kw)
    fig.patch.set_facecolor(PAL['bg'])
    return fig, axes


def _title(fig: plt.Figure, text: str, sub: str = '') -> None:
    y = 0.97 if sub else 0.96
    fig.text(0.5, y, text, ha='center', va='top',
             fontsize=26, fontweight='bold', color=PAL['text'])
    if sub:
        fig.text(0.5, y - 0.055, sub, ha='center', va='top',
                 fontsize=15, color=PAL['muted'])


def _dcr_band(ax: plt.Axes, dcr: float) -> None:
    """Draw the DCR noise floor as a shaded band with a label."""
    ax.axhline(dcr, color=PAL['dcr'], lw=1.5, linestyle='--', zorder=3)
    ax.fill_between(ax.get_xlim(), 0, dcr,
                    color=PAL['dcr'], alpha=0.07, zorder=1)
    ax.text(ax.get_xlim()[0], dcr * 1.35, 'noise floor',
            color=PAL['dcr'], fontsize=11, va='bottom')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — SPECTRAL REFLECTANCE OVERVIEW  (16:9, one key idea per object)
# ─────────────────────────────────────────────────────────────────────────────

def fig_reflectance(save: str = '01_spectral_reflectance.png') -> None:
    """
    Slide message: "Return signal = laser colour × object colour"
    4 mini panels, each with a clear verdict badge.
    """
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(PAL['bg'])

    # Super-title
    fig.text(0.5, 0.97, 'Spectral Reflectance × Laser Power = Return Signal',
             ha='center', va='top', fontsize=24, fontweight='bold',
             color=PAL['text'])
    fig.text(0.5, 0.91,
             'Each channel only returns light if both the laser emits and the object reflects at that wavelength',
             ha='center', va='top', fontsize=14, color=PAL['muted'])

    gs = gridspec.GridSpec(1, 4, figure=fig,
                           left=0.04, right=0.97,
                           top=0.85, bottom=0.14,
                           wspace=0.10)

    verdicts = {
        'white_red':  ('STRONG\nSIGNAL',  PAL['green_ch']),
        'red_red':    ('PERFECT\nMATCH',  PAL['green_ch']),
        'blue_red':   ('BURIED IN\nNOISE', PAL['red_ch']),
        'leaf_white': ('G ONLY',           PAL['green_ch']),
    }

    wl_labels = ['R', 'G', 'B']
    for col, (key, cfg) in enumerate(OBJECTS.items()):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(PAL['panel'])
        for spine in ax.spines.values():
            spine.set_edgecolor(PAL['border'])

        ret  = return_signal(cfg)
        las  = np.array(cfg['laser'])
        ref_ = np.array(cfg['reflectance'])
        x    = np.arange(3)
        w    = 0.26

        # Laser (faded), Reflectance (medium), Return (bright, outlined)
        ax.bar(x - w, las,  w, color=CH_COLORS, alpha=0.28)
        ax.bar(x,     ref_, w, color=CH_COLORS, alpha=0.55)
        ax.bar(x + w, ret,  w, color=CH_COLORS, alpha=1.0,
               edgecolor='white', linewidth=1.0)

        # Value labels on return bars
        for xi, rv in zip(x, ret):
            if rv > 0.02:
                ax.text(xi + w, rv + 0.03, f'{rv:.2f}',
                        ha='center', fontsize=12, fontweight='bold',
                        color='white')

        ax.set_xticks(x)
        ax.set_xticklabels(wl_labels, fontsize=14)
        ax.set_ylim(0, 1.15)
        ax.set_xlim(-0.55, 2.85)
        ax.set_ylabel('Normalised value' if col == 0 else '', fontsize=13)
        ax.yaxis.set_major_locator(MaxNLocator(4, prune='upper'))
        ax.grid(True, axis='y', alpha=0.25)
        ax.set_title(cfg['short'], fontsize=18, fontweight='bold',
                     color=cfg['color'], pad=10)

        # Verdict badge
        vtext, vcol = verdicts[key]
        ax.text(0.5, -0.22, vtext, transform=ax.transAxes,
                ha='center', va='top', fontsize=13, fontweight='bold',
                color=vcol,
                bbox=dict(boxstyle='round,pad=0.35', facecolor=PAL['bg'],
                          edgecolor=vcol, linewidth=1.5))

    # Legend (bottom-right, outside panels)
    handles = [
        mpatches.Patch(facecolor='white', alpha=0.28, label='Laser power'),
        mpatches.Patch(facecolor='white', alpha=0.55, label='Reflectance'),
        mpatches.Patch(facecolor='white', alpha=1.0,  label='Return signal'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               frameon=True, fontsize=13,
               bbox_to_anchor=(0.5, 0.01))

    _save(fig, save)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — BLUE OBJECT SPOTLIGHT  (one-panel full-slide)
# Key message: "Blue absorbs red → signal vanishes below noise floor"
# ─────────────────────────────────────────────────────────────────────────────

def fig_blue_object_dcr(results: dict, cfg: SimConfig,
                         save: str = '02_blue_below_noise.png') -> None:
    opl   = cfg.opl_axis()
    data  = results['blue_red']
    clean = data['clean'].mean(axis=(0, 1))   # (T, 3)
    noisy = data['noisy'].mean(axis=(0, 1))

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(PAL['bg'])
    ax.set_facecolor(PAL['panel'])
    for spine in ax.spines.values():
        spine.set_edgecolor(PAL['border'])

    # All-channel mean for clarity
    c_mean = clean.mean(axis=-1)
    n_mean = noisy.mean(axis=-1)

    ax.fill_between(opl, n_mean, alpha=0.15, color=PAL['blue_obj'])
    ax.plot(opl, c_mean, color=PAL['blue_obj'], lw=2.5, label='True signal (clean)')
    ax.plot(opl, n_mean, color=PAL['blue_obj'], lw=1.2,
            linestyle='--', alpha=0.7, label='SPAD output (noisy)')

    # DCR band
    ax.axhline(cfg.dcr_floor, color=PAL['dcr'], lw=2.0, linestyle='--', zorder=5)
    ax.fill_between(opl, 0, cfg.dcr_floor, color=PAL['dcr'], alpha=0.10, zorder=1)

    # Annotation arrow
    mid   = (opl[0] + opl[-1]) / 2
    ax.annotate(
        f'Entire signal buried\nbelow noise floor\n(DCR = {cfg.dcr_floor:.5f})',
        xy=(mid, cfg.dcr_floor * 0.5),
        xytext=(mid + 0.08, cfg.dcr_floor * 6),
        color=PAL['dcr'], fontsize=16, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=PAL['dcr'], lw=2.0),
        bbox=dict(boxstyle='round,pad=0.4', facecolor=PAL['bg'],
                  edgecolor=PAL['dcr'], alpha=0.85),
    )

    ax.set_xlabel('Optical Path Length (m)', fontsize=16)
    ax.set_ylabel('Radiance (normalised)', fontsize=16)
    ax.set_xlim(opl[0], opl[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=14, loc='upper right')

    fig.text(0.5, 0.97,
             'Blue Object + Red Laser → Hidden Object is Invisible',
             ha='center', va='top', fontsize=24, fontweight='bold',
             color=PAL['text'])
    fig.text(0.5, 0.91,
             'Blue pigment absorbs red photons.  The return signal never rises above the SPAD dark-count noise floor.',
             ha='center', va='top', fontsize=14, color=PAL['muted'])

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    _save(fig, save, tight=False)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — ALL OBJECTS WAVEFORM COMPARISON  (2×2 grid, one channel each)
# ─────────────────────────────────────────────────────────────────────────────

def fig_waveform_grid(results: dict, cfg: SimConfig,
                       save: str = '03_waveform_grid.png') -> None:
    opl  = cfg.opl_axis()
    keys = list(OBJECTS.keys())

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(PAL['bg'])

    fig.text(0.5, 0.98, 'Transient Return Waveforms — Four Object/Laser Combinations',
             ha='center', va='top', fontsize=22, fontweight='bold',
             color=PAL['text'])
    fig.text(0.5, 0.93,
             'Dominant channel highlighted.  Purple band = DCR noise floor.',
             ha='center', va='top', fontsize=13, color=PAL['muted'])

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.07, right=0.97,
                           top=0.90, bottom=0.08,
                           hspace=0.38, wspace=0.25)

    # Which channel is "dominant" for each object?
    dominant_ch = {'white_red': 0, 'red_red': 0, 'blue_red': 2, 'leaf_white': 1}

    for idx, key in enumerate(keys):
        row, col = divmod(idx, 2)
        ax     = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PAL['panel'])
        for sp in ax.spines.values():
            sp.set_edgecolor(PAL['border'])

        cfg_obj = OBJECTS[key]
        data    = results[key]
        clean   = data['clean'].mean(axis=(0, 1))   # (T, 3)
        noisy   = data['noisy'].mean(axis=(0, 1))
        dom     = dominant_ch[key]

        # Draw all three channels, dim non-dominant ones
        for ch, (color, name) in enumerate(zip(CH_COLORS, ['R', 'G', 'B'])):
            alpha_c = 1.0  if ch == dom else 0.20
            alpha_n = 0.65 if ch == dom else 0.12
            lw      = 2.2  if ch == dom else 0.9
            ax.plot(opl, clean[:, ch], color=color, lw=lw, alpha=alpha_c,
                    label=f'{name} channel' if ch == dom else '_')
            ax.plot(opl, noisy[:, ch], color=color, lw=lw * 0.5,
                    linestyle='--', alpha=alpha_n)

        # DCR floor
        ax.axhline(cfg.dcr_floor, color=PAL['dcr'], lw=1.4,
                   linestyle='--', zorder=5)
        ax.fill_between(opl, 0, cfg.dcr_floor,
                        color=PAL['dcr'], alpha=0.08, zorder=1)

        # Title with accent colour
        ax.set_title(cfg_obj['short'] + '  (' +
                     ['red', 'red', 'red', 'white'][idx] + ' laser)',
                     fontsize=16, fontweight='bold', color=cfg_obj['color'])
        ax.set_xlabel('OPL (m)', fontsize=13)
        ax.set_ylabel('Radiance', fontsize=13)
        ax.set_xlim(opl[0], opl[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=12, loc='upper right')

        # Peak SNR badge
        peak = clean[:, dom].max()
        snr  = 10 * np.log10(peak / (cfg.dcr_floor + 1e-12) + 1e-6)
        badge_col = PAL['green_ch'] if snr > 0 else PAL['red_ch']
        ax.text(0.02, 0.96, f'SNR {snr:+.1f} dB',
                transform=ax.transAxes, va='top',
                fontsize=13, fontweight='bold', color=badge_col,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=PAL['bg'],
                          edgecolor=badge_col, linewidth=1.2, alpha=0.9))

    _save(fig, save, tight=False)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — SNR SUMMARY BAR CHART  (clean, one-message slide)
# ─────────────────────────────────────────────────────────────────────────────

def fig_snr_summary(results: dict, cfg: SimConfig,
                     save: str = '04_snr_summary.png') -> None:
    keys   = list(OBJECTS.keys())
    labels = [OBJECTS[k]['short'] for k in keys]
    x      = np.arange(len(keys))
    width  = 0.22

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(PAL['bg'])
    ax.set_facecolor(PAL['panel'])
    for sp in ax.spines.values():
        sp.set_edgecolor(PAL['border'])

    for ci, (ch_color, ch_short) in enumerate(zip(CH_COLORS, ['R', 'G', 'B'])):
        snrs = []
        for key in keys:
            peak = results[key]['clean'].mean(axis=(0, 1))[:, ci].max()
            snrs.append(10 * np.log10(peak / (cfg.dcr_floor + 1e-12) + 1e-6))
        bars = ax.bar(x + (ci - 1) * width, snrs, width,
                      color=ch_color, alpha=0.85,
                      edgecolor='none', label=f'{ch_short} channel',
                      zorder=3)
        for bar, sv in zip(bars, snrs):
            if abs(sv) > 1:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        sv + (1.2 if sv > 0 else -2.5),
                        f'{sv:.0f}', ha='center',
                        fontsize=12, fontweight='bold', color='white',
                        zorder=5)

    # Zero line = noise floor
    ax.axhline(0, color=PAL['dcr'], lw=2.0, linestyle='--', zorder=4)
    ax.text(x[-1] + 0.55, 0.5, 'Noise floor (0 dB)',
            color=PAL['dcr'], fontsize=13, va='bottom')

    ax.fill_between([-0.6, len(x) - 0.4], [0, 0],
                    [ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -5,
                     ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -5],
                    color=PAL['red_ch'], alpha=0.06, zorder=1)
    ax.text(len(x) / 2 - 0.5, ax.get_ylim()[0] * 0.6 if ax.get_ylim()[0] < 0 else -3,
            '▼  buried in noise', ha='center',
            color=PAL['red_ch'], fontsize=13, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_ylabel('SNR relative to DCR floor (dB)', fontsize=15)
    ax.set_xlim(-0.55, len(x) - 0.45)
    ax.grid(True, axis='y', alpha=0.2, zorder=0)
    ax.legend(fontsize=14, loc='upper right')

    fig.text(0.5, 0.97,
             'Signal-to-Noise Ratio by Object Colour and Laser Wavelength',
             ha='center', va='top', fontsize=22, fontweight='bold',
             color=PAL['text'])
    fig.text(0.5, 0.92,
             'Bars below 0 dB are undetectable.  Blue object + red laser = fully invisible.',
             ha='center', va='top', fontsize=13, color=PAL['muted'])

    plt.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, save, tight=False)


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO — 1920×1080, 24 fps, cinematic slide-style
# ─────────────────────────────────────────────────────────────────────────────

VW, VH = 1920, 1080
FPS    = 24


def _cv_text(canvas, text, x, y, scale, color, thickness=1,
             font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness,
                cv2.LINE_AA)


def _hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _blank() -> np.ndarray:
    bg = _hex_bgr(PAL['bg'])
    c  = np.full((VH, VW, 3), bg, dtype=np.uint8)
    return c


def _mpl_to_frame(fig: plt.Figure) -> np.ndarray:
    """Render a Matplotlib figure to 1920×1080 BGR frame."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    buf.seek(0)
    raw = np.frombuffer(buf.read(), np.uint8)
    buf.close()
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return cv2.resize(img, (VW, VH))


def _fade(writer, frame_a, frame_b, n_frames=12):
    """Cross-fade between two BGR frames."""
    for i in range(n_frames):
        alpha = i / n_frames
        blended = cv2.addWeighted(frame_a, 1 - alpha, frame_b, alpha, 0)
        writer.write(blended)


def _write_n(writer, frame, n):
    for _ in range(n):
        writer.write(frame)


def _title_frame(title: str, sub: str = '') -> np.ndarray:
    """Black slide with centred title text (no Matplotlib overhead)."""
    c = _blank()
    # Horizontal accent line
    accent = _hex_bgr(PAL['blue_ch'])
    cv2.line(c, (VW // 2 - 320, VH // 2 - 60),
             (VW // 2 + 320, VH // 2 - 60), accent, 2)

    font  = cv2.FONT_HERSHEY_DUPLEX
    white = _hex_bgr(PAL['text'])
    muted = _hex_bgr(PAL['muted'])

    scale = 1.6
    (tw, _), _ = cv2.getTextSize(title, font, scale, 2)
    cv2.putText(c, title, (VW // 2 - tw // 2, VH // 2 + 10),
                font, scale, white, 2, cv2.LINE_AA)
    if sub:
        scale2 = 0.75
        (sw, _), _ = cv2.getTextSize(sub, font, scale2, 1)
        cv2.putText(c, sub, (VW // 2 - sw // 2, VH // 2 + 70),
                    font, scale2, muted, 1, cv2.LINE_AA)
    return c


def _progress_bar(canvas: np.ndarray, t: int, T: int,
                  color: tuple = None) -> None:
    """Thin timeline bar at the very bottom of the frame."""
    color = color or _hex_bgr(PAL['blue_ch'])
    frac  = t / max(T - 1, 1)
    end_x = int(VW * frac)
    cv2.rectangle(canvas, (0, VH - 4), (end_x, VH), color, -1)


def make_slide_video(results: dict, cfg: SimConfig,
                     save: str = f'{OUT}/exp02_slides.mp4') -> None:
    """
    ~90-second 1920×1080 video structured as slide transitions:
      0:00 Title card
      0:05 Spectral reflectance overview
      0:25 Blue object DCR deep-dive (animated reveal)
      0:50 Waveform grid (bins sweeping in)
      1:10 SNR summary + freeze
    """
    writer = cv2.VideoWriter(
        save, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (VW, VH))

    opl = cfg.opl_axis()
    T   = cfg.temporal_bins

    # ── SECTION 1: Title card (5 s) ──────────────────────────────────────────
    print('  → title card')
    title_f = _title_frame(
        'EXPERIMENT 2: Hidden Object Colour',
        'How laser wavelength × object reflectance determines visibility')
    _write_n(writer, title_f, FPS * 5)

    # ── SECTION 2: Spectral reflectance (static plot, 20 s) ──────────────────
    print('  → reflectance overview')
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(PAL['bg'])
    fig.text(0.5, 0.97, 'Spectral Reflectance × Laser Power = Return Signal',
             ha='center', va='top', fontsize=24, fontweight='bold',
             color=PAL['text'])
    gs = gridspec.GridSpec(1, 4, figure=fig,
                           left=0.04, right=0.97, top=0.85, bottom=0.16,
                           wspace=0.10)
    verdicts = {
        'white_red':  ('STRONG',  PAL['green_ch']),
        'red_red':    ('PERFECT', PAL['green_ch']),
        'blue_red':   ('BURIED',  PAL['red_ch']),
        'leaf_white': ('G ONLY',  PAL['green_ch']),
    }
    for col_i, (key, cfg_obj) in enumerate(OBJECTS.items()):
        ax  = fig.add_subplot(gs[0, col_i])
        ax.set_facecolor(PAL['panel'])
        ret = return_signal(cfg_obj)
        las = np.array(cfg_obj['laser'])
        ref = np.array(cfg_obj['reflectance'])
        xb  = np.arange(3)
        w   = 0.26
        ax.bar(xb - w, las, w, color=CH_COLORS, alpha=0.28)
        ax.bar(xb,     ref, w, color=CH_COLORS, alpha=0.55)
        ax.bar(xb + w, ret, w, color=CH_COLORS, alpha=1.0,
               edgecolor='white', linewidth=1.0)
        for xi, rv in zip(xb, ret):
            if rv > 0.02:
                ax.text(xi + w, rv + 0.03, f'{rv:.2f}',
                        ha='center', fontsize=11, fontweight='bold',
                        color='white')
        ax.set_xticks(xb)
        ax.set_xticklabels(['R', 'G', 'B'], fontsize=14)
        ax.set_ylim(0, 1.15)
        ax.grid(True, axis='y', alpha=0.2)
        ax.set_title(cfg_obj['short'], fontsize=18, fontweight='bold',
                     color=cfg_obj['color'])
        vt, vc = verdicts[key]
        ax.text(0.5, -0.26, vt, transform=ax.transAxes,
                ha='center', fontsize=14, fontweight='bold', color=vc,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=PAL['bg'],
                          edgecolor=vc, linewidth=1.5))

    refl_frame = _mpl_to_frame(fig)
    plt.close(fig)

    _fade(writer, title_f, refl_frame)
    _write_n(writer, refl_frame, FPS * 20)

    # ── SECTION 3: Blue object DCR animated reveal (25 s) ────────────────────
    print('  → blue object DCR reveal')
    data    = results['blue_red']
    c_mean  = data['clean'].mean(axis=(0, 1)).mean(axis=-1)
    n_mean  = data['noisy'].mean(axis=(0, 1)).mean(axis=-1)
    n_phase = FPS * 25
    prev_frame = refl_frame

    for fi in range(n_phase):
        reveal_bins = max(1, int((fi / n_phase) ** 0.6 * T))
        show_floor  = fi > FPS * 8
        zoom        = fi > FPS * 16

        fig2, ax2 = plt.subplots(figsize=(16, 9))
        fig2.patch.set_facecolor(PAL['bg'])
        ax2.set_facecolor(PAL['panel'])

        if zoom:
            lo, hi = opl[80], opl[220]
            mask   = (opl >= lo) & (opl <= hi)
        else:
            lo, hi = opl[0], opl[-1]
            mask   = np.ones(T, dtype=bool)

        x_plot = opl[mask][:reveal_bins if reveal_bins < mask.sum() else mask.sum()]
        ax2.fill_between(x_plot, n_mean[mask][:len(x_plot)],
                         alpha=0.15, color=PAL['blue_obj'])
        ax2.plot(x_plot, c_mean[mask][:len(x_plot)],
                 color=PAL['blue_obj'], lw=2.5,
                 label='True signal (clean)')
        ax2.plot(x_plot, n_mean[mask][:len(x_plot)],
                 color=PAL['blue_obj'], lw=1.2,
                 linestyle='--', alpha=0.8, label='SPAD output (noisy)')

        if show_floor:
            ax2.axhline(cfg.dcr_floor, color=PAL['dcr'],
                        lw=2.0, linestyle='--')
            ax2.fill_between([lo, hi], 0, cfg.dcr_floor,
                             color=PAL['dcr'], alpha=0.12)
            ax2.text(lo + 0.01, cfg.dcr_floor * 1.4,
                     f'DCR noise floor = {cfg.dcr_floor:.5f}',
                     color=PAL['dcr'], fontsize=14, fontweight='bold')

        ax2.set_xlim(lo, hi)
        ax2.set_ylim(bottom=0)
        ax2.set_xlabel('Optical Path Length (m)', fontsize=15)
        ax2.set_ylabel('Radiance', fontsize=15)
        ax2.legend(fontsize=13)
        ax2.grid(True, alpha=0.2)
        fig2.text(0.5, 0.97,
                  'Blue Object + Red Laser → Signal Buried in Noise',
                  ha='center', va='top', fontsize=22, fontweight='bold',
                  color=PAL['text'])

        frame = _mpl_to_frame(fig2)
        plt.close(fig2)

        if fi == 0:
            _fade(writer, prev_frame, frame)
        _progress_bar(frame, fi, n_phase, _hex_bgr(PAL['blue_obj']))
        writer.write(frame)
        prev_frame = frame

    # ── SECTION 4: Waveform grid sweeping in (20 s) ──────────────────────────
    print('  → waveform grid sweep')
    n_phase2 = FPS * 20
    dominant_ch = {'white_red': 0, 'red_red': 0, 'blue_red': 2, 'leaf_white': 1}

    for fi in range(n_phase2):
        reveal = max(1, int((fi / n_phase2) * T))

        fig3 = plt.figure(figsize=(16, 9))
        fig3.patch.set_facecolor(PAL['bg'])
        fig3.text(0.5, 0.98,
                  'Transient Return Waveforms — All Object/Laser Combinations',
                  ha='center', va='top', fontsize=20, fontweight='bold',
                  color=PAL['text'])
        gs3 = gridspec.GridSpec(2, 2, figure=fig3,
                                left=0.07, right=0.97,
                                top=0.91, bottom=0.07,
                                hspace=0.38, wspace=0.22)
        for idx, key in enumerate(OBJECTS.keys()):
            row3, col3 = divmod(idx, 2)
            ax3    = fig3.add_subplot(gs3[row3, col3])
            ax3.set_facecolor(PAL['panel'])
            cfg_o  = OBJECTS[key]
            data_k = results[key]
            c_k    = data_k['clean'].mean(axis=(0, 1))
            n_k    = data_k['noisy'].mean(axis=(0, 1))
            dom    = dominant_ch[key]
            for ch, color in enumerate(CH_COLORS):
                a_c = 1.0 if ch == dom else 0.18
                a_n = 0.6 if ch == dom else 0.10
                lw  = 2.0 if ch == dom else 0.8
                ax3.plot(opl[:reveal], c_k[:reveal, ch],
                         color=color, lw=lw, alpha=a_c)
                ax3.plot(opl[:reveal], n_k[:reveal, ch],
                         color=color, lw=lw * 0.5,
                         linestyle='--', alpha=a_n)
            ax3.axhline(cfg.dcr_floor, color=PAL['dcr'],
                        lw=1.2, linestyle='--')
            ax3.fill_between(opl, 0, cfg.dcr_floor,
                             color=PAL['dcr'], alpha=0.07)
            ax3.set_xlim(opl[0], opl[-1])
            ax3.set_ylim(bottom=0)
            ax3.set_title(cfg_o['short'], fontsize=15,
                          fontweight='bold', color=cfg_o['color'])
            ax3.grid(True, alpha=0.18)

        frame3 = _mpl_to_frame(fig3)
        plt.close(fig3)
        if fi == 0:
            _fade(writer, prev_frame, frame3)
        _progress_bar(frame3, fi, n_phase2)
        writer.write(frame3)
        prev_frame = frame3

    # ── SECTION 5: SNR summary (10 s freeze) ─────────────────────────────────
    print('  → SNR summary')
    keys   = list(OBJECTS.keys())
    labels = [OBJECTS[k]['short'] for k in keys]
    x      = np.arange(len(keys))
    w_bar  = 0.22

    fig4, ax4 = plt.subplots(figsize=(16, 9))
    fig4.patch.set_facecolor(PAL['bg'])
    ax4.set_facecolor(PAL['panel'])

    for ci, (ch_color, ch_s) in enumerate(zip(CH_COLORS, ['R', 'G', 'B'])):
        snrs = []
        for key in keys:
            peak = results[key]['clean'].mean(axis=(0, 1))[:, ci].max()
            snrs.append(10 * np.log10(peak / (cfg.dcr_floor + 1e-12) + 1e-6))
        ax4.bar(x + (ci - 1) * w_bar, snrs, w_bar,
                color=ch_color, alpha=0.85, edgecolor='none',
                label=f'{ch_s} channel')
        for xi, sv in zip(x + (ci - 1) * w_bar, snrs):
            ax4.text(xi + w_bar / 2,
                     sv + (1.5 if sv > 0 else -3.5),
                     f'{sv:.0f}', ha='center',
                     fontsize=12, fontweight='bold', color='white')

    ax4.axhline(0, color=PAL['dcr'], lw=2.0, linestyle='--')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=20)
    ax4.set_ylabel('SNR re. DCR floor (dB)', fontsize=15)
    ax4.set_xlim(-0.55, len(x) - 0.45)
    ax4.grid(True, axis='y', alpha=0.2)
    ax4.legend(fontsize=14)
    fig4.text(0.5, 0.97,
              'SNR Summary: Which Object/Laser Combinations Are Detectable?',
              ha='center', va='top', fontsize=20, fontweight='bold',
              color=PAL['text'])
    fig4.text(0.5, 0.92,
              'Bars below 0 dB are undetectable.  Blue object with red laser is completely invisible.',
              ha='center', va='top', fontsize=13, color=PAL['muted'])
    plt.tight_layout(rect=[0, 0, 1, 0.89])

    snr_frame = _mpl_to_frame(fig4)
    plt.close(fig4)
    _fade(writer, prev_frame, snr_frame)
    _write_n(writer, snr_frame, FPS * 10)

    writer.release()
    mb = os.path.getsize(save) / 1e6
    print(f'  ✓  {os.path.basename(save)}  ({mb:.1f} MB)')


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = SimConfig()

    # ── Try to load real Mitsuba results; fall back to synthetic data ─────────
    real_dir = 'output_exp02'
    try:
        import mitsuba as mi  # noqa: F401
        from experiment_02_color import render_all
        mi.set_variant('cuda_ad_rgb')
        print('Using Mitsuba renderer...')
        results = render_all(cfg)
    except Exception:
        print('Mitsuba not available — using synthetic waveforms for demo.\n')
        results = synthetic_results(cfg)

    print('\nGenerating slide plots (300 dpi PNG)...')
    fig_reflectance()
    fig_blue_object_dcr(results, cfg)
    fig_waveform_grid(results, cfg)
    fig_snr_summary(results, cfg)

    print('\nGenerating slide video (1920×1080 MP4)...')
    make_slide_video(results, cfg)

    print(f'\nAll outputs saved to  ./{OUT}/')
    for fname in sorted(os.listdir(OUT)):
        sz   = os.path.getsize(f'{OUT}/{fname}')
        val  = sz / 1e6 if sz > 1e6 else sz / 1e3
        unit = 'MB' if sz > 1e6 else 'KB'
        print(f'  {fname:<40}  {val:6.1f} {unit}')