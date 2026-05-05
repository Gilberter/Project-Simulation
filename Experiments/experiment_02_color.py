# experiment_02_color.py
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         EXPERIMENT 2: Hidden Object Color vs Laser Wavelength               ║
║                                                                              ║
║  THEORY — Spectral Reflectance:                                             ║
║  Every object has a reflectance spectrum R(λ): the fraction of incident    ║
║  light it reflects at each wavelength λ.                                    ║
║                                                                              ║
║  The return signal in each RGB channel is:                                  ║
║      signal_ch = laser_power_ch × R_object(ch) × geometry_factor           ║
║                                                                              ║
║  Key cases in this experiment:                                               ║
║  • Red laser  + White object  → strong return in R, weaker in G/B          ║
║  • Red laser  + Red object    → strong return in R channel only             ║
║  • Red laser  + Blue object   → near-zero return (blue absorbs red!)        ║
║  • White laser+ Green leaf    → return only in G channel (R+B absorbed)    ║
║                                                                              ║
║  The SPAD shot noise model then shows that:                                 ║
║  Low-reflectance objects produce so few photons that their signal           ║
║  is buried beneath the Dark Count Rate noise floor.                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import NamedTuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.stats import skewnorm

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'figure.facecolor': '#0f0f1a', 'axes.facecolor':   '#1a1a2e',
    'axes.edgecolor':   '#444466', 'axes.labelcolor':  '#ccccee',
    'xtick.color':      '#aaaacc', 'ytick.color':      '#aaaacc',
    'text.color':       '#ccccee', 'grid.color':       '#2a2a4a',
    'grid.linewidth':    0.6,      'legend.facecolor': '#1a1a2e',
    'legend.edgecolor': '#444466', 'figure.dpi':        150,
})

CH_COLORS = ['#ef5350', '#81c784', '#4fc3f7']
CH_NAMES  = ['Red channel', 'Green channel', 'Blue channel']
NL        = '\n'   # avoids chr(10) hacks inside f-strings

OUT = 'output_exp02'
os.makedirs(OUT, exist_ok=True)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SimConfig:
    """All physical / sensor constants in one place."""
    photons_per_unit: float = 500.0       # PPU
    dcr_cps:          float = 1_000.0     # dark-count rate (counts/s)
    bin_width_opl:    float = 0.006       # metres of optical-path-length per bin
    start_opl:        float = 1.85        # OPL of first bin (metres)
    temporal_bins:    int   = 300
    spp:              int   = 256         # samples per pixel for Mitsuba
    speed_of_light:   float = 299_792_458.0  # m/s

    # Derived quantities --------------------------------------------------
    @property
    def bin_width_s(self) -> float:
        return self.bin_width_opl / self.speed_of_light

    @property
    def dcr_per_bin(self) -> float:
        return self.dcr_cps * self.bin_width_s

    @property
    def dcr_floor(self) -> float:
        """DCR in radiance units (after dividing by PPU)."""
        return self.dcr_per_bin / self.photons_per_unit

    def opl_axis(self) -> np.ndarray:
        return self.start_opl + self.bin_width_opl * (
            np.arange(self.temporal_bins) + 0.5)


# ── Object / scene definitions ────────────────────────────────────────────────

class ObjectSpec(NamedTuple):
    """Spectral properties of one hidden object + laser combination."""
    label:       str          # short display label (no newlines)
    reflectance: list[float]  # [R, G, B] ∈ [0, 1]
    laser:       list[float]  # [R, G, B] power weights
    color:       str          # hex colour for plots
    desc:        str          # one-line description


OBJECTS: dict[str, ObjectSpec] = {
    'white_red': ObjectSpec(
        label='White Object (red laser)',
        reflectance=[0.90, 0.90, 0.90],
        laser=[1.0, 0.1, 0.1],
        color='#ef9a9a',
        desc='White reflects all λ → strong R, weak G/B return',
    ),
    'red_red': ObjectSpec(
        label='Red Object (red laser)',
        reflectance=[0.85, 0.05, 0.05],
        laser=[1.0, 0.1, 0.1],
        color='#ef5350',
        desc='Red reflects only R → perfect match, strong signal',
    ),
    'blue_red': ObjectSpec(
        label='Blue Object (red laser)',
        reflectance=[0.05, 0.05, 0.85],
        laser=[1.0, 0.1, 0.1],
        color='#5c6bc0',
        desc='Blue absorbs red → near-zero signal → disappears in noise',
    ),
    'leaf_white': ObjectSpec(
        label='Green Leaf (white laser)',
        reflectance=[0.08, 0.75, 0.08],
        laser=[1.0, 1.0, 1.0],
        color='#81c784',
        desc='Leaf absorbs R+B for photosynthesis, reflects G only',
    ),
}


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class RenderResult:
    spec:  ObjectSpec
    clean: np.ndarray   # shape (H, W, T, 3)
    noisy: np.ndarray   # shape (H, W, T, 3)

    def spatial_mean(self, arr: np.ndarray) -> np.ndarray:
        """Collapse spatial dims → (T, 3)."""
        return arr.mean(axis=(0, 1))

    def peak_per_channel(self, arr: np.ndarray | None = None) -> np.ndarray:
        """Max over time per channel, shape (3,)."""
        arr = arr if arr is not None else self.clean
        return self.spatial_mean(arr).max(axis=0)


# ── Physics helpers ───────────────────────────────────────────────────────────

def apply_shot_and_dcr(
    arr: np.ndarray,
    cfg: SimConfig,
    seed: int = 0,
) -> np.ndarray:
    """Add Poisson shot noise and dark-count-rate background."""
    rng      = np.random.default_rng(seed)
    counts   = rng.poisson(np.clip(arr, 0, None) * cfg.photons_per_unit)
    dark     = rng.poisson(cfg.dcr_per_bin, size=arr.shape)
    return (counts + dark).astype(np.float64) / cfg.photons_per_unit


def return_signal(spec: ObjectSpec) -> np.ndarray:
    """Per-channel return signal = laser × reflectance, shape (3,)."""
    return np.array(spec.laser) * np.array(spec.reflectance)


# ── Mitsuba scene builder ─────────────────────────────────────────────────────

def build_nlos_scene(spec: ObjectSpec, cfg: SimConfig) -> dict:
    """Return a Mitsuba scene dict for one object/laser combination."""
    import mitsuba as mi  # deferred — only needed when actually rendering

    T = mi.ScalarTransform4f
    r, g, b = spec.reflectance
    lr, lg, lb = spec.laser

    return {
        'type': 'scene',
        'integrator': {
            'type': 'transient_nlos_path',
            'nlos_laser_sampling':   True,
            'nlos_hidden_geometry_sampling': True,
            'nlos_hidden_geometry_sampling_includes_relay_wall': False,
            'temporal_filter': 'box',
            'max_depth': -1,
        },
        'laser': {
            'type': 'projector',
            'id':   'laser',
            'to_world': T.translate([-0.5, 0.0, 0.25]),
            'irradiance': {'type': 'rgb', 'value': [lr, lg, lb]},
            'fov': 0.2,
        },
        'relay_wall': {
            'type': 'rectangle',
            'id':   'relay_wall',
            'bsdf': {'type': 'diffuse',
                     'reflectance': {'type': 'rgb', 'value': [1, 1, 1]}},
            'sensor': {
                'type': 'nlos_capture_meter',
                'sensor_origin': [-0.5, 0.0, 0.25],
                'sampler': {'type': 'independent',
                            'sample_count': 512, 'seed': 0},
                'film': {
                    'type': 'transient_hdr_film',
                    'width': 16, 'height': 16,
                    'temporal_bins': cfg.temporal_bins,
                    'bin_width_opl': cfg.bin_width_opl,
                    'start_opl':     cfg.start_opl,
                    'rfilter': {'type': 'box'},
                },
            },
        },
        'hidden_object': {
            'type': 'cube',
            'to_world': T.translate([0.0, 0.0, 1.5]).scale(0.3),
            'bsdf': {'type': 'diffuse',
                     'reflectance': {'type': 'rgb', 'value': [r, g, b]}},
        },
    }


def render_all(cfg: SimConfig) -> dict[str, RenderResult]:
    """Render every object and return a dict keyed by OBJECTS keys."""
    import mitsuba as mi
    import mitransient as mitr

    results: dict[str, RenderResult] = {}
    ruler = '═' * 55

    for key, spec in OBJECTS.items():
        print(f'\n{ruler}')
        print(f'RENDERING: {spec.label}')
        print(ruler)

        scene_dict = build_nlos_scene(spec, cfg)
        scene      = mi.load_dict(scene_dict)

        # Focus laser on relay-wall centre
        relay_wall = next(
            (s for s in scene.shapes()
             if hasattr(s, 'sensor') and s.sensor() is not None),
            scene.shapes()[0],
        )
        mitr.nlos.focus_emitter_at_relay_wall_uv(
            mi.Point2f(0.5, 0.5), relay_wall, scene.emitters()[0])

        _, img_t = mi.render(scene, spp=cfg.spp)
        clean    = np.array(img_t, dtype=np.float64)
        noisy    = apply_shot_and_dcr(clean, cfg, seed=42)

        result = RenderResult(spec=spec, clean=clean, noisy=noisy)
        results[key] = result

        peaks = result.peak_per_channel()
        above = peaks > cfg.dcr_floor
        print(f'  Peak per channel  R:{peaks[0]:.5f}  '
              f'G:{peaks[1]:.5f}  B:{peaks[2]:.5f}')
        print(f'  DCR floor       : {cfg.dcr_floor:.5f}')
        print(f'  Above DCR floor : R={above[0]}  G={above[1]}  B={above[2]}')

    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_reflectance_diagram(
    save: str = f'{OUT}/01_reflectance.png',
) -> None:
    """Bar chart: laser power × reflectance = return signal, per object."""
    wl_labels  = ['Red (640 nm)', 'Green (550 nm)', 'Blue (460 nm)']
    fig, axes  = plt.subplots(1, len(OBJECTS), figsize=(18, 5))

    for ax, spec in zip(axes, OBJECTS.values()):
        ref = np.array(spec.reflectance)
        las = np.array(spec.laser)
        ret = return_signal(spec)

        x, w = np.arange(3), 0.25
        ax.bar(x - w, las, w, color=CH_COLORS, alpha=0.5,  label='Laser power')
        ax.bar(x,     ref, w, color=CH_COLORS, alpha=0.8,  label='Reflectance')
        ax.bar(x + w, ret, w, color=CH_COLORS, alpha=1.0,
               edgecolor='white', linewidth=1.2, label='Return signal')

        ax.set_xticks(x)
        ax.set_xticklabels(wl_labels, fontsize=7.5)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Normalized value')
        ax.set_title(spec.label, color=spec.color)
        ax.grid(True, alpha=0.3, axis='y')

        for xi, rv in zip(x, ret):
            ax.text(xi + w, rv + 0.03, f'{rv:.2f}',
                    ha='center', fontsize=7.5, color='white')
        ax.text(0.5, -0.25, spec.desc, transform=ax.transAxes,
                ha='center', fontsize=6.5, color='#aaaacc')

    axes[0].legend(fontsize=7, loc='upper right')
    fig.suptitle(
        'Spectral Reflectance × Laser Power = Return Signal\n'
        'Low return → buried in shot noise + DCR',
        fontsize=12, fontweight='bold', color='white',
    )
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save}')


def plot_channel_comparison(
    results: dict[str, RenderResult],
    cfg: SimConfig,
    save: str = f'{OUT}/02_channel_comparison.png',
) -> None:
    """Per-channel (R, G, B) waveforms for every object."""
    opl      = cfg.opl_axis()
    fig, axes = plt.subplots(len(results), 3, figsize=(18, 4.5 * len(results)))

    for row, result in enumerate(results.values()):
        c_mean = result.spatial_mean(result.clean)   # (T, 3)
        n_mean = result.spatial_mean(result.noisy)

        for col, (ch_color, ch_name) in enumerate(zip(CH_COLORS, CH_NAMES)):
            ax   = axes[row, col]
            peak = c_mean[:, col].max()
            snr  = peak / (cfg.dcr_floor + 1e-12)

            ax.plot(opl, c_mean[:, col], color=ch_color, lw=2.0,
                    label='Clean', alpha=0.9)
            ax.plot(opl, n_mean[:, col], color=ch_color, lw=0.9,
                    linestyle='--', alpha=0.7, label='SPAD noisy')
            ax.axhline(cfg.dcr_floor, color='#ce93d8', lw=1.2,
                       linestyle=':', label=f'DCR floor={cfg.dcr_floor:.5f}')
            ax.fill_between(opl, cfg.dcr_floor, color='#ce93d8', alpha=0.08)
            ax.set_title(
                f'{result.spec.label} — {ch_name}\n'
                f'peak={peak:.4f}  peak/DCR={snr:.1f}×',
                fontsize=8.5, color=ch_color,
            )
            ax.set_xlabel('OPL (m)')
            ax.set_ylabel('Radiance')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Per-Channel Return Signal: Color vs Laser Wavelength\n'
        'Dashed=SPAD output  ·  Purple dotted=DCR noise floor',
        fontsize=13, fontweight='bold', color='white',
    )
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save}')


def plot_peak_signals(
    results: dict[str, RenderResult],
    cfg: SimConfig,
    save: str = f'{OUT}/03_peak_signals.png',
) -> None:
    """Stacked bar of peak signal per channel and SNR vs DCR floor."""
    fig, (ax_bar, ax_snr) = plt.subplots(1, 2, figsize=(14, 6))
    labels = [r.spec.label for r in results.values()]
    x      = np.arange(len(labels))

    # ── Stacked peak bars ──
    bottom = np.zeros(len(labels))
    for ci, (ch_name, ch_color) in enumerate(zip(('R', 'G', 'B'), CH_COLORS)):
        peaks = np.array([r.peak_per_channel()[ci] for r in results.values()])
        ax_bar.bar(x, peaks, bottom=bottom, color=ch_color, alpha=0.8,
                   label=f'{ch_name} channel', edgecolor='none')
        for xi, pv, bv in zip(x, peaks, bottom):
            if pv > 1e-4:
                ax_bar.text(xi, bv + pv / 2, f'{pv:.4f}',
                            ha='center', fontsize=7, color='white',
                            fontweight='bold')
        bottom += peaks

    ax_bar.axhline(cfg.dcr_floor, color='#ce93d8', lw=1.5,
                   linestyle='--', label='DCR floor')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=8)
    ax_bar.set_ylabel('Peak radiance')
    ax_bar.set_title('Peak Return Signal per Channel\n(stacked by RGB)')
    ax_bar.legend()
    ax_bar.grid(True, alpha=0.3, axis='y')

    # ── SNR bars ──
    for ci, (ch_name, ch_color) in enumerate(zip(('R', 'G', 'B'), CH_COLORS)):
        snrs = [
            10 * np.log10(r.peak_per_channel()[ci] / (cfg.dcr_floor + 1e-12) + 1e-6)
            for r in results.values()
        ]
        ax_snr.bar(x + (ci - 1) * 0.25, snrs, 0.25,
                   color=ch_color, alpha=0.8, label=ch_name, edgecolor='none')

    ax_snr.axhline(0, color='#ef5350', lw=1.5, linestyle='--',
                   label='SNR = 0 dB (noise floor)')
    ax_snr.set_xticks(x)
    ax_snr.set_xticklabels(labels, fontsize=8)
    ax_snr.set_ylabel('SNR re. DCR floor (dB)')
    ax_snr.set_title('SNR relative to DCR Noise Floor\nNegative = buried in noise')
    ax_snr.legend(fontsize=8)
    ax_snr.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Peak Signals & SNR — Color vs Laser Wavelength',
                 fontsize=13, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save}')


def save_color_video(
    results: dict[str, RenderResult],
    save: str = f'{OUT}/04_color_comparison.mp4',
) -> None:
    """Side-by-side transient video of all objects (SPAD output)."""
    items  = list(results.values())
    H, W   = items[0].noisy.shape[:2]
    T      = items[0].noisy.shape[2]
    n      = len(items)
    W_tot  = W * n + 2 * (n - 1)

    writer = cv2.VideoWriter(
        save, cv2.VideoWriter_fourcc(*'mp4v'), 24, (W_tot, H + 30))

    # Pre-compute tonemapping scale per result
    def tonemap(arr: np.ndarray) -> np.ndarray:
        v = np.quantile(arr, 0.995)
        return np.clip(arr / (v + 1e-12), 0, 1)

    tonemapped = [tonemap(r.noisy) for r in items]
    opl        = items[0].spec  # placeholder; we recompute below

    for t in range(T):
        canvas = np.zeros((H + 30, W_tot, 3), dtype=np.uint8)
        for pi, (result, tm) in enumerate(zip(items, tonemapped)):
            x0    = pi * (W + 2)
            frame = np.clip(tm[:, :, t, :3], 0, 1)
            canvas[30:, x0:x0 + W] = cv2.cvtColor(
                (frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.putText(canvas, result.spec.label,
                        (x0 + 3, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (200, 200, 255), 1)

        opl_v = 1.85 + 0.006 * (t + 0.5)
        cv2.putText(canvas, f't={t:03d}  OPL={opl_v:.3f}m',
                    (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 255), 1)
        writer.write(canvas)

    writer.release()
    mb = os.path.getsize(save) / 1e6
    print(f'Saved: {save}  ({mb:.2f} MB)')


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    ruler = '═' * 55
    print(f'\n{ruler}\nSUMMARY\n{ruler}')
    for fname in sorted(os.listdir(OUT)):
        sz  = os.path.getsize(f'{OUT}/{fname}')
        val = sz / 1e6 if sz > 1e6 else sz / 1e3
        unit = 'MB' if sz > 1e6 else 'KB'
        print(f'  {fname:<35} {val:6.1f} {unit}')
    print('\nDone.')


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')

    cfg     = SimConfig()
    results = render_all(cfg)

    plot_reflectance_diagram()
    plot_channel_comparison(results, cfg)
    plot_peak_signals(results, cfg)
    save_color_video(results)
    print_summary()