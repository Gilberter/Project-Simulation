# videos_exp02_exp03.py
"""
Generate visualisation videos for Experiment 02 (color) and Experiment 03
(wall material).

Run *after* experiment_02_color.py and experiment_03_wall.py have saved their
respective .npy files.

All shared video-building utilities live in _VideoWriter (below).
The two experiment functions call those helpers; there is no copy-pasted code.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from video_utils import (
    make_canvas, draw_panel, put_text, put_text_centered,
    draw_progress_bar, tonemap, to_bgr_u8, open_writer,
    title_card, report_video,
    WHITE, GRAY, ACCENT, PANEL_BG,
)

# ── Constants shared across both experiments ──────────────────────────────────

FPS = 24
DCR_FLOOR = 1_000.0 * (0.006 / 299_792_458.0) / 500.0   # single definition

MPL_STYLE = {
    'figure.facecolor': '#0f0f1a',
    'axes.facecolor':   '#1a1a2e',
}

CH_COLORS_RGB = [(0.93, 0.32, 0.31), (0.51, 0.78, 0.52), (0.31, 0.76, 0.97)]
CH_NAMES      = ['Red ch', 'Green ch', 'Blue ch']


# ── Shared helpers ────────────────────────────────────────────────────────────

def _fig_to_bgr(fig: Any, w: int, h: int) -> np.ndarray:
    """Render a Matplotlib figure to a BGR uint8 array of size (h, w, 3)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100,
                facecolor='#0f0f1a', bbox_inches='tight')
    buf.seek(0)
    raw = np.frombuffer(buf.read(), np.uint8)
    buf.close()
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return cv2.resize(img, (w, h))


def _opl_axis(n_bins: int, start: float = 1.85, bw: float = 0.006) -> np.ndarray:
    return start + bw * (np.arange(n_bins) + 0.5)


def _styled_ax(ax: Any) -> None:
    """Apply the shared dark theme to a Matplotlib axis."""
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#aaaacc', labelsize=6)
    ax.grid(True, alpha=0.3, color='#2a2a4a')


def _short(name: str) -> str:
    """Return the first line of a (possibly multi-line) name."""
    return name.split('\n')[0]


def _make_theory_slides(
    writer: Any,
    title: str,
    lines: list[str],
    w: int,
    h: int,
    duration_s: float = 5.0,
) -> None:
    """Animated bullet-reveal slide, written directly to an open writer."""
    total = int(FPS * duration_s)
    for fi in range(total):
        canvas = make_canvas(h, w)
        draw_panel(canvas, 20, 20, w - 40, h - 40)
        cv2.rectangle(canvas, (20, 20), (w - 20, 65), (40, 20, 80), -1)
        put_text_centered(canvas, title, 50, 0.50, WHITE, 1)
        visible = min(len(lines), fi // (FPS // 3) + 1)
        for li, line in enumerate(lines[:visible]):
            col = tuple(ACCENT) if (':' in line and not line.startswith(' ')) \
                else tuple(GRAY)
            put_text(canvas, line, 60, 90 + li * 37, 0.37, col, 1)
        draw_progress_bar(canvas, fi, total, width=w - 20)
        writer.write(canvas)


def _waveform_sweep(
    writer: Any,
    results: dict,
    opl: np.ndarray,
    colors_bgr: list[tuple],
    w: int,
    h: int,
    title: str,
    per_channel: bool = False,
) -> np.ndarray:
    """
    Animate waveforms growing from left to right.
    Returns the last rendered frame (for freeze-frames).
    """
    T     = opl.shape[0]
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    for reveal in range(0, T + 4, 4):
        r = min(reveal, T)

        if per_channel:
            names  = list(results.keys())
            fig, axes = plt.subplots(len(names), 3, figsize=(w / 100, h / 100))
            plt.rcParams.update(MPL_STYLE)
            for ri, (name, data) in enumerate(results.items()):
                c = data['clean'].mean(axis=(0, 1))
                n = data['noisy'].mean(axis=(0, 1))
                for ci, (cc, cn) in enumerate(zip(CH_COLORS_RGB, CH_NAMES)):
                    ax = axes[ri, ci]
                    _styled_ax(ax)
                    if r > 0:
                        ax.plot(opl[:r], c[:r, ci], color=cc, lw=1.8)
                        ax.plot(opl[:r], n[:r, ci], color=cc, lw=0.7,
                                linestyle='--', alpha=0.7)
                    ax.axhline(DCR_FLOOR, color='#9c6bce', lw=0.8, linestyle=':')
                    ax.set_xlim(opl[0], opl[-1])
                    if ri == 0:
                        ax.set_title(cn, color=cc, fontsize=8)
                    if ci == 0:
                        ax.set_ylabel(_short(name), color='#ccccee', fontsize=7)
        else:
            fig, ax = plt.subplots(figsize=(w / 100, h / 100))
            _styled_ax(ax)
            for ni, (name, data) in enumerate(results.items()):
                col   = tuple(c / 255 for c in colors_bgr[ni][::-1])
                prof  = data['clean'].mean(axis=(0, 1, -1))
                nprof = data['noisy'].mean(axis=(0, 1, -1))
                if r > 0:
                    ax.plot(opl[:r], prof[:r],  color=col, lw=2.0,
                            label=_short(name))
                    ax.plot(opl[:r], nprof[:r], color=col, lw=0.8,
                            linestyle='--', alpha=0.6)
            ax.axhline(DCR_FLOOR, color='#ce93d8', lw=1.0, linestyle=':',
                       label='DCR floor')
            ax.set_xlim(opl[0], opl[-1])
            ax.set_xlabel('OPL (m)', color='#aaaacc')
            ax.set_ylabel('Mean radiance', color='#aaaacc')
            if r > 0:
                ax.legend(fontsize=8)

        fig.suptitle(f'{title}  [{r}/{T} bins]', color='white', fontsize=10)
        frame = _fig_to_bgr(fig, w, h)
        plt.close(fig)
        draw_progress_bar(frame, r, T, width=w - 20)
        writer.write(frame)

    return frame


def _transient_grid(
    writer: Any,
    results: dict,
    opl: np.ndarray,
    colors_bgr: list[tuple],
    cell_w: int = 200,
    cell_h: int = 200,
) -> None:
    """Side-by-side transient video grid (clean + SPAD rows)."""
    names    = list(results.keys())
    T        = opl.shape[0]
    n        = len(names)
    W        = cell_w * n + 2 * (n - 1)
    H        = cell_h * 2 + 60

    tms_c = [(nm, tonemap(results[nm]['clean'])) for nm in names]
    tms_n = [(nm, tonemap(results[nm]['noisy'])) for nm in names]

    # Resize writer canvas to match grid — caller must open writer at (W, H)
    for t in range(T):
        canvas = make_canvas(H, W)
        for pi, ((nm, c_tm), (_, n_tm)) in enumerate(zip(tms_c, tms_n)):
            x0  = pi * (cell_w + 2)
            col = colors_bgr[pi]
            for ri, (arr_tm, lbl) in enumerate([(c_tm, 'CLEAN'), (n_tm, 'SPAD')]):
                y0 = 30 + ri * (cell_h + 2)
                fr = cv2.resize(
                    to_bgr_u8(np.clip(arr_tm[:, :, t, :3], 0, 1)),
                    (cell_w, cell_h))
                canvas[y0:y0 + cell_h, x0:x0 + cell_w] = fr
                put_text(canvas, lbl, x0 + 3, y0 - 3, 0.28, col, 1)
            put_text(canvas, _short(nm), x0 + 3, 20, 0.30, col, 1)

        opl_v = opl[t]
        put_text(canvas, f't={t:03d} OPL={opl_v:.3f}m',
                 4, 12, 0.30, tuple(GRAY), 1)
        draw_progress_bar(canvas, t, T, y0=H - 8, width=W - 20)
        writer.write(canvas)


# ── Experiment 02 videos ──────────────────────────────────────────────────────

def make_exp02_videos(results: dict, out: str = '/home/hensemberk/dev/project_simulation/Experiments/output_exp02') -> None:
    os.makedirs(out, exist_ok=True)
    names      = list(results.keys())
    T          = list(results.values())[0]['clean'].shape[2]
    opl        = _opl_axis(T)
    colors_bgr = [
        (80, 180, 239),   # white object
        (80,  83, 239),   # red object
        (239, 130, 80),   # blue object
        (132, 199, 129),  # green leaf
    ]

    # VE2-1: Title + theory ──────────────────────────────────────────────────
    w, h   = 960, 540
    writer = open_writer(f'{out}/V1_color_intro.mp4', w, h, FPS)
    title_card(writer,
               'EXPERIMENT 2: HIDDEN OBJECT COLOR',
               'Does a red laser "see" a blue object?',
               w=w, h=h, duration_s=3.0, fps=FPS)
    _make_theory_slides(
        writer, 'Spectral Reflectance Theory',
        [
            'Spectral Reflectance R(λ):',
            '  Return signal = Laser(λ) x R_object(λ)',
            '',
            'Red laser + Blue object:',
            '  Blue absorbs red photons → near-zero return',
            '  Signal disappears below DCR noise floor',
            '',
            'White laser + Green leaf:',
            '  Leaf absorbs R+B (photosynthesis)',
            '  Only G channel shows return signal',
        ],
        w, h,
    )
    writer.release()
    report_video(f'{out}/V1_color_intro.mp4')

    # VE2-2: Per-channel waveform animation ──────────────────────────────────
    w, h   = 960, 600
    writer = open_writer(f'{out}/V2_channel_waveforms.mp4', w, h, FPS)
    frame  = _waveform_sweep(writer, results, opl, colors_bgr,
                              w, h, 'Per-channel waveforms', per_channel=True)
    for _ in range(FPS * 2):
        writer.write(frame)
    writer.release()
    report_video(f'{out}/V2_channel_waveforms.mp4')

    # VE2-3: Transient grid ──────────────────────────────────────────────────
    cell_w, cell_h = 200, 200
    n    = len(results)
    W    = cell_w * n + 2 * (n - 1)
    H    = cell_h * 2 + 60
    writer = open_writer(f'{out}/V3_transient_grid.mp4', W, H, FPS)
    _transient_grid(writer, results, opl, colors_bgr, cell_w, cell_h)
    writer.release()
    report_video(f'{out}/V3_transient_grid.mp4')

    # VE2-4: Blue object DCR reveal ──────────────────────────────────────────
    w, h      = 960, 420
    writer    = open_writer(f'{out}/V4_dcr_floor.mp4', w, h, FPS)
    zoom_key  = names[2]   # blue object
    zm_clean  = results[zoom_key]['clean'].mean(axis=(0, 1))
    zm_noisy  = results[zoom_key]['noisy'].mean(axis=(0, 1))
    total     = FPS * 6

    for fi in range(total):
        show_dcr = fi > FPS * 2
        xlim     = (opl[100], opl[200]) if fi > FPS * 4 else (opl[0], opl[-1])

        fig, ax = plt.subplots(figsize=(9.4, 3.8))
        fig.patch.set_facecolor('#0f0f1a')
        _styled_ax(ax)
        ax.plot(opl, zm_clean.mean(-1), color='#4fc3f7', lw=2.0,
                label='Clean (all channels)')
        ax.plot(opl, zm_noisy.mean(-1), color='#ef5350', lw=0.9,
                alpha=0.8, label='SPAD noisy')

        if show_dcr:
            ax.axhline(DCR_FLOOR, color='#ce93d8', lw=2.0, linestyle='--',
                       label=f'DCR floor = {DCR_FLOOR:.5f}')
            ax.fill_between(opl, DCR_FLOOR, color='#ce93d8', alpha=0.12)
            ax.text(opl[10], DCR_FLOOR * 1.5,
                    'Blue object signal < DCR floor\n→ disappears in noise',
                    color='#ce93d8', fontsize=9)

        ax.set_xlim(*xlim)
        ax.set_xlabel('OPL (m)', color='#aaaacc')
        ax.set_ylabel('Radiance', color='#aaaacc')
        ax.set_title('Blue Object + Red Laser → Signal buried in DCR',
                     color='white', fontsize=10)
        ax.legend(fontsize=8)

        frame = _fig_to_bgr(fig, w, h)
        plt.close(fig)
        draw_progress_bar(frame, fi, total, width=w - 20)
        writer.write(frame)

    writer.release()
    report_video(f'{out}/V4_dcr_floor.mp4')
    print(f'\nAll Exp02 videos saved to {out}/')


# ── Experiment 03 videos ──────────────────────────────────────────────────────

def make_exp03_videos(results: dict, out: str = '/home/hensemberk/dev/project_simulation/Experiments/output_exp03') -> None:
    os.makedirs(out, exist_ok=True)
    names      = list(results.keys())
    T          = list(results.values())[0]['clean'].shape[2]
    opl        = _opl_axis(T)
    colors_bgr = [(247, 195, 79), (79, 183, 255), (80, 83, 239)]

    # VE3-1: Title + WOSR theory ─────────────────────────────────────────────
    w, h   = 960, 540
    writer = open_writer(f'{out}/V1_wall_intro.mp4', w, h, FPS)
    title_card(writer,
               'EXPERIMENT 3: MIRROR vs MATTE RELAY WALL',
               'How wall material affects hidden object visibility',
               w=w, h=h, duration_s=3.0, fps=FPS)
    _make_theory_slides(
        writer, 'BSDF & Wall-to-Object Signal Ratio',
        [
            'Relay Wall BSDF controls scatter pattern:',
            '',
            '  Diffuse (matte):  energy spread uniformly',
            '    → broad low peak from wall',
            '    → hidden object signal comparable',
            '',
            '  Specular (shiny): sharp mirror reflection',
            '    → massive early "glint" peak at wall OPL',
            '    → hidden object buried below glint',
            '',
            'Metric: WOSR = wall_peak / object_peak',
            '  High WOSR = harder reconstruction',
        ],
        w, h, duration_s=6.0,
    )
    writer.release()
    report_video(f'{out}/V1_wall_intro.mp4')

    # VE3-2: Waveform sweep ──────────────────────────────────────────────────
    w, h   = 960, 420
    writer = open_writer(f'{out}/V2_waveform_sweep.mp4', w, h, FPS)
    frame  = _waveform_sweep(writer, results, opl, colors_bgr,
                              w, h, 'Relay Wall Material Comparison')
    for _ in range(FPS * 2):
        writer.write(frame)
    writer.release()
    report_video(f'{out}/V2_waveform_sweep.mp4')

    # VE3-3: WOSR animated bar chart ─────────────────────────────────────────
    w, h   = 960, 480
    writer = open_writer(f'{out}/V3_wosr_animated.mp4', w, h, FPS)

    wosrs = []
    for data in results.values():
        prof = data['clean'].mean(axis=(0, 1, -1))
        wosrs.append(prof[:80].max() / (prof[80:].max() + 1e-12))
    wosr_max = max(wosrs) * 1.15
    frames_per_bar = 35

    for bi, (name, wosr) in enumerate(zip(names, wosrs)):
        for fi in range(frames_per_bar):
            vals = [
                wosrs[i] if i < bi else (wosr * fi / frames_per_bar if i == bi else 0)
                for i in range(len(names))
            ]
            cols = [tuple(c / 255 for c in colors_bgr[i][::-1])
                    for i in range(len(names))]

            fig, ax = plt.subplots(figsize=(9.4, 4.2))
            fig.patch.set_facecolor('#0f0f1a')
            _styled_ax(ax)
            ax.bar(range(len(names)), vals, color=cols, alpha=0.85,
                   edgecolor='none')
            ax.axhline(1, color='#81c784', lw=1.5, linestyle='--',
                       label='WOSR=1 (equal)')
            for i, bv in enumerate(vals):
                if bv > 0.5:
                    ax.text(i, bv + 0.2, f'{bv:.1f}×',
                            ha='center', color='white',
                            fontsize=10, fontweight='bold')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([_short(n) for n in names],
                               color='#aaaacc', fontsize=8)
            ax.set_ylim(0, wosr_max)
            ax.set_ylabel('WOSR (Wall/Object ratio)', color='#aaaacc')
            ax.set_title('Wall-to-Object Signal Ratio\n'
                         'Higher = specular glint dominates',
                         color='white', fontsize=10)
            ax.legend(fontsize=8)

            frame = _fig_to_bgr(fig, w, h)
            plt.close(fig)
            writer.write(frame)

    for _ in range(FPS * 2):
        writer.write(frame)
    writer.release()
    report_video(f'{out}/V3_wosr_animated.mp4')

    # VE3-4: Side-by-side transient grid ─────────────────────────────────────
    cell_w, cell_h = 256, 256
    n    = len(results)
    W    = cell_w * n + 2 * (n - 1)
    H    = cell_h * 2 + 50
    writer = open_writer(f'{out}/V4_material_grid.mp4', W, H, FPS)
    _transient_grid(writer, results, opl, colors_bgr, cell_w, cell_h)
    writer.release()
    report_video(f'{out}/V4_material_grid.mp4')

    print(f'\nAll Exp03 videos saved to {out}/')


# ── Entry point ───────────────────────────────────────────────────────────────

def _load_results(
    base_path: str,
    names: list[str],
) -> dict[str, dict]:
    """Load clean/noisy .npy pairs, return only those that exist."""
    results = {}
    for name in names:
        key = _short(name).replace(' ', '_').lower()
        cp  = f'{base_path}/clean_{key}.npy'
        np_ = f'{base_path}/noisy_{key}.npy'
        if os.path.exists(cp) and os.path.exists(np_):
            results[name] = {
                'clean': np.load(cp).astype(np.float64),
                'noisy': np.load(np_).astype(np.float64),
            }
    return results


if __name__ == '__main__':
    # ── Experiment 02 ──
    print('Loading Exp02 results...')
    r02 = _load_results(
        'output_exp02',
        [
            'White Object\n(red laser)',
            'Red Object\n(red laser)',
            'Blue Object\n(red laser)',
            'Green Leaf\n(white laser)',
        ],
    )
    if r02:
        make_exp02_videos(r02)
    else:
        print('  No Exp02 .npy files found — run experiment_02_color.py first')

    # ── Experiment 03 ──
    print('\nLoading Exp03 results...')
    r03 = _load_results(
        'output_exp03',
        [
            'Diffuse\n(matte drywall)',
            'Rough Plastic\n(shiny paint)',
            'Smooth Plastic\n(glossy)',
        ],
    )
    if r03:
        make_exp03_videos(r03)
    else:
        print('  No Exp03 .npy files found — run experiment_03_wall.py first')

    print('\nDone.')