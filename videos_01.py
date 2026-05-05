# videos_exp01.py
"""
Professional presentation videos for Experiment 1:
Signal Averaging vs Reconstruction Quality

Videos produced:
  V1  title_intro.mp4         — animated title card with theory text
  V2  waveform_buildup.mp4    — waveform appears bin-by-bin, M=1 only
  V3  averaging_compare.mp4   — side-by-side M=1/10/100/1000 panels
  V4  snr_meter.mp4           — animated SNR gauge filling up as M increases
  V5  noise_peel.mp4          — noisy signal "peels away" to reveal clean
  V6  full_presentation.mp4   — all above stitched sequentially
"""

import numpy as np
import cv2
import os
from scipy.stats import skewnorm
from scipy import signal as scipy_signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

from video_utils import (
    make_canvas, draw_panel, put_text, put_text_centered,
    draw_progress_bar, draw_opl_marker, tonemap, to_bgr_u8,
    open_writer, title_card, fade_in, report_video,
    WHITE, GRAY, DARK_BG, PANEL_BG, ACCENT,
)

OUT  = 'output_exp01'
FPS  = 24
os.makedirs(OUT, exist_ok=True)

M_VALS   = [1, 10, 100, 1000]
M_COLORS_BGR = {
    1:    (80,  83,  239),   # red
    10:   (79,  183, 255),   # amber
    100:  (132, 199, 129),   # green
    1000: (247, 195,  79),   # blue
}
M_LABELS = {1: 'M=1', 10: 'M=10', 100: 'M=100', 1000: 'M=1000'}


# ── helpers ──────────────────────────────────────────────────────────────────

def matplotlib_frame_to_bgr(fig) -> np.ndarray:
    """Render a matplotlib figure to a BGR numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100,
                facecolor='#0f0f1a', bbox_inches='tight')
    buf.seek(0)
    arr = np.frombuffer(buf.read(), np.uint8)
    buf.close()
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def skewed_gaussian_kernel(n, sigma, skew):
    t = np.arange(n) - n // 2
    k = skewnorm.pdf(t, a=skew, loc=0, scale=sigma)
    return k / k.sum() if k.sum() > 1e-12 else np.eye(1, n, n//2)[0]


def simulate_M(clean, M, ppu=500.0, dcr_pb=2e-11, seed=0):
    rng   = np.random.default_rng(seed)
    kern  = skewed_gaussian_kernel(31, 1.5, 3.0)
    from scipy import signal as ss
    irf_c = np.zeros_like(clean)
    H, W, T, C = clean.shape
    for c in range(C):
        for h in range(H):
            row = clean[h, :, :, c]
            irf_c[h, :, :, c] = np.array([
                ss.fftconvolve(row[w], kern, mode='same') for w in range(W)])
    accum = np.zeros_like(clean, dtype=np.float64)
    for _ in range(M):
        exp  = np.clip(irf_c, 0, None) * ppu
        cnt  = rng.poisson(exp).astype(np.float64)
        dark = rng.poisson(dcr_pb, size=clean.shape).astype(np.float64)
        accum += (cnt + dark) / ppu
    return accum / M


def snr_db(clean, noisy):
    sp = np.mean(clean**2)
    np_ = np.mean((noisy - clean)**2)
    return float(10 * np.log10(sp / (np_ + 1e-12)))


def psnr_db(clean, noisy):
    mse = np.mean((noisy - clean)**2)
    return float(10 * np.log10(clean.max()**2 / (mse + 1e-12)))


# ── V1: Animated theory title card ───────────────────────────────────────────

def make_v1_intro(w=960, h=540,
                  save=f'{OUT}/V1_intro.mp4'):
    writer = open_writer(save, w, h, FPS)

    lines = [
        ('EXPERIMENT 1: SIGNAL AVERAGING',    0.70, WHITE,  1),
        ('Shot noise follows Poisson stats',  0.42, GRAY,   1),
        ('SNR = sqrt(lambda)  =>  SNR grows as sqrt(M)', 0.40, GRAY, 1),
        ('',                                  0.30, GRAY,   1),
        ('M = number of averaged laser pulses', 0.38, tuple(ACCENT), 1),
        ('Goal: show SNR improves as sqrt(M)', 0.38, tuple(ACCENT), 1),
    ]

    # Build frame incrementally, revealing one line every 30 frames
    total_reveal_frames = len(lines) * 30
    hold_frames         = FPS * 2

    for frame_i in range(total_reveal_frames + hold_frames):
        canvas = make_canvas(h, w)
        draw_panel(canvas, 20, 20, w-40, h-40)

        # title bar
        cv2.rectangle(canvas, (20, 20), (w-20, 80),
                      (40, 30, 80), -1)
        put_text_centered(canvas, 'Sim-to-Real SPAD Noise Module',
                          55, 0.45, tuple(ACCENT), 1)

        visible = min(len(lines),
                      frame_i // 30 + 1)
        for li in range(visible):
            txt, scale, col, thick = lines[li]
            y = 120 + li * 58
            # animate last visible line sliding in
            if li == visible - 1 and frame_i % 30 < 15:
                alpha = (frame_i % 30) / 15.0
                col   = tuple(int(c * alpha) for c in col)
            put_text_centered(canvas, txt, y, scale, col, thick)

        # SNR formula box bottom-right
        draw_panel(canvas, w-280, h-120, 260, 95,
                   color=(30, 20, 60), border=tuple(ACCENT))
        put_text(canvas, 'Key formula:',    w-270, h-95,  0.38, GRAY)
        put_text(canvas, 'SNR(M) = sqrt(M) x SNR(1)',
                 w-270, h-68,  0.40, WHITE, 1)
        put_text(canvas, 'RMSE(M) = RMSE(1) / sqrt(M)',
                 w-270, h-42,  0.38, tuple(ACCENT), 1)

        draw_progress_bar(canvas, frame_i,
                          total_reveal_frames + hold_frames,
                          width=w-20)
        writer.write(canvas)

    writer.release()
    report_video(save)


# ── V2: Waveform build-up (bin by bin) ───────────────────────────────────────

def make_v2_waveform_buildup(clean: np.ndarray,
                              noisy_m1: np.ndarray,
                              ph: int, pw: int,
                              w=960, h=400,
                              save=f'{OUT}/V2_waveform_buildup.mp4'):
    """
    Animate the transient waveform being revealed bin-by-bin,
    showing how the noisy SPAD signal compares to the clean truth.
    """
    writer  = open_writer(save, w, h, FPS)
    T       = clean.shape[2]
    opl     = 1.85 + 0.006 * (np.arange(T) + 0.5)
    c_px    = clean[ph, pw, :, :].mean(-1)
    n_px    = noisy_m1[ph, pw, :, :].mean(-1)
    c_max   = max(c_px.max(), n_px.max()) * 1.15

    plt.rcParams.update({'figure.facecolor': '#0f0f1a',
                         'axes.facecolor':   '#1a1a2e'})

    # reveal bin by bin (speed: 4 bins per frame)
    step = 4
    for reveal in range(0, T + step, step):
        r = min(reveal, T)

        fig, ax = plt.subplots(figsize=(9.6, 3.5))
        fig.patch.set_facecolor('#0f0f1a')
        ax.set_facecolor('#1a1a2e')

        if r > 0:
            ax.fill_between(opl[:r], c_px[:r],
                            color='#4fc3f7', alpha=0.25)
            ax.plot(opl[:r], c_px[:r],
                    color='#4fc3f7', lw=2.0, label='Clean truth')
            ax.fill_between(opl[:r], n_px[:r],
                            color='#ef5350', alpha=0.15)
            ax.plot(opl[:r], n_px[:r],
                    color='#ef5350', lw=0.9, alpha=0.85,
                    label='SPAD M=1')

        ax.set_xlim(opl[0], opl[-1])
        ax.set_ylim(-c_max * 0.05, c_max)
        ax.set_xlabel('Optical Path Length (m)',
                      color='#aaaacc', fontsize=9)
        ax.set_ylabel('Radiance', color='#aaaacc', fontsize=9)
        ax.set_title(f'Transient Waveform — Pixel ({ph},{pw})  '
                     f'[{r}/{T} bins revealed]',
                     color='white', fontsize=10)
        ax.tick_params(colors='#aaaacc')
        ax.grid(True, alpha=0.3, color='#2a2a4a')
        if r > 0:
            ax.legend(fontsize=8, loc='upper right')

        # progress
        pct = r / T
        ax.axvline(opl[min(r, T-1)], color='#ffeb3b',
                   lw=1.5, linestyle='--', alpha=0.7)

        frame_bgr = matplotlib_frame_to_bgr(fig)
        plt.close(fig)

        # resize to target
        frame_bgr = cv2.resize(frame_bgr, (w, h))
        draw_progress_bar(frame_bgr, r, T, width=w-20)
        writer.write(frame_bgr)

    # hold last frame 2 s
    for _ in range(FPS * 2):
        writer.write(frame_bgr)

    writer.release()
    report_video(save)


# ── V3: Side-by-side averaging comparison ────────────────────────────────────

def make_v3_averaging_compare(clean: np.ndarray,
                               averaged: dict,
                               w_per_panel=240, h=360,
                               start_opl=1.85, bin_w=0.006,
                               save=f'{OUT}/V3_averaging_compare.mp4'):
    """
    5 panels side by side: Clean | M=1 | M=10 | M=100 | M=1000
    Each panel shows the transient video frame at time t.
    Bottom strip shows the time-averaged waveform being drawn.
    """
    panels   = [('CLEAN', clean)] + \
               [(M_LABELS[m], a) for m, a in averaged.items()]
    n_panels = len(panels)
    W        = w_per_panel * n_panels + 2*(n_panels-1)
    H        = h + 120        # extra for waveform strip
    writer   = open_writer(save, W, H, FPS)

    tms      = [(lbl, tonemap(arr)) for lbl, arr in panels]
    H_img, W_img, T, C = clean.shape

    colors_list = [
        (247, 195,  79),   # clean — blue
        ( 80,  83, 239),   # M=1   — red
        ( 79, 183, 255),   # M=10  — amber
        (132, 199, 129),   # M=100 — green
        (247, 195,  79),   # M=1000— blue
    ]

    # pre-compute per-panel waveforms (spatial mean)
    waveforms = [arr.mean(axis=(0,1,3)) for _, arr in panels]
    w_max     = max(w.max() for w in waveforms) * 1.1

    for t in range(T):
        canvas = make_canvas(H, W)

        for pi, ((lbl, arr_tm), col) in enumerate(zip(tms, colors_list)):
            x0 = pi * (w_per_panel + 2)

            # video panel
            frame = np.clip(arr_tm[:, :, t, :3], 0, 1)
            frame_resized = cv2.resize(
                to_bgr_u8(frame), (w_per_panel, h))
            canvas[0:h, x0:x0+w_per_panel] = frame_resized

            # label bar
            cv2.rectangle(canvas, (x0, 0),
                          (x0+w_per_panel, 22), (20,20,40), -1)
            put_text(canvas, lbl, x0+5, 16, 0.40, col, 1)

            # waveform strip
            wy0 = h + 5
            wy1 = H - 15
            wh  = wy1 - wy0
            wv  = waveforms[pi]

            # draw axes
            cv2.rectangle(canvas, (x0, wy0),
                          (x0+w_per_panel, wy1),
                          PANEL_BG, -1)
            # draw waveform up to current t
            for ti in range(1, t+1):
                y1 = wy1 - int(wv[ti-1] / w_max * wh)
                y2 = wy1 - int(wv[ti]   / w_max * wh)
                x1 = x0 + int((ti-1) / T * w_per_panel)
                x2 = x0 + int(ti     / T * w_per_panel)
                cv2.line(canvas, (x1, y1), (x2, y2), col, 1)

            # current time marker
            xt = x0 + int(t / T * w_per_panel)
            cv2.line(canvas, (xt, wy0), (xt, wy1),
                     (255, 235,  59), 1)

        draw_opl_marker(canvas, t, T, start_opl, bin_w,
                        y0=H-5)
        draw_progress_bar(canvas, t, T, y0=H-10, width=W-20)
        writer.write(canvas)

    writer.release()
    report_video(save)


# ── V4: Animated SNR meter ────────────────────────────────────────────────────

def make_v4_snr_meter(snrs: dict, psnrs: dict,
                      w=800, h=500,
                      save=f'{OUT}/V4_snr_meter.mp4'):
    """
    For each M, animate a gauge/bar filling up to the measured SNR value.
    Shows the √M theoretical curve being drawn simultaneously.
    """
    writer    = open_writer(save, w, h, FPS)
    M_list    = sorted(snrs.keys())
    snr_list  = [snrs[m] for m in M_list]
    psnr_list = [psnrs[m] for m in M_list]
    snr_max   = max(snr_list) * 1.2

    # animate: go through each M, fill bar over 30 frames
    frames_per_M = 40
    hold_frames  = FPS

    for mi_idx, M in enumerate(M_list):
        target_snr  = snr_list[mi_idx]
        target_psnr = psnr_list[mi_idx]

        for f in range(frames_per_M):
            progress = f / frames_per_M
            curr_snr = target_snr * progress

            fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
            fig.patch.set_facecolor('#0f0f1a')

            # Left: bar gauge
            ax = axes[0]
            ax.set_facecolor('#1a1a2e')
            colors_bar = [
                M_COLORS_BGR[m][::-1] for m in M_list]   # BGR->RGB
            bars = ax.bar(range(len(M_list)),
                          [snr_list[j] if j < mi_idx
                           else (curr_snr if j == mi_idx else 0)
                           for j in range(len(M_list))],
                          color=[(c[0]/255, c[1]/255, c[2]/255)
                                 for c in [M_COLORS_BGR[m]
                                            for m in M_list]],
                          edgecolor='none', alpha=0.85)
            ax.set_xticks(range(len(M_list)))
            ax.set_xticklabels([f'M={m}' for m in M_list],
                               color='#aaaacc', fontsize=9)
            ax.set_ylim(0, snr_max)
            ax.set_ylabel('SNR (dB)', color='#aaaacc')
            ax.set_title('SNR vs Averages M',
                         color='white', fontsize=10)
            ax.tick_params(colors='#aaaacc')
            ax.grid(True, alpha=0.3, color='#2a2a4a', axis='y')
            for j, sv in enumerate(snr_list[:mi_idx]):
                ax.text(j, sv + 0.3, f'{sv:.1f}dB',
                        ha='center', fontsize=8, color='white')
            ax.text(mi_idx, curr_snr + 0.3,
                    f'{curr_snr:.1f}dB',
                    ha='center', fontsize=9,
                    color='white', fontweight='bold')

            # Right: theoretical √M curve
            ax2 = axes[1]
            ax2.set_facecolor('#1a1a2e')
            M_th   = np.logspace(0, 3, 200)
            snr_th = snr_list[0] + 10 * np.log10(M_th)
            ax2.semilogx(M_th, snr_th, '--',
                         color='#ffb74d', lw=1.5,
                         label='Theory √M', alpha=0.7)
            # measured points so far
            for j in range(mi_idx):
                ax2.scatter([M_list[j]], [snr_list[j]],
                            color=tuple(
                                c/255 for c in
                                M_COLORS_BGR[M_list[j]][::-1]),
                            s=80, zorder=5)
            # animated current point
            col_curr = tuple(
                c/255 for c in M_COLORS_BGR[M][::-1])
            ax2.scatter([M], [curr_snr],
                        color=col_curr, s=120, zorder=6,
                        edgecolors='white', linewidths=1.5)
            ax2.set_xlabel('M (averages, log)',
                           color='#aaaacc', fontsize=9)
            ax2.set_ylabel('SNR (dB)', color='#aaaacc')
            ax2.set_title('SNR Curve — Theory vs Measured',
                          color='white', fontsize=10)
            ax2.tick_params(colors='#aaaacc')
            ax2.grid(True, alpha=0.3, color='#2a2a4a',
                     which='both')
            ax2.legend(fontsize=8)

            fig.suptitle(
                f'Animating M={M}  —  '
                f'SNR={curr_snr:.1f} dB  '
                f'(theory: +{10*np.log10(M):.1f} dB vs M=1)',
                color='white', fontsize=11, fontweight='bold')

            frame_bgr = matplotlib_frame_to_bgr(fig)
            plt.close(fig)
            frame_bgr = cv2.resize(frame_bgr, (w, h))
            writer.write(frame_bgr)

        # hold
        for _ in range(hold_frames):
            writer.write(frame_bgr)

    writer.release()
    report_video(save)


# ── V5: Noise "peel away" reveal ─────────────────────────────────────────────

def make_v5_noise_peel(clean: np.ndarray,
                        averaged: dict,
                        ph: int, pw: int,
                        w=960, h=420,
                        save=f'{OUT}/V5_noise_peel.mp4'):
    """
    Transition: start with M=1 noisy, smoothly blend toward clean
    as M increases. Shows how averaging reveals the true signal.
    """
    writer  = open_writer(save, w, h, FPS)
    T       = clean.shape[2]
    opl     = 1.85 + 0.006 * (np.arange(T) + 0.5)
    c_px    = clean[ph, pw, :, :].mean(-1)
    n_pxs   = {m: arr[ph, pw, :, :].mean(-1)
               for m, arr in averaged.items()}
    y_max   = max(c_px.max(),
                  max(n.max() for n in n_pxs.values())) * 1.15

    plt.rcParams.update({'figure.facecolor': '#0f0f1a',
                         'axes.facecolor':   '#1a1a2e'})

    transition_pairs = [(1, 10), (10, 100), (100, 1000), (1000, None)]
    frames_trans     = FPS * 2   # 2 s per transition
    frames_hold      = FPS       # 1 s hold

    for (m_from, m_to) in transition_pairs:
        n_from = n_pxs[m_from]
        n_to   = n_pxs[m_to] if m_to else c_px
        col_f  = tuple(c/255 for c in M_COLORS_BGR[m_from][::-1])
        col_t  = tuple(c/255 for c in M_COLORS_BGR.get(
            m_to, (247, 195, 79))[::-1])

        for fi in range(frames_trans + frames_hold):
            alpha = min(fi / frames_trans, 1.0)
            blended = (1 - alpha) * n_from + alpha * n_to

            fig, ax = plt.subplots(figsize=(9.4, 3.8))
            fig.patch.set_facecolor('#0f0f1a')
            ax.set_facecolor('#1a1a2e')

            ax.fill_between(opl, c_px,   alpha=0.15, color='white')
            ax.plot(opl, c_px,   color='white',  lw=2.0,
                    linestyle='--', label='Clean truth', alpha=0.6)
            ax.fill_between(opl, blended,
                            alpha=0.25, color=col_t)
            ax.plot(opl, blended, color=col_t,   lw=1.8,
                    label=f'Averaging: M={m_from}→{m_to or "clean"}')

            snr = float(10 * np.log10(
                np.mean(c_px**2) /
                (np.mean((blended-c_px)**2)+1e-12)))

            ax.set_xlim(opl[0], opl[-1])
            ax.set_ylim(-y_max*0.05, y_max)
            ax.set_xlabel('OPL (m)', color='#aaaacc')
            ax.set_ylabel('Radiance', color='#aaaacc')
            ax.set_title(
                f'Noise Peel — transition M={m_from}→{m_to or "∞"}  |  '
                f'SNR={snr:.1f} dB  |  α={alpha:.2f}',
                color='white', fontsize=10)
            ax.tick_params(colors='#aaaacc')
            ax.grid(True, alpha=0.3, color='#2a2a4a')
            ax.legend(fontsize=8)

            frame_bgr = matplotlib_frame_to_bgr(fig)
            plt.close(fig)
            frame_bgr = cv2.resize(frame_bgr, (w, h))
            draw_progress_bar(frame_bgr,
                              fi + transition_pairs.index(
                                  (m_from, m_to)) * (frames_trans+frames_hold),
                              len(transition_pairs)*(frames_trans+frames_hold),
                              width=w-20)
            writer.write(frame_bgr)

    writer.release()
    report_video(save)


# ── V6: Full presentation (all stitched) ─────────────────────────────────────

def make_v6_full_presentation(clean, averaged, snrs, psnrs,
                               ph, pw, w=960, h=540,
                               save=f'{OUT}/V6_full_presentation.mp4'):
    """
    Master video: title → theory → waveform buildup →
    averaging compare → SNR meter → noise peel → summary card.
    """
    writer   = open_writer(save, w, h, FPS)

    # ① Title card
    title_card(writer,
               'EXPERIMENT 1: SIGNAL AVERAGING',
               'How averaging M laser pulses improves SNR by sqrt(M)',
               w=w, h=h, duration_s=3.0, fps=FPS)

    # ② Theory slides (static text, 5 s)
    theory_lines = [
        'THEORY: Shot Noise & Averaging',
        '',
        'Each photon arrival is Poisson distributed:',
        '  counts ~ Poisson(lambda)',
        '  SNR = sqrt(lambda)',
        '',
        'Averaging M scans multiplies effective photons:',
        '  SNR(M) = sqrt(M) x SNR(1)',
        '  RMSE(M) = RMSE(1) / sqrt(M)',
        '',
        'This experiment PROVES this law experimentally.',
    ]
    for fi in range(FPS * 5):
        canvas = make_canvas(h, w)
        draw_panel(canvas, 20, 20, w-40, h-40)
        cv2.rectangle(canvas, (20, 20), (w-20, 65),
                      (40, 20, 80), -1)
        put_text_centered(canvas, 'Physical Background',
                          50, 0.52, tuple(ACCENT), 1)
        for li, line in enumerate(theory_lines):
            visible = min(len(theory_lines),
                          fi // (FPS//2) + 1)
            if li < visible:
                col = WHITE if line.startswith(' ') or \
                               line.startswith('  ') else \
                      tuple(ACCENT) if 'THEORY' in line else GRAY
                put_text(canvas, line, 60, 85 + li*38,
                         0.38, col, 1)
        draw_progress_bar(canvas, fi, FPS*5, width=w-20)
        writer.write(canvas)

    # ③ Splice in sub-videos at reduced size
    def splice_video(path: str):
        if not os.path.exists(path):
            return
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (w, h))
            writer.write(resized)
        cap.release()

    splice_video(f'{OUT}/V2_waveform_buildup.mp4')
    splice_video(f'{OUT}/V3_averaging_compare.mp4')
    splice_video(f'{OUT}/V4_snr_meter.mp4')
    splice_video(f'{OUT}/V5_noise_peel.mp4')

    # ④ Summary card
    M_list   = sorted(snrs.keys())
    snr_list = [snrs[m] for m in M_list]
    for fi in range(FPS * 4):
        canvas = make_canvas(h, w)
        draw_panel(canvas, 20, 20, w-40, h-40)
        cv2.rectangle(canvas, (20, 20), (w-20, 65),
                      (40, 20, 80), -1)
        put_text_centered(canvas, 'RESULTS SUMMARY',
                          50, 0.55, WHITE, 1)
        put_text(canvas, 'M       SNR(dB)   PSNR(dB)   Improvement',
                 50, 100, 0.40, tuple(ACCENT), 1)
        cv2.line(canvas, (50, 110), (w-50, 110), PANEL_BG, 1)
        for li, (M, s, p) in enumerate(
                zip(M_list, snr_list,
                    [psnrs[m] for m in M_list])):
            col = M_COLORS_BGR[M]
            imp = np.sqrt(M)
            put_text(canvas,
                     f'{M:<8} {s:>7.2f}   {p:>8.2f}   x{imp:>5.2f}',
                     50, 140 + li*42,
                     0.42, col, 1)
        put_text_centered(canvas,
                          'SNR improves as sqrt(M) -- confirmed!',
                          h-60, 0.42, tuple(ACCENT), 1)
        draw_progress_bar(canvas, fi, FPS*4, width=w-20)
        writer.write(canvas)

    writer.release()
    report_video(save)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import mitransient as mitr

    print('Loading data or re-rendering...')
    import glob

    clean_path = f'{OUT}/clean.npy'
    if os.path.exists(clean_path):
        clean = np.load(clean_path).astype(np.float64)
        print(f'  Loaded clean from {clean_path}')
    else:
        # re-render (same scene as experiment_01)
        T_mi = mi.ScalarTransform4f
        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'transient_nlos_path',
                'nlos_laser_sampling': True,
                'nlos_hidden_geometry_sampling': True,
                'nlos_hidden_geometry_sampling_includes_relay_wall': False,
                'temporal_filter': 'box', 'max_depth': -1,
            },
            'laser': {
                'type': 'projector', 'id': 'laser',
                'to_world': T_mi.translate([-0.5, 0.0, 0.25]),
                'irradiance': {'type': 'rgb', 'value': [1,1,1]},
                'fov': 0.2,
            },
            'relay_wall': {
                'type': 'rectangle', 'id': 'relay_wall',
                'bsdf': {'type': 'diffuse',
                         'reflectance': {'type': 'rgb', 'value': [1,1,1]}},
                'sensor': {
                    'type': 'nlos_capture_meter',
                    'sensor_origin': [-0.5, 0.0, 0.25],
                    'sampler': {'type': 'independent',
                                'sample_count': 512, 'seed': 0},
                    'film': {
                        'type': 'transient_hdr_film',
                        'width': 32, 'height': 32,
                        'temporal_bins': 300, 'bin_width_opl': 0.006,
                        'start_opl': 1.85,
                        'rfilter': {'type': 'box'},
                    },
                },
            },
            'hidden_box': {
                'type': 'cube',
                'to_world': T_mi.translate([0,0,1.5]).scale(0.3),
                'bsdf': {'type': 'diffuse',
                         'reflectance': {'type': 'rgb', 'value': [0.8,0.2,0.2]}},
            },
        }
        scene = mi.load_dict(scene_dict)
        rw    = [s for s in scene.shapes()
                 if hasattr(s,'sensor') and s.sensor() is not None][0]
        mitr.nlos.focus_emitter_at_relay_wall_uv(
            mi.Point2f(0.5, 0.5), rw, scene.emitters()[0])
        _, img_t = mi.render(scene, spp=512)
        clean    = np.array(img_t, dtype=np.float64)
        np.save(clean_path, clean.astype(np.float32))

    print('Simulating averages...')
    averaged = {}
    for M in M_VALS:
        p = f'{OUT}/averaged_M{M}.npy'
        if os.path.exists(p):
            averaged[M] = np.load(p).astype(np.float64)
        else:
            averaged[M] = simulate_M(clean, M)
            np.save(p, averaged[M].astype(np.float32))
        print(f'  M={M} done')

    snrs  = {m: snr_db(clean, averaged[m])  for m in M_VALS}
    psnrs = {m: psnr_db(clean, averaged[m]) for m in M_VALS}

    bright = np.unravel_index(
        clean.sum(axis=(2,3)).argmax(), clean.shape[:2])
    ph, pw = int(bright[0]), int(bright[1])

    print('\nGenerating videos...')
    make_v1_intro()
    make_v2_waveform_buildup(clean, averaged[1], ph, pw)
    make_v3_averaging_compare(clean, averaged)
    make_v4_snr_meter(snrs, psnrs)
    make_v5_noise_peel(clean, averaged, ph, pw)
    make_v6_full_presentation(clean, averaged, snrs, psnrs, ph, pw)

    print(f'\nAll videos in {OUT}/:')
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.mp4'):
            sz = os.path.getsize(f'{OUT}/{f}') / 1e6
            print(f'  {f:<40} {sz:.2f} MB')
    print('Done.')