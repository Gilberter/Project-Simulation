# render_video.py
"""
Slide-quality video of mitransient renders.

What this shows
───────────────
The output of mi.render(scene, spp=...) is a transient tensor:
    img_t : (H, W, T, 3)
    - H, W  : spatial pixels on the relay wall
    - T     : time bins (each = one OPL slice of 'light in flight')
    - 3     : RGB channels

This script turns that tensor into three complementary videos:

  V1_transient_sweep.mp4
      The main show.  Each frame = one time bin rendered at 1920×1080.
      Clean + SPAD side by side.  A waveform strip at the bottom shows
      where in time you are and how the pulse builds up.
      Slow motion through the early wall reflection, then the hidden
      object echo — the two most important moments.

  V2_waveform_buildup.mp4
      The SPAD histogram accumulating bin by bin.
      Starts empty, each new bin lights up and is added to the running
      profile — exactly how a real SPAD experiment works.

  V3_channel_sweep.mp4
      Same transient sweep but shown split by R / G / B channel.
      Reveals which channel carries the signal and which is dark.

Usage
─────
    python render_video.py                        # uses synthetic data
    python render_video.py --real output_exp02/   # loads real .npy files

Run experiment_02_color.py first and save .npy files, or let this script
use its built-in synthetic waveforms.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Output ────────────────────────────────────────────────────────────────────
OUT = Path('output_render_video')
OUT.mkdir(exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = (9,   9,  15)      # BGR
PANEL  = (18, 18,  30)
TEXT   = (232, 232, 240)
MUTED  = (106, 106, 136)
CH_BGR = [                   # R, G, B channel colours in BGR
    ( 82, 82, 255),          # red channel
    (174, 240, 105),         # green channel
    (255, 196,  64),         # blue channel
]
DCR_BGR   = (216, 147, 206)
WHITE_BGR = (240, 240, 240)
ACCENT    = ( 64, 196, 255)  # timeline accent

# ── Video parameters ──────────────────────────────────────────────────────────
W, H  = 1920, 1080
FPS   = 30

# How many video frames per time bin in the sweep videos
# (higher = slower / more cinematic)
FRAMES_PER_BIN = 3


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class Cfg:
    photons_per_unit = 500.0
    dcr_cps          = 1_000.0
    bin_width_opl    = 0.006      # m
    start_opl        = 1.85       # m
    temporal_bins    = 300
    c                = 299_792_458.0

    @classmethod
    def dcr_per_bin(cls):
        return cls.dcr_cps * (cls.bin_width_opl / cls.c)

    @classmethod
    def dcr_floor(cls):
        return cls.dcr_per_bin() / cls.photons_per_unit

    @classmethod
    def opl(cls):
        return cls.start_opl + cls.bin_width_opl * (
            np.arange(cls.temporal_bins) + 0.5)


def add_noise(arr: np.ndarray, seed: int = 0) -> np.ndarray:
    rng    = np.random.default_rng(seed)
    counts = rng.poisson(np.clip(arr, 0, None) * Cfg.photons_per_unit)
    dark   = rng.poisson(Cfg.dcr_per_bin(), size=arr.shape)
    return (counts + dark).astype(np.float64) / Cfg.photons_per_unit


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA  (realistic-looking NLOS transient without Mitsuba)
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic(T: int = None, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (clean, noisy), each shape (16, 16, T, 3).

    Simulates a Cornell-box NLOS scene:
    - Broad Gaussian at ~2.05 m OPL  : wall retroreflection
    - Narrow Gaussian at ~2.55 m OPL : hidden object echo (green-dominant)
    - Faint multi-bounce tail
    """
    T   = T or Cfg.temporal_bins
    opl = Cfg.opl()[:T]
    rng = np.random.default_rng(seed)

    def gauss(center, sigma, amps):
        g = np.exp(-0.5 * ((opl - center) / sigma) ** 2)
        return np.stack([a * g for a in amps], axis=-1)  # (T, 3)

    profile  = np.zeros((T, 3))
    # Wall retroreflection — white, broad
    profile += gauss(2.05, 0.055, [0.30, 0.30, 0.30])
    # Hidden object — green leaf: G channel dominant
    profile += gauss(2.55, 0.035, [0.04, 0.48, 0.04])
    # Faint second-bounce from wall back to object
    profile += gauss(3.05, 0.08,  [0.012, 0.025, 0.012])

    # Spatial texture: 16×16 relay-wall pixels, each slightly different
    spatial_noise = rng.normal(0, 0.006, (16, 16, T, 3))
    spatial_scale = rng.uniform(0.85, 1.15, (16, 16, 1, 1))
    clean = np.broadcast_to(profile, (16, 16, T, 3)).copy()
    clean = clean * spatial_scale + spatial_noise
    clean = np.clip(clean, 0, None)

    noisy = add_noise(clean, seed=seed)
    return clean, noisy


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL DRAW HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def blank() -> np.ndarray:
    c = np.empty((H, W, 3), dtype=np.uint8)
    c[:] = BG
    return c


def fill_rect(img, x0, y0, x1, y1, color):
    img[y0:y1, x0:x1] = color


def put(img, text: str, x: int, y: int, scale: float,
        color=WHITE_BGR, thickness: int = 1,
        font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, (x, y), font, scale, color,
                thickness, cv2.LINE_AA)


def put_center(img, text: str, cy: int, scale: float,
               color=WHITE_BGR, thickness: int = 1):
    (tw, _), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
    cv2.putText(img, text, (W // 2 - tw // 2, cy),
                cv2.FONT_HERSHEY_DUPLEX, scale, color,
                thickness, cv2.LINE_AA)


def hline(img, y: int, color=MUTED, thickness: int = 1):
    cv2.line(img, (0, y), (W, y), color, thickness)


def fade_frames(writer, fa: np.ndarray, fb: np.ndarray, n: int = 18):
    for i in range(n):
        a = i / n
        writer.write(cv2.addWeighted(fa, 1 - a, fb, a, 0))


def hold(writer, frame: np.ndarray, n: int):
    for _ in range(n):
        writer.write(frame)


def open_writer(path: str) -> cv2.VideoWriter:
    return cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))


def report(path: str):
    mb = Path(path).stat().st_size / 1e6
    print(f'  ✓  {Path(path).name}  ({mb:.1f} MB)')


# ─────────────────────────────────────────────────────────────────────────────
# TONEMAP  — maps (H, W, T, 3) float → (H, W, T, 3) uint8
# ─────────────────────────────────────────────────────────────────────────────

def tonemap(arr: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """
    Per-channel percentile tonemapping so dim channels aren't crushed.
    Returns float32 in [0, 1].
    """
    out = np.zeros_like(arr, dtype=np.float32)
    for ch in range(arr.shape[-1]):
        v = np.percentile(arr[..., ch], percentile)
        if v > 1e-12:
            out[..., ch] = np.clip(arr[..., ch] / v, 0, 1)
    return out


def to_bgr_u8(rgb_01: np.ndarray) -> np.ndarray:
    """(H, W, 3) float [0,1] RGB → (H, W, 3) uint8 BGR."""
    return cv2.cvtColor(
        (np.clip(rgb_01, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# WAVEFORM STRIP  — drawn at the bottom of sweep frames
# ─────────────────────────────────────────────────────────────────────────────

STRIP_H  = 200   # pixel height of the waveform strip
STRIP_Y0 = H - STRIP_H


def draw_waveform_strip(
    img:       np.ndarray,
    profile:   np.ndarray,   # (T, 3) spatial-mean radiance
    t_current: int,
    T:         int,
    dcr:       float,
    opl:       np.ndarray,
) -> None:
    """
    Draws a waveform strip into the bottom STRIP_H rows of img (in-place).

    Left axis:   radiance
    Bottom axis: OPL (m)
    Vertical red line: current time bin
    Horizontal dashed line: DCR noise floor
    """
    strip_x0, strip_x1 = 60, W - 20
    strip_y0, strip_y1 = STRIP_Y0 + 8, H - 28
    pw = strip_x1 - strip_x0
    ph = strip_y1 - strip_y0

    # Background
    fill_rect(img, 0, STRIP_Y0, W, H, PANEL)
    hline(img, STRIP_Y0, MUTED)

    # Y scale
    vmax = profile.max() * 1.12 + 1e-12

    def to_px(val, t_idx):
        px = strip_x0 + int(t_idx / T * pw)
        py = strip_y1 - int(np.clip(val / vmax, 0, 1) * ph)
        return px, py

    # DCR floor line
    _, dcr_py = to_px(dcr, 0)
    cv2.line(img, (strip_x0, dcr_py), (strip_x1, dcr_py),
             DCR_BGR, 1, cv2.LINE_AA)
    put(img, 'DCR', strip_x1 + 2, dcr_py + 4, 0.32, DCR_BGR)

    # Past waveform (already revealed)
    for ch, color in enumerate(CH_BGR):
        pts = []
        for ti in range(min(t_current + 1, T)):
            pts.append(to_px(profile[ti, ch], ti))
        if len(pts) > 1:
            cv2.polyline(img,
                         [np.array(pts, dtype=np.int32)],
                         False, color, 1, cv2.LINE_AA)

    # Current-bin vertical marker
    cur_px, _ = to_px(0, t_current)
    cv2.line(img, (cur_px, strip_y0), (cur_px, strip_y1),
             ACCENT, 2, cv2.LINE_AA)

    # OPL label at cursor
    opl_label = f'{opl[t_current]:.3f} m'
    put(img, opl_label, cur_px + 4, strip_y1 - 6, 0.38, ACCENT)

    # Axis labels
    put(img, f'OPL: {opl[0]:.2f}', strip_x0, H - 8,
        0.34, MUTED)
    put(img, f'{opl[-1]:.2f} m', strip_x1 - 60, H - 8,
        0.34, MUTED)
    put(img, 'radiance', 4, (STRIP_Y0 + H) // 2,
        0.34, MUTED)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER A SINGLE TRANSIENT FRAME  (one time bin → one video frame)
# ─────────────────────────────────────────────────────────────────────────────

RENDER_H = STRIP_Y0 - 80   # vertical pixels for the render panels
RENDER_Y0 = 70


def render_frame_side_by_side(
    tm_clean:  np.ndarray,   # (H_sp, W_sp, T, 3) tonemapped float
    tm_noisy:  np.ndarray,
    profile_c: np.ndarray,   # (T, 3) spatial mean of clean
    profile_n: np.ndarray,
    t:         int,
    T:         int,
    opl:       np.ndarray,
    label:     str = '',
    dcr:       float = 0.0,
) -> np.ndarray:
    """
    Compose a 1920×1080 frame for time bin t.

    Layout:
    ┌─────────────────────────────────────────────────────┐
    │   HEADER BAR  (OPL, label)                          │
    ├───────────────────────┬─────────────────────────────┤
    │   CLEAN render        │   SPAD noisy render         │
    │   (tonemapped, ×4)    │   (tonemapped, ×4)          │
    ├───────────────────────┴─────────────────────────────┤
    │   WAVEFORM STRIP  (running profile + cursor)        │
    └─────────────────────────────────────────────────────┘
    """
    img = blank()

    # ── Header ──
    fill_rect(img, 0, 0, W, 60, PANEL)
    hline(img, 60, MUTED)
    opl_v = opl[t]
    put_center(img, f't = {t:03d}   OPL = {opl_v:.4f} m',
               42, 0.85, TEXT, 2)
    if label:
        put(img, label, 20, 42, 0.60, MUTED)
    # bin progress ticks
    tick_x = int(t / T * (W - 40)) + 20
    cv2.rectangle(img, (20, 54), (W - 20, 59), MUTED, -1)
    cv2.rectangle(img, (20, 54), (tick_x, 59), ACCENT, -1)

    # ── Render panels ──
    panel_w  = (W - 60) // 2
    panel_h  = RENDER_H
    gap      = 20
    p_left   = 20
    p_right  = p_left + panel_w + gap
    panel_y0 = RENDER_Y0

    for px0, arr_tm, panel_label in [
        (p_left,  tm_clean, 'CLEAN  (ground truth)'),
        (p_right, tm_noisy, 'SPAD   (shot noise + DCR)'),
    ]:
        # Label bar
        fill_rect(img, px0, panel_y0, px0 + panel_w, panel_y0 + 28, PANEL)
        put(img, panel_label, px0 + 8, panel_y0 + 20, 0.52, TEXT)

        # Rendered image for this time bin
        sp_h, sp_w = arr_tm.shape[:2]
        frame_rgb  = arr_tm[:, :, t, :3]                    # (sp_h, sp_w, 3)
        frame_bgr  = to_bgr_u8(frame_rgb)

        # Upscale to fill the panel — integer scale looks crisper than bilinear
        scale_x = panel_w // sp_w
        scale_y = (panel_h - 30) // sp_h
        scale   = max(1, min(scale_x, scale_y))
        ups_w   = sp_w * scale
        ups_h   = sp_h * scale

        frame_up = cv2.resize(frame_bgr, (ups_w, ups_h),
                              interpolation=cv2.INTER_NEAREST)

        # Centre in panel
        cx = px0 + (panel_w - ups_w) // 2
        cy = panel_y0 + 30 + (panel_h - 30 - ups_h) // 2
        img[cy:cy + ups_h, cx:cx + ups_w] = frame_up

        # Thin border
        cv2.rectangle(img, (cx - 1, cy - 1),
                      (cx + ups_w, cy + ups_h), MUTED, 1)

    # ── Waveform strip ──
    draw_waveform_strip(img, profile_c, t, T, dcr, opl)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO 1 — TRANSIENT SWEEP  (main show)
# ─────────────────────────────────────────────────────────────────────────────

def video_transient_sweep(
    clean:  np.ndarray,   # (H_sp, W_sp, T, 3)
    noisy:  np.ndarray,
    label:  str = '',
    name:   str = 'V1_transient_sweep.mp4',
) -> None:
    """
    Sweeps through every time bin.  Three speed zones:
    - Bins 0  … wall_bin-20  :  fast  (FRAMES_PER_BIN frames each)
    - Bins wall_bin±20        :  slow  (FRAMES_PER_BIN × 4) — wall reflection
    - Bins obj_bin±20         :  slow  (FRAMES_PER_BIN × 4) — hidden object echo
    - Tail                    :  fast again
    Ends with a 3-second freeze on the object echo peak.
    """
    T        = clean.shape[2]
    opl      = Cfg.opl()[:T]
    tm_c     = tonemap(clean)
    tm_n     = tonemap(noisy)
    profile_c = clean.mean(axis=(0, 1))   # (T, 3)
    profile_n = noisy.mean(axis=(0, 1))
    dcr       = Cfg.dcr_floor()

    # Find the two key time bins automatically
    total_intensity = profile_c.sum(axis=-1)
    wall_bin   = int(np.argmax(total_intensity[:T // 2]))
    obj_bin    = int(np.argmax(total_intensity[T // 2:]) + T // 2)
    slow_zone  = 20  # ± bins around key events

    print(f'    wall echo  : bin {wall_bin}  (OPL {opl[wall_bin]:.3f} m)')
    print(f'    object echo: bin {obj_bin}  (OPL {opl[obj_bin]:.3f} m)')

    path   = str(OUT / name)
    writer = open_writer(path)

    # Title card (1.5 s)
    title = blank()
    fill_rect(title, 0, 0, W, 60, PANEL)
    put_center(title, 'MITRANSIENT RENDER — Light in Flight',
               42, 0.90, TEXT, 2)
    if label:
        put_center(title, label, 74, 0.55, MUTED)
    hold(writer, title, FPS)

    prev_frame = title

    for t in range(T):
        # Speed
        if abs(t - wall_bin) <= slow_zone or abs(t - obj_bin) <= slow_zone:
            n_frames = FRAMES_PER_BIN * 4
        else:
            n_frames = FRAMES_PER_BIN

        frame = render_frame_side_by_side(
            tm_c, tm_n, profile_c, profile_n,
            t, T, opl, label=label, dcr=dcr)

        if t == 0:
            fade_frames(writer, prev_frame, frame, n=12)

        for _ in range(n_frames):
            writer.write(frame)

        # Annotation overlays at key moments
        if t == wall_bin:
            ann = frame.copy()
            cv2.putText(ann, 'WALL REFLECTION',
                        (W // 4 - 100, RENDER_Y0 + RENDER_H // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, ACCENT, 2, cv2.LINE_AA)
            hold(writer, ann, FPS)

        if t == obj_bin:
            ann = frame.copy()
            cv2.putText(ann, 'HIDDEN OBJECT ECHO',
                        (W // 4 - 140, RENDER_Y0 + RENDER_H // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4,
                        CH_BGR[1], 2, cv2.LINE_AA)   # green
            hold(writer, ann, int(FPS * 1.5))

    # Freeze on object echo frame
    peak_frame = render_frame_side_by_side(
        tm_c, tm_n, profile_c, profile_n,
        obj_bin, T, opl, label=label, dcr=dcr)
    hold(writer, peak_frame, FPS * 3)

    writer.release()
    report(path)


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO 2 — SPAD HISTOGRAM BUILDUP  (bin-by-bin accumulation)
# ─────────────────────────────────────────────────────────────────────────────

def video_waveform_buildup(
    noisy:  np.ndarray,   # (H_sp, W_sp, T, 3)
    label:  str = '',
    name:   str = 'V2_waveform_buildup.mp4',
) -> None:
    """
    Shows the SPAD histogram accumulating one bin at a time.
    The current bin lights up bright, the history dims.
    The noise floor is revealed after the wall echo appears.
    """
    T        = noisy.shape[2]
    opl      = Cfg.opl()[:T]
    profile  = noisy.mean(axis=(0, 1))   # (T, 3)
    dcr      = Cfg.dcr_floor()
    vmax     = profile.max() * 1.15 + 1e-12

    # Find wall and object bin
    total    = profile.sum(axis=-1)
    wall_bin = int(np.argmax(total[:T // 2]))
    obj_bin  = int(np.argmax(total[T // 2:]) + T // 2)

    path   = str(OUT / name)
    writer = open_writer(path)

    PLOT_W, PLOT_H = W - 40, H - 140
    PLOT_X0, PLOT_Y0 = 20, 100
    PLOT_X1 = PLOT_X0 + PLOT_W
    PLOT_Y1 = PLOT_Y0 + PLOT_H

    def opl_to_px(t_idx):
        return PLOT_X0 + int(t_idx / T * PLOT_W)

    def val_to_py(v):
        return PLOT_Y1 - int(np.clip(v / vmax, 0, 1) * PLOT_H)

    dcr_py    = val_to_py(dcr)
    floor_revealed = False

    for t in range(T):
        img = blank()

        # Header
        fill_rect(img, 0, 0, W, 70, PANEL)
        hline(img, 70, MUTED)
        put_center(img, 'SPAD Histogram Accumulation — Bin by Bin',
                   46, 0.85, TEXT, 2)
        if label:
            put(img, label, 20, 46, 0.52, MUTED)
        put(img, f'Bin {t:03d} / {T-1}   OPL = {opl[t]:.4f} m',
            W - 340, 46, 0.52, ACCENT)

        # Axes
        cv2.rectangle(img, (PLOT_X0, PLOT_Y0), (PLOT_X1, PLOT_Y1), MUTED, 1)

        # Reveal DCR floor after wall echo
        if t >= wall_bin and not floor_revealed:
            floor_revealed = True
        if floor_revealed:
            cv2.line(img, (PLOT_X0, dcr_py), (PLOT_X1, dcr_py),
                     DCR_BGR, 1, cv2.LINE_AA)
            put(img, f'DCR floor = {dcr:.5f}',
                PLOT_X1 - 220, dcr_py - 6, 0.40, DCR_BGR)

        # History bars + current-bin highlight
        bar_w = max(1, PLOT_W // T)

        for ch, color in enumerate(CH_BGR):
            for ti in range(t + 1):
                x0 = opl_to_px(ti)
                x1 = x0 + bar_w
                py = val_to_py(profile[ti, ch])
                # Current bin = bright; history = dimmed
                if ti == t:
                    c = color
                    alpha = 1.0
                else:
                    c = tuple(int(v * 0.45) for v in color)
                    alpha = 1.0
                cv2.rectangle(img, (x0, py), (x1, PLOT_Y1),
                              c, -1)

        # Current bin vertical marker
        cur_x = opl_to_px(t)
        cv2.line(img, (cur_x, PLOT_Y0), (cur_x, PLOT_Y1),
                 ACCENT, 2, cv2.LINE_AA)

        # Key event labels
        if t >= wall_bin:
            wx = opl_to_px(wall_bin)
            put(img, 'wall', wx - 18, PLOT_Y0 - 10, 0.42, ACCENT)
        if t >= obj_bin:
            ox = opl_to_px(obj_bin)
            put(img, 'object', ox - 24, PLOT_Y0 - 10, 0.42, CH_BGR[1])

        # OPL axis ticks
        for tick in np.linspace(opl[0], opl[-1], 7):
            ti_approx = int((tick - Cfg.start_opl) / Cfg.bin_width_opl)
            tx        = opl_to_px(ti_approx)
            cv2.line(img, (tx, PLOT_Y1), (tx, PLOT_Y1 + 5), MUTED, 1)
            put(img, f'{tick:.2f}', tx - 18, PLOT_Y1 + 18, 0.33, MUTED)

        put_center(img, 'Optical Path Length (m)', H - 12, 0.40, MUTED)

        # Legend
        for chi, (cname, ccolor) in enumerate(
                zip(['Red ch', 'Green ch', 'Blue ch'], CH_BGR)):
            lx = PLOT_X0 + chi * 160
            cv2.rectangle(img, (lx, PLOT_Y0 + 10),
                          (lx + 20, PLOT_Y0 + 26), ccolor, -1)
            put(img, cname, lx + 26, PLOT_Y0 + 24, 0.40, TEXT)

        # Repeat each frame FRAMES_PER_BIN times (≈ smooth animation speed)
        for _ in range(FRAMES_PER_BIN):
            writer.write(img)

        # Pause at key events
        if t == wall_bin:
            overlay = img.copy()
            put_center(overlay, '← Wall reflection', PLOT_Y0 + PLOT_H // 3,
                       0.75, ACCENT, 1)
            hold(writer, overlay, FPS)

        if t == obj_bin:
            overlay = img.copy()
            put_center(overlay, '← Hidden object echo', PLOT_Y0 + PLOT_H // 3,
                       0.75, CH_BGR[1], 1)
            hold(writer, overlay, int(FPS * 1.5))

    # Final freeze
    hold(writer, img, FPS * 3)
    writer.release()
    report(path)


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO 3 — CHANNEL SWEEP  (R / G / B split)
# ─────────────────────────────────────────────────────────────────────────────

def video_channel_sweep(
    clean:  np.ndarray,
    noisy:  np.ndarray,
    label:  str = '',
    name:   str = 'V3_channel_sweep.mp4',
) -> None:
    """
    Three panels side by side: Red channel | Green channel | Blue channel.
    Shows which channel carries the signal — critical for the colour experiment.
    """
    T         = clean.shape[2]
    opl       = Cfg.opl()[:T]
    dcr       = Cfg.dcr_floor()
    profile_c = clean.mean(axis=(0, 1))   # (T, 3)
    profile_n = noisy.mean(axis=(0, 1))

    # Tonemap each channel independently so dim channels are still visible
    tm_c = tonemap(clean)
    tm_n = tonemap(noisy)

    sp_h, sp_w = clean.shape[:2]
    path   = str(OUT / name)
    writer = open_writer(path)

    ch_names = ['RED CHANNEL  (640 nm)',
                'GREEN CHANNEL  (550 nm)',
                'BLUE CHANNEL  (460 nm)']

    PANEL_W  = (W - 80) // 3
    PANEL_H  = STRIP_Y0 - 90
    GAP      = 20
    LABEL_H  = 34

    total    = profile_c.sum(axis=-1)
    wall_bin = int(np.argmax(total[:T // 2]))
    obj_bin  = int(np.argmax(total[T // 2:]) + T // 2)

    # Waveform strip layout — three sub-strips stacked
    sub_strip_h = STRIP_H // 3 - 4

    for t in range(T):
        img = blank()

        # Header
        fill_rect(img, 0, 0, W, 60, PANEL)
        hline(img, 60, MUTED)
        put_center(img,
                   f'Channel Separation   t = {t:03d}   OPL = {opl[t]:.4f} m',
                   42, 0.80, TEXT, 2)
        if label:
            put(img, label, 20, 42, 0.52, MUTED)

        # Top progress bar
        tick_x = 20 + int(t / T * (W - 40))
        cv2.rectangle(img, (20, 54), (W - 20, 58), MUTED, -1)
        cv2.rectangle(img, (20, 54), (tick_x, 58), ACCENT, -1)

        for chi, (ch_name, ch_color) in enumerate(
                zip(ch_names, CH_BGR)):
            px0 = 20 + chi * (PANEL_W + GAP)

            # Channel label bar
            fill_rect(img, px0, 68, px0 + PANEL_W, 68 + LABEL_H, PANEL)
            cv2.rectangle(img, (px0, 68), (px0 + PANEL_W, 68 + LABEL_H),
                          ch_color, 1)
            put(img, ch_name, px0 + 8, 68 + 24, 0.48, ch_color, 1)

            # Render panel — single channel (replicate to RGB for display)
            frame_ch    = tm_n[:, :, t, chi]            # (sp_h, sp_w)
            frame_rgb   = np.stack([frame_ch] * 3, -1)  # grey

            # Tint with channel colour (BGR)
            tint   = np.array(ch_color, dtype=np.float32) / 255.0
            tint_bgr = tint[[0, 1, 2]]                   # already BGR
            frame_tinted = np.clip(frame_rgb * tint_bgr, 0, 1)
            frame_bgr    = (frame_tinted * 255).astype(np.uint8)

            render_y0 = 68 + LABEL_H + 6
            render_h  = PANEL_H - LABEL_H - 10
            scale_x   = PANEL_W // sp_w
            scale_y   = render_h // sp_h
            scale     = max(1, min(scale_x, scale_y))
            ups_w     = sp_w * scale
            ups_h     = sp_h * scale
            frame_up  = cv2.resize(frame_bgr, (ups_w, ups_h),
                                   interpolation=cv2.INTER_NEAREST)
            cx = px0 + (PANEL_W - ups_w) // 2
            cy = render_y0 + (render_h - ups_h) // 2
            img[cy:cy + ups_h, cx:cx + ups_w] = frame_up
            cv2.rectangle(img, (cx - 1, cy - 1),
                          (cx + ups_w, cy + ups_h), ch_color, 1)

            # SNR badge
            peak = profile_c[:, chi].max()
            snr  = 10 * np.log10(peak / (dcr + 1e-12) + 1e-6)
            snr_col = CH_BGR[1] if snr > 0 else CH_BGR[0]   # green / red
            put(img, f'SNR {snr:+.0f} dB',
                px0 + 6, cy + ups_h - 8, 0.48, snr_col, 1)

        # Bottom: three mini waveform strips
        vmax = profile_c.max() * 1.15 + 1e-12
        for chi, ch_color in enumerate(CH_BGR):
            sy0  = STRIP_Y0 + 2 + chi * (sub_strip_h + 2)
            sy1  = sy0 + sub_strip_h
            sw   = W - 80

            fill_rect(img, 40, sy0, 40 + sw, sy1, PANEL)
            hline(img, sy0, MUTED)

            def to_py_ch(v):
                return sy1 - int(np.clip(v / vmax, 0, 1) * (sub_strip_h - 4))

            def to_px_ch(ti):
                return 40 + int(ti / T * sw)

            # DCR line
            dpy = to_py_ch(dcr)
            cv2.line(img, (40, dpy), (40 + sw, dpy), DCR_BGR, 1)

            # Waveform so far
            pts = [[to_px_ch(ti), to_py_ch(profile_c[ti, chi])]
                   for ti in range(min(t + 1, T))]
            if len(pts) > 1:
                cv2.polyline(img, [np.array(pts, np.int32)],
                             False, ch_color, 1, cv2.LINE_AA)

            # Current bin marker
            cv2.line(img,
                     (to_px_ch(t), sy0), (to_px_ch(t), sy1),
                     ACCENT, 1)

            ch_label = ['R', 'G', 'B'][chi]
            put(img, ch_label, 20, (sy0 + sy1) // 2 + 5,
                0.40, ch_color)

        hline(img, STRIP_Y0, MUTED)

        if t == obj_bin:
            put_center(img, 'OBJECT ECHO', RENDER_Y0 + PANEL_H // 2,
                       1.2, ACCENT, 2)

        n_frames = FRAMES_PER_BIN * 4 \
            if (abs(t - wall_bin) <= 15 or abs(t - obj_bin) <= 15) \
            else FRAMES_PER_BIN
        for _ in range(n_frames):
            writer.write(img)

        if t == wall_bin:
            overlay = img.copy()
            put_center(overlay, 'ALL CHANNELS: Wall retroreflection',
                       68 + LABEL_H + PANEL_H // 2,
                       0.80, ACCENT, 2)
            hold(writer, overlay, FPS)

        if t == obj_bin:
            overlay = img.copy()
            put_center(overlay, 'GREEN CHANNEL ONLY: Hidden object',
                       68 + LABEL_H + PANEL_H // 2,
                       0.80, CH_BGR[1], 2)
            hold(writer, overlay, int(FPS * 1.5))

    hold(writer, img, FPS * 3)
    writer.release()
    report(path)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def load_real(npy_dir: str, key: str = 'leaf_white') -> tuple | None:
    """
    Try to load clean/noisy .npy from a previous experiment run.
    Returns (clean, noisy) or None.
    """
    p = Path(npy_dir)
    cp = p / f'clean_{key}.npy'
    np_ = p / f'noisy_{key}.npy'
    if cp.exists() and np_.exists():
        return (np.load(cp).astype(np.float64),
                np.load(np_).astype(np.float64))
    return None


def run_mitsuba(scene_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Render one scene with Mitsuba and return (clean, noisy)."""
    import mitsuba as mi
    import mitransient as mitr

    scene = mi.load_dict(scene_dict)
    rw    = next(
        (s for s in scene.shapes()
         if hasattr(s, 'sensor') and s.sensor() is not None),
        scene.shapes()[0])
    mitr.nlos.focus_emitter_at_relay_wall_uv(
        mi.Point2f(0.5, 0.5), rw, scene.emitters()[0])
    _, img_t = mi.render(scene, spp=256)
    clean    = np.array(img_t, dtype=np.float64)
    noisy    = add_noise(clean)
    return clean, noisy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mitransient render videos for slides')
    parser.add_argument('--real', metavar='NPY_DIR',
                        help='Directory with clean_*.npy / noisy_*.npy files')
    parser.add_argument('--key', default='leaf_white',
                        help='Object key to load (default: leaf_white)')
    parser.add_argument('--mitsuba', action='store_true',
                        help='Run a live Mitsuba render (requires GPU)')
    args = parser.parse_args()

    clean = noisy = None
    label = ''

    if args.mitsuba:
        print('Running Mitsuba render...')
        try:
            import mitsuba as mi
            mi.set_variant('cuda_ad_rgb')
            # Example scene — swap in your own scene dict here
            from experiment_02_color import build_nlos_scene, SimConfig, OBJECTS
            cfg_obj  = OBJECTS['leaf_white']
            sim_cfg  = SimConfig()
            scene_d  = build_nlos_scene(cfg_obj, sim_cfg)
            clean, noisy = run_mitsuba(scene_d)
            label = cfg_obj['label'].replace('\n', '  ')
        except Exception as e:
            print(f'  Mitsuba failed: {e}\n  Falling back to synthetic data.')

    if clean is None and args.real:
        result = load_real(args.real, args.key)
        if result:
            clean, noisy = result
            label = args.key.replace('_', ' ').title()
            print(f'Loaded real data from {args.real}  (key={args.key})')
        else:
            print(f'  .npy files not found in {args.real} — using synthetic data.')

    if clean is None:
        print('Using synthetic transient data...')
        clean, noisy = make_synthetic()
        label = 'Green Leaf  (white laser)  — synthetic'

    print(f'\nTensor shape: {clean.shape}  '
          f'(H×W×T×3 = {clean.shape[0]}×{clean.shape[1]}'
          f'×{clean.shape[2]}×{clean.shape[3]})')
    print(f'DCR floor: {Cfg.dcr_floor():.6f}\n')

    print('Rendering videos...')
    video_transient_sweep(clean, noisy, label=label)
    video_waveform_buildup(noisy, label=label)
    video_channel_sweep(clean, noisy, label=label)

    print(f'\nAll videos saved to ./{OUT}/')