# video_utils.py
"""
Shared video utilities for all experiments.
Professional presentation-quality videos with annotations,
progress bars, titles, and smooth transitions.
"""

import numpy as np
import cv2
import os


# ── Color palette ────────────────────────────────────────────────────────────
DARK_BG    = (15,  15,  26)    # #0f0f1a  BGR
PANEL_BG   = (26,  26,  46)    # #1a1a2e  BGR
ACCENT     = (247, 195, 79)    # #4fc3f7  BGR (light blue)
WHITE      = (255, 255, 255)
GRAY       = (170, 170, 200)
RED_C      = (80,  83,  239)   # BGR
GREEN_C    = (132, 199, 129)
BLUE_C     = (247, 195,  79)


def make_canvas(h: int, w: int) -> np.ndarray:
    c = np.zeros((h, w, 3), dtype=np.uint8)
    c[:] = DARK_BG
    return c


def draw_panel(canvas, x0, y0, w, h, color=PANEL_BG, border=ACCENT):
    cv2.rectangle(canvas, (x0, y0), (x0+w, y0+h), color,  -1)
    cv2.rectangle(canvas, (x0, y0), (x0+w, y0+h), border,  1)


def put_text(canvas, text, x, y, scale=0.45, color=WHITE,
             thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness,
                cv2.LINE_AA)


def put_text_centered(canvas, text, y, scale=0.5, color=WHITE,
                      thickness=1):
    W = canvas.shape[1]
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (W - tw) // 2
    cv2.putText(canvas, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv2.LINE_AA)


def draw_progress_bar(canvas, t: int, T: int,
                      x0=10, y0=None, width=None, height=6,
                      color_done=ACCENT, color_bg=(50, 50, 70)):
    if y0 is None:
        y0 = canvas.shape[0] - 12
    if width is None:
        width = canvas.shape[1] - 20
    cv2.rectangle(canvas, (x0, y0), (x0+width, y0+height), color_bg, -1)
    filled = int(width * t / max(T-1, 1))
    if filled > 0:
        cv2.rectangle(canvas, (x0, y0),
                      (x0+filled, y0+height), color_done, -1)


def draw_opl_marker(canvas, t: int, T: int,
                    start_opl: float, bin_w: float,
                    x0=10, y0=None):
    if y0 is None:
        y0 = canvas.shape[0] - 22
    opl    = start_opl + bin_w * (t + 0.5)
    c_light= 299_792_458.0
    time_ps= opl / c_light * 1e12
    put_text(canvas,
             f'OPL={opl:.3f}m   t={time_ps:.0f}ps   bin={t:03d}/{T}',
             x0, y0, 0.38, GRAY)


def tonemap(arr: np.ndarray, q: float = 0.995) -> np.ndarray:
    v = np.quantile(arr, q)
    return np.clip(arr / (v + 1e-12), 0, 1)


def to_bgr_u8(frame_rgb_f32: np.ndarray) -> np.ndarray:
    u8 = (np.clip(frame_rgb_f32, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)


def open_writer(path: str, w: int, h: int, fps: int = 24):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    assert writer.isOpened(), f"Could not open writer for {path}"
    return writer


def title_card(writer, title: str, subtitle: str = '',
               w: int = 800, h: int = 120,
               duration_s: float = 2.0, fps: int = 24):
    """Render a solid title card for N frames."""
    canvas = make_canvas(h, w)
    draw_panel(canvas, 5, 5, w-10, h-10)
    put_text_centered(canvas, title,    h//2 - 10, 0.55, WHITE,      1)
    put_text_centered(canvas, subtitle, h//2 + 18, 0.36, tuple(GRAY), 1)
    for _ in range(int(duration_s * fps)):
        writer.write(canvas)


def fade_in(writer, frame: np.ndarray,
            duration_s: float = 0.5, fps: int = 24):
    n = max(1, int(duration_s * fps))
    for i in range(n):
        alpha = i / n
        faded = (frame * alpha).astype(np.uint8)
        writer.write(faded)


def report_video(path: str, fps: int = 24):
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1e6
        print(f'  Saved: {path}  ({mb:.2f} MB)')
    else:
        print(f'  WARNING: {path} not created!')