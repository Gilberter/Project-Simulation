# render_videos.py
"""
Pure render visualization videos for all experiments.
No statistics — only the actual rendered transient frames,
showing how light physically travels through each scene.

Videos:
  RV1  cornell_box_transient.mp4     — light spreading through cornell box
  RV2  nlos_averaging_transient.mp4  — NLOS hidden object, M=1 vs M=1000
  RV3  color_objects_transient.mp4   — red/blue/white/green objects
  RV4  relay_wall_transient.mp4      — diffuse vs specular wall
  RV5  light_wavefront.mp4           — single pixel trace + spatial sweep
  RV6  multiview_cornell.mp4         — cornell box from multiple angles
"""

import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import mitransient as mitr

OUT = 'output_render_videos'
os.makedirs(OUT, exist_ok=True)
FPS = 24
SPP = 512


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def tonemap_frame(frame: np.ndarray, vmax: float) -> np.ndarray:
    """
    Map a single (H,W,3) float frame to uint8 BGR.
    Uses a gentle gamma + global scale so dim early/late frames
    are still visible.
    """
    f = np.clip(frame / (vmax + 1e-12), 0, 1)
    f = np.power(f, 0.5)          # gamma 0.5 — brightens dim frames
    u8 = (f * 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)


def global_vmax(transient: np.ndarray, q: float = 0.9999) -> float:
    return float(np.quantile(transient, q))


def add_info_bar(frame_bgr: np.ndarray,
                 t: int, T: int,
                 start_opl: float, bin_w: float,
                 label: str = '',
                 bar_h: int = 28) -> np.ndarray:
    """Attach a dark info bar below the frame."""
    H, W = frame_bgr.shape[:2]
    bar  = np.zeros((bar_h, W, 3), dtype=np.uint8)
    bar[:] = (20, 15, 35)

    opl    = start_opl + bin_w * (t + 0.5)
    c_m_s  = 299_792_458.0
    t_ps   = opl / c_m_s * 1e12
    dist_m = opl                   # optical path length ≈ distance in vacuum

    pct = int(W * t / max(T - 1, 1))
    cv2.rectangle(bar, (0, bar_h-4), (pct, bar_h),
                  (247, 195, 79), -1)

    txt = (f't={t:03d}/{T}   OPL={opl:.3f}m   '
           f'time={t_ps:.0f}ps   dist≈{dist_m:.2f}m')
    if label:
        txt = f'[{label}]  ' + txt
    cv2.putText(bar, txt, (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (200, 200, 255), 1, cv2.LINE_AA)

    return np.vstack([frame_bgr, bar])


def write_video(path: str, frames_bgr: list, fps: int = FPS):
    if not frames_bgr:
        print(f'  WARNING: no frames for {path}')
        return
    H, W = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for f in frames_bgr:
        writer.write(f)
    writer.release()
    mb = os.path.getsize(path) / 1e6
    print(f'  Saved: {path}  ({mb:.2f} MB, {len(frames_bgr)} frames)')


def slow_motion_ramp(transient: np.ndarray,
                     start_opl: float, bin_w: float,
                     label: str = '',
                     slow_factor: int = 3,
                     hold_peak_s: float = 1.5) -> list:
    """
    Convert a (H,W,T,C) transient into a list of BGR frames.
    Each time bin is repeated slow_factor times (slow motion).
    The peak frame is held for hold_peak_s seconds.
    """
    T      = transient.shape[2]
    vmax   = global_vmax(transient)
    frames = []

    # find peak bin (brightest frame)
    peak_t = int(transient.sum(axis=(0, 1, 3)).argmax())

    for t in range(T):
        raw = transient[:, :, t, :3]
        bgr = tonemap_frame(raw, vmax)
        bgr = add_info_bar(bgr, t, T, start_opl, bin_w, label)

        repeats = slow_factor
        if t == peak_t:
            repeats = max(slow_factor, int(hold_peak_s * FPS))
        for _ in range(repeats):
            frames.append(bgr)

    return frames


def side_by_side(left_frames: list, right_frames: list,
                 label_l: str = 'LEFT',
                 label_r: str = 'RIGHT') -> list:
    """Stitch two frame lists side by side (pad shorter one)."""
    N    = max(len(left_frames), len(right_frames))
    out  = []
    dummy_l = left_frames[-1]  if left_frames  else None
    dummy_r = right_frames[-1] if right_frames else None

    for i in range(N):
        fl = left_frames[i]  if i < len(left_frames)  else dummy_l
        fr = right_frames[i] if i < len(right_frames) else dummy_r

        # match heights
        Hl, Wl = fl.shape[:2]
        Hr, Wr = fr.shape[:2]
        H = max(Hl, Hr)
        if Hl < H:
            fl = np.vstack([fl, np.zeros((H-Hl, Wl, 3), np.uint8)])
        if Hr < H:
            fr = np.vstack([fr, np.zeros((H-Hr, Wr, 3), np.uint8)])

        # label banners
        bw = max(Wl, Wr)
        for canvas, lbl, col in [(fl, label_l, (247,195,79)),
                                  (fr, label_r, (132,199,129))]:
            cv2.putText(canvas, lbl, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        col, 1, cv2.LINE_AA)

        divider = np.zeros((H, 3, 3), np.uint8)
        divider[:] = (60, 50, 100)
        out.append(np.hstack([fl, divider, fr]))

    return out


def grid_4panel(panels: list,
                labels: list) -> list:
    """
    4 transient arrays → 2×2 grid video.
    panels: list of (H,W,T,C) arrays
    labels: list of str
    """
    assert len(panels) == 4 and len(labels) == 4
    T     = min(p.shape[2] for p in panels)
    vmaxs = [global_vmax(p) for p in panels]

    start_opl = 1.85
    bin_w     = 0.006
    frames    = []

    for t in range(T):
        rows = []
        for row in range(2):
            cells = []
            for col in range(2):
                idx = row * 2 + col
                raw = panels[idx][:, :, t, :3]
                bgr = tonemap_frame(raw, vmaxs[idx])
                # resize all to same size
                bgr = cv2.resize(bgr, (256, 256))
                # label
                cv2.putText(bgr, labels[idx], (4, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                            (255, 235, 59), 1, cv2.LINE_AA)
                cells.append(bgr)
            rows.append(np.hstack(cells))
        grid = np.vstack(rows)

        # bottom bar
        opl = start_opl + bin_w * (t + 0.5)
        bar = np.zeros((24, grid.shape[1], 3), np.uint8)
        bar[:] = (20, 15, 35)
        pct = int(grid.shape[1] * t / max(T-1, 1))
        cv2.rectangle(bar, (0, 20), (pct, 24), (247,195,79), -1)
        cv2.putText(bar, f'bin {t:03d}  OPL={opl:.3f}m',
                    (6, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (200,200,255), 1, cv2.LINE_AA)
        frames.append(np.vstack([grid, bar]))

    return frames


# ══════════════════════════════════════════════════════════════════════════════
# RV1 — Cornell Box: light spreading through the scene
# ══════════════════════════════════════════════════════════════════════════════

def render_rv1_cornell():
    print('\n── RV1: Cornell Box transient ──')

    scene = mi.load_dict(mitr.cornell_box())
    _, img = mi.render(scene, spp=SPP)
    tr = np.array(img, dtype=np.float64)   # (H,W,T,3)
    print(f'  Shape: {tr.shape}')

    T          = tr.shape[2]
    start_opl  = 3.5
    bin_w      = 0.02
    vmax       = global_vmax(tr)
    frames     = []

    for t in range(T):
        raw = tr[:, :, t, :3]
        bgr = tonemap_frame(raw, vmax)
        bgr = add_info_bar(bgr, t, T, start_opl, bin_w,
                           'Cornell Box — light wavefront')
        frames.append(bgr)

    # ── slow-motion version (3× speed) ──
    frames_slow = slow_motion_ramp(tr, start_opl, bin_w,
                                   'Cornell Box slow-mo',
                                   slow_factor=3,
                                   hold_peak_s=2.0)

    write_video(f'{OUT}/RV1a_cornell_realtime.mp4',  frames)
    write_video(f'{OUT}/RV1b_cornell_slowmo.mp4',    frames_slow)

    # ── RGB channel split ──
    ch_colors = [(0.9,0.3,0.3), (0.3,0.9,0.3), (0.2,0.5,1.0)]
    ch_names  = ['RED channel', 'GREEN channel', 'BLUE channel']
    ch_frames = []
    for t in range(T):
        row = []
        for ci, (col_rgb, cname) in enumerate(zip(ch_colors, ch_names)):
            single = np.zeros((*tr.shape[:2], 3), dtype=np.float64)
            single[:, :, ci] = tr[:, :, t, ci]
            bgr = tonemap_frame(single, vmax)
            bgr = cv2.resize(bgr, (256, 256))
            cv2.putText(bgr, cname, (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
            row.append(bgr)
        strip = np.hstack(row)
        opl   = start_opl + bin_w * (t + 0.5)
        bar   = np.zeros((22, strip.shape[1], 3), np.uint8)
        bar[:] = (20, 15, 35)
        pct   = int(strip.shape[1] * t / max(T-1, 1))
        cv2.rectangle(bar, (0, 18), (pct, 22), (247,195,79), -1)
        cv2.putText(bar, f'bin={t:03d}  OPL={opl:.3f}m',
                    (6, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (200,200,255), 1, cv2.LINE_AA)
        ch_frames.append(np.vstack([strip, bar]))

    write_video(f'{OUT}/RV1c_cornell_rgb_channels.mp4', ch_frames)

    # ── time-integrated "light painting" accumulation ──
    accum_frames = []
    accum = np.zeros_like(tr[:, :, 0, :3])
    for t in range(T):
        accum = accum + tr[:, :, t, :3]
        bgr   = tonemap_frame(accum, global_vmax(accum) + 1e-12)
        bgr   = add_info_bar(bgr, t, T, start_opl, bin_w,
                             'Accumulation — light painting')
        accum_frames.append(bgr)

    write_video(f'{OUT}/RV1d_cornell_accumulation.mp4', accum_frames)

    np.save(f'{OUT}/cornell_transient.npy', tr.astype(np.float32))
    return tr


# ══════════════════════════════════════════════════════════════════════════════
# RV2 — NLOS scene: hidden object revealed by averaging
# ══════════════════════════════════════════════════════════════════════════════

def build_nlos_scene(hidden_color=(0.8, 0.2, 0.2)):
    T_mi = mi.ScalarTransform4f
    r, g, b = hidden_color
    return {
        'type': 'scene',
        'integrator': {
            'type': 'transient_nlos_path',
            'nlos_laser_sampling': True,
            'nlos_hidden_geometry_sampling': True,
            'nlos_hidden_geometry_sampling_includes_relay_wall': False,
            'temporal_filter': 'box',
            'max_depth': -1,
        },
        'laser': {
            'type': 'projector', 'id': 'laser',
            'to_world': T_mi.translate([-0.5, 0.0, 0.25]),
            'irradiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]},
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
                            'sample_count': SPP, 'seed': 0},
                'film': {
                    'type': 'transient_hdr_film',
                    'width': 32, 'height': 32,
                    'temporal_bins': 300,
                    'bin_width_opl': 0.006,
                    'start_opl': 1.85,
                    'rfilter': {'type': 'box'},
                },
            },
        },
        'hidden_object': {
            'type': 'cube',
            'to_world': T_mi.translate([0.0, 0.0, 1.5]).scale(0.3),
            'bsdf': {'type': 'diffuse',
                     'reflectance': {'type': 'rgb', 'value': [r,g,b]}},
        },
    }


def render_rv2_nlos():
    print('\n── RV2: NLOS averaging transient ──')

    scene = mi.load_dict(build_nlos_scene())
    rw    = [s for s in scene.shapes()
             if hasattr(s,'sensor') and s.sensor() is not None][0]
    mitr.nlos.focus_emitter_at_relay_wall_uv(
        mi.Point2f(0.5,0.5), rw, scene.emitters()[0])

    # render low-spp (noisy) and high-spp (clean)
    print('  Rendering low-SPP (noisy)...')
    _, img_low  = mi.render(scene, spp=16)
    print('  Rendering high-SPP (clean)...')
    _, img_high = mi.render(scene, spp=SPP)

    tr_low  = np.array(img_low,  dtype=np.float64)
    tr_high = np.array(img_high, dtype=np.float64)
    print(f'  Shape: {tr_high.shape}')

    start_opl = 1.85
    bin_w     = 0.006

    # single real-time video (high spp)
    frames_high = slow_motion_ramp(tr_high, start_opl, bin_w,
                                   'NLOS high-SPP (512)',
                                   slow_factor=2, hold_peak_s=1.5)
    write_video(f'{OUT}/RV2a_nlos_highspp.mp4', frames_high)

    # side by side: low vs high
    frames_low_list = slow_motion_ramp(tr_low, start_opl, bin_w,
                                       'Low SPP (16)',
                                       slow_factor=2, hold_peak_s=1.5)
    frames_high_list = slow_motion_ramp(tr_high, start_opl, bin_w,
                                        'High SPP (512)',
                                        slow_factor=2, hold_peak_s=1.5)
    sbs = side_by_side(frames_low_list, frames_high_list,
                       'SPP=16 (noisy)', 'SPP=512 (clean)')
    write_video(f'{OUT}/RV2b_nlos_spp_compare.mp4', sbs)

    # accumulation video — watch the hidden object "emerge"
    T        = tr_high.shape[2]
    vmax     = global_vmax(tr_high)
    acc_f    = []
    accum    = np.zeros_like(tr_high[:,:,0,:3])
    for t in range(T):
        accum += tr_high[:,:,t,:3]
        bgr    = tonemap_frame(accum, global_vmax(accum)+1e-12)
        bgr    = cv2.resize(bgr, (512, 512))
        bgr    = add_info_bar(bgr, t, T, start_opl, bin_w,
                              'NLOS accumulation — hidden object emerges')
        acc_f.append(bgr)
    write_video(f'{OUT}/RV2c_nlos_accumulation.mp4', acc_f)

    # sweep across scan pixels — show where signal comes from
    T_bins  = tr_high.shape[2]
    H, W    = tr_high.shape[:2]
    sweep_f = []
    for pi in range(H):
        for _ in range(2):    # 2 frames per pixel row
            canvas = np.zeros((300+24, 512, 3), np.uint8)
            canvas[:] = (20,15,35)
            # show the scan pixel highlight
            scan_vis = tonemap_frame(
                tr_high.sum(axis=2)[:,:,:3] /
                (tr_high.sum(axis=2)[:,:,:3].max()+1e-12),
                1.0)
            scan_vis = cv2.resize(scan_vis, (256,256))
            cv2.rectangle(scan_vis,
                          (0, int(pi/H*256)),
                          (256, int(pi/H*256)+4),
                          (255,235,59), -1)
            # waveform for that pixel
            profile = tr_high[pi, :, :, :].mean(axis=(0,-1))  # (T,)
            plot_h, plot_w = 300, 256
            plot_canvas = np.zeros((plot_h, plot_w, 3), np.uint8)
            plot_canvas[:] = (26,26,46)
            pmax = profile.max() + 1e-12
            for ti in range(1, T_bins):
                y1 = plot_h - int(profile[ti-1]/pmax * (plot_h-10)) - 5
                y2 = plot_h - int(profile[ti]/pmax   * (plot_h-10)) - 5
                x1 = int((ti-1)/T_bins * plot_w)
                x2 = int(ti    /T_bins * plot_w)
                cv2.line(plot_canvas, (x1,y1), (x2,y2),
                         (79,183,255), 1)
            cv2.putText(plot_canvas, f'Row {pi:02d}/{H}',
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (247,195,79), 1, cv2.LINE_AA)
            canvas[22:22+256, :256]  = scan_vis
            canvas[0:300, 256:]      = plot_canvas
            cv2.putText(canvas, 'Scan grid (integrated)',
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.33, (200,200,255), 1, cv2.LINE_AA)
            bar = np.zeros((24, 512, 3), np.uint8)
            bar[:] = (20,15,35)
            pct = int(512 * pi / max(H-1,1))
            cv2.rectangle(bar,(0,20),(pct,24),(247,195,79),-1)
            sweep_f.append(np.vstack([canvas, bar]))

    write_video(f'{OUT}/RV2d_nlos_pixel_sweep.mp4', sweep_f)

    np.save(f'{OUT}/nlos_transient.npy', tr_high.astype(np.float32))
    return tr_high


# ══════════════════════════════════════════════════════════════════════════════
# RV3 — Color objects: spectral return per object type
# ══════════════════════════════════════════════════════════════════════════════

def render_rv3_color():
    print('\n── RV3: Color objects transient ──')

    configs = {
        'White\n(red laser)':  ([0.9,0.9,0.9],   [1.0,0.1,0.1]),
        'Red\n(red laser)':    ([0.85,0.05,0.05], [1.0,0.1,0.1]),
        'Blue\n(red laser)':   ([0.05,0.05,0.85], [1.0,0.1,0.1]),
        'Green\n(white laser)':([0.08,0.75,0.08], [1.0,1.0,1.0]),
    }

    transients = {}
    for label, (refl, laser) in configs.items():
        print(f'  Rendering {label.split(chr(10))[0]}...')
        scene_d = build_nlos_scene(hidden_color=refl)
        # override laser irradiance
        scene_d['laser']['irradiance'] = {
            'type': 'rgb', 'value': laser}
        scene = mi.load_dict(scene_d)
        rw    = [s for s in scene.shapes()
                 if hasattr(s,'sensor') and s.sensor() is not None][0]
        mitr.nlos.focus_emitter_at_relay_wall_uv(
            mi.Point2f(0.5,0.5), rw, scene.emitters()[0])
        _, img = mi.render(scene, spp=256)
        transients[label] = np.array(img, dtype=np.float64)

    # 4-panel grid
    labels_list = list(transients.keys())
    panels_list = [transients[k] for k in labels_list]
    grid_f = grid_4panel(panels_list,
                         [l.replace('\n',' ') for l in labels_list])
    write_video(f'{OUT}/RV3a_color_grid.mp4', grid_f)

    # individual slow-mo per object
    for label, tr in transients.items():
        slug   = label.split('\n')[0].replace(' ','_').lower()
        frames = slow_motion_ramp(tr, 1.85, 0.006,
                                  label.replace('\n',' '),
                                  slow_factor=2, hold_peak_s=1.5)
        write_video(f'{OUT}/RV3b_{slug}.mp4', frames)

    # channel comparison: same object, 3 channels
    for label, tr in transients.items():
        slug   = label.split('\n')[0].replace(' ','_').lower()
        T      = tr.shape[2]
        vmax   = global_vmax(tr)
        ch_f   = []
        for t in range(T):
            cells = []
            ch_labels = ['R','G','B']
            ch_col    = [(0.9,0.3,0.3),(0.3,0.9,0.3),(0.2,0.5,1.0)]
            for ci, (cn, cc) in enumerate(zip(ch_labels, ch_col)):
                single      = np.zeros((*tr.shape[:2], 3),
                                       dtype=np.float64)
                single[:,:,ci] = tr[:,:,t,ci]
                bgr = tonemap_frame(single, vmax)
                bgr = cv2.resize(bgr, (200,200))
                cv2.putText(bgr, f'{cn} ch', (4,16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (255,255,255), 1, cv2.LINE_AA)
                cells.append(bgr)
            strip = np.hstack(cells)
            opl   = 1.85 + 0.006*(t+0.5)
            bar   = np.zeros((22, strip.shape[1], 3), np.uint8)
            bar[:] = (20,15,35)
            pct   = int(strip.shape[1]*t/max(T-1,1))
            cv2.rectangle(bar,(0,18),(pct,22),(247,195,79),-1)
            cv2.putText(bar,
                        f'{label.split(chr(10))[0]}  '
                        f'bin={t:03d}  OPL={opl:.3f}m',
                        (6,13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.30, (200,200,255), 1, cv2.LINE_AA)
            ch_f.append(np.vstack([strip, bar]))
        write_video(f'{OUT}/RV3c_{slug}_channels.mp4', ch_f)

    for k, v in transients.items():
        slug = k.split('\n')[0].replace(' ','_').lower()
        np.save(f'{OUT}/color_{slug}.npy', v.astype(np.float32))

    return transients


# ══════════════════════════════════════════════════════════════════════════════
# RV4 — Relay wall material: diffuse vs specular glint
# ══════════════════════════════════════════════════════════════════════════════

def build_wall_scene(wall_bsdf: dict):
    T_mi = mi.ScalarTransform4f
    return {
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
            'to_world': T_mi.translate([-0.5,0.0,0.25]),
            'irradiance': {'type':'rgb','value':[1,1,1]},
            'fov': 0.2,
        },
        'relay_wall': {
            'type': 'rectangle', 'id': 'relay_wall',
            'bsdf': wall_bsdf,
            'sensor': {
                'type': 'nlos_capture_meter',
                'sensor_origin': [-0.5,0.0,0.25],
                'sampler': {'type':'independent',
                            'sample_count':SPP,'seed':0},
                'film': {
                    'type': 'transient_hdr_film',
                    'width':16,'height':16,
                    'temporal_bins':300,
                    'bin_width_opl':0.006,
                    'start_opl':1.85,
                    'rfilter':{'type':'box'},
                },
            },
        },
        'hidden_object': {
            'type':'cube',
            'to_world': T_mi.translate([0,0,1.5]).scale(0.3),
            'bsdf':{'type':'diffuse',
                    'reflectance':{'type':'rgb','value':[0.8,0.8,0.8]}},
        },
    }


def render_rv4_wall():
    print('\n── RV4: Relay wall material transient ──')

    materials = {
        'Diffuse\n(matte)': {
            'type':'diffuse',
            'reflectance':{'type':'rgb','value':[0.8,0.8,0.8]}},
        'Rough Plastic\n(shiny)': {
            'type':'roughplastic',
            'distribution':'ggx',
            'alpha':0.1,
            'int_ior':'polypropylene'},
        'Smooth Plastic\n(glossy)': {
            'type':'roughplastic',
            'distribution':'ggx',
            'alpha':0.01,
            'int_ior':'polypropylene'},
    }

    transients = {}
    for label, bsdf in materials.items():
        print(f'  Rendering {label.split(chr(10))[0]}...')
        scene = mi.load_dict(build_wall_scene(bsdf))
        rw    = [s for s in scene.shapes()
                 if hasattr(s,'sensor') and s.sensor() is not None][0]
        mitr.nlos.focus_emitter_at_relay_wall_uv(
            mi.Point2f(0.5,0.5), rw, scene.emitters()[0])
        _, img = mi.render(scene, spp=256)
        transients[label] = np.array(img, dtype=np.float64)

    # individual videos
    for label, tr in transients.items():
        slug   = label.split('\n')[0].replace(' ','_').lower()
        frames = slow_motion_ramp(tr, 1.85, 0.006,
                                  label.replace('\n',' '),
                                  slow_factor=2, hold_peak_s=2.0)
        write_video(f'{OUT}/RV4a_{slug}.mp4', frames)

    # side by side: diffuse vs glossy
    names  = list(transients.keys())
    tr_d   = transients[names[0]]
    tr_g   = transients[names[-1]]
    f_d    = slow_motion_ramp(tr_d, 1.85, 0.006,
                              names[0].replace('\n',' '),
                              slow_factor=2)
    f_g    = slow_motion_ramp(tr_g, 1.85, 0.006,
                              names[-1].replace('\n',' '),
                              slow_factor=2)
    sbs    = side_by_side(f_d, f_g,
                          'Diffuse (matte)',
                          'Smooth Plastic (glossy)')
    write_video(f'{OUT}/RV4b_diffuse_vs_glossy.mp4', sbs)

    # 3-way comparison strip
    T      = min(t.shape[2] for t in transients.values())
    labels = list(transients.keys())
    trs    = list(transients.values())
    vmaxs  = [global_vmax(t) for t in trs]
    strip_f = []
    for t_bin in range(T):
        cells = []
        for tr, vm, lbl in zip(trs, vmaxs, labels):
            bgr = tonemap_frame(tr[:,:,t_bin,:3], vm)
            bgr = cv2.resize(bgr, (256,256))
            cv2.putText(bgr, lbl.split('\n')[0], (4,18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255,235,59), 1, cv2.LINE_AA)
            cells.append(bgr)
        strip = np.hstack(cells)
        opl   = 1.85+0.006*(t_bin+0.5)
        bar   = np.zeros((24, strip.shape[1], 3), np.uint8)
        bar[:] = (20,15,35)
        pct   = int(strip.shape[1]*t_bin/max(T-1,1))
        cv2.rectangle(bar,(0,20),(pct,24),(247,195,79),-1)
        cv2.putText(bar,
                    f'bin={t_bin:03d}  OPL={opl:.3f}m  '
                    f'← early glint visible in specular walls',
                    (6,14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.30, (200,200,255), 1, cv2.LINE_AA)
        strip_f.append(np.vstack([strip, bar]))
    write_video(f'{OUT}/RV4c_three_materials.mp4', strip_f)

    for k, v in transients.items():
        slug = k.split('\n')[0].replace(' ','_').lower()
        np.save(f'{OUT}/wall_{slug}.npy', v.astype(np.float32))

    return transients


# ══════════════════════════════════════════════════════════════════════════════
# RV5 — Cornell box: light wavefront radial sweep
# ══════════════════════════════════════════════════════════════════════════════

def render_rv5_wavefront(cornell_tr: np.ndarray):
    print('\n── RV5: Light wavefront visualization ──')

    H, W, T, C = cornell_tr.shape
    start_opl   = 3.5
    bin_w       = 0.02
    vmax        = global_vmax(cornell_tr)

    # ── brightness envelope over time (spatially summed) ──
    brightness = cornell_tr.sum(axis=(0,1,3))   # (T,)
    bmax       = brightness.max() + 1e-12

    frames = []
    for t in range(T):
        # left: current transient frame
        raw    = cornell_tr[:,:,t,:3]
        bgr    = tonemap_frame(raw, vmax)
        bgr    = cv2.resize(bgr, (400, 400))

        # right: brightness timeline with moving cursor
        timeline = np.zeros((400, 400, 3), np.uint8)
        timeline[:] = (26,26,46)

        # draw brightness curve up to t
        for ti in range(1, T):
            y1 = 380 - int(brightness[ti-1]/bmax * 340)
            y2 = 380 - int(brightness[ti]  /bmax * 340)
            x1 = int((ti-1)/T * 380) + 10
            x2 = int(ti     /T * 380) + 10
            col = (79,195,247) if ti <= t else (60,60,80)
            cv2.line(timeline, (x1,y1),(x2,y2), col, 1)

        # cursor
        xc = int(t/T * 380) + 10
        yc = 380 - int(brightness[t]/bmax * 340)
        cv2.circle(timeline, (xc,yc), 5, (255,235,59), -1)
        cv2.line(timeline, (xc,0),(xc,400),(255,235,59,), 1)

        # axis labels
        cv2.putText(timeline, 'Total scene brightness',
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (200,200,255), 1, cv2.LINE_AA)
        cv2.putText(timeline, 'Time →',
                    (330,395), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (170,170,200), 1, cv2.LINE_AA)

        opl = start_opl + bin_w*(t+0.5)
        cv2.putText(timeline,
                    f'OPL={opl:.2f}m',
                    (10,395), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (247,195,79), 1, cv2.LINE_AA)

        combined = np.hstack([bgr, timeline])
        bar      = np.zeros((24, combined.shape[1], 3), np.uint8)
        bar[:]   = (20,15,35)
        pct      = int(combined.shape[1]*t/max(T-1,1))
        cv2.rectangle(bar,(0,20),(pct,24),(247,195,79),-1)
        cv2.putText(bar,
                    f'Cornell Box — Light Wavefront   bin={t:03d}/{T}',
                    (6,14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (200,200,255), 1, cv2.LINE_AA)
        frames.append(np.vstack([combined, bar]))

    write_video(f'{OUT}/RV5a_wavefront_timeline.mp4', frames)

    # ── heatmap evolution ──
    heat_frames = []
    for t in range(T):
        raw  = cornell_tr[:,:,t,:].mean(axis=-1)   # (H,W)
        norm = raw / (vmax + 1e-12)
        norm = np.clip(norm, 0, 1)

        # apply colormap
        norm_u8  = (norm * 255).astype(np.uint8)
        heatmap  = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
        heatmap  = cv2.resize(heatmap, (512, 512))

        opl = start_opl + bin_w*(t+0.5)
        bar = np.zeros((28, 512, 3), np.uint8)
        bar[:] = (20,15,35)
        pct    = int(512*t/max(T-1,1))
        cv2.rectangle(bar,(0,24),(pct,28),(247,195,79),-1)
        cv2.putText(bar,
                    f'Energy heatmap  bin={t:03d}  OPL={opl:.2f}m',
                    (6,16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (200,200,255), 1, cv2.LINE_AA)
        heat_frames.append(np.vstack([heatmap, bar]))

    write_video(f'{OUT}/RV5b_heatmap.mp4', heat_frames)

    # ── false-color temporal "rainbow" video ──
    # Each pixel is colored by WHEN it receives its peak photon
    peak_t_map = cornell_tr.sum(axis=-1).argmax(axis=2)  # (H,W)
    peak_norm  = (peak_t_map / T * 255).astype(np.uint8)
    rainbow    = cv2.applyColorMap(peak_norm, cv2.COLORMAP_JET)
    rainbow    = cv2.resize(rainbow, (512, 512))

    rainbow_frames = [rainbow] * (FPS * 3)   # hold 3 s
    cv2.putText(rainbow,
                'Peak arrival time — each color = a different moment',
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (255,255,255), 1, cv2.LINE_AA)
    write_video(f'{OUT}/RV5c_peak_arrival_map.mp4', rainbow_frames)


# ══════════════════════════════════════════════════════════════════════════════
# RV6 — Cornell box re-rendered at 3 different spatial resolutions
#        to show how detail emerges with more pixels
# ══════════════════════════════════════════════════════════════════════════════

def render_rv6_resolution():
    print('\n── RV6: Resolution comparison ──')

    resolutions = [(32,32), (128,128), (256,256)]
    labels_res  = ['32×32', '128×128', '256×256']
    transients  = []

    for (rw, rh), lbl in zip(resolutions, labels_res):
        print(f'  Rendering {lbl}...')
        cb = mitr.cornell_box()
        cb['sensor']['film']['width']  = rw
        cb['sensor']['film']['height'] = rh
        scene = mi.load_dict(cb)
        _, img = mi.render(scene, spp=256)
        transients.append(np.array(img, dtype=np.float64))

    T     = min(t.shape[2] for t in transients)
    vmaxs = [global_vmax(t) for t in transients]

    res_frames = []
    for t_bin in range(T):
        cells = []
        for tr, vm, lbl in zip(transients, vmaxs, labels_res):
            bgr = tonemap_frame(tr[:,:,t_bin,:3], vm)
            bgr = cv2.resize(bgr, (256, 256),
                             interpolation=cv2.INTER_NEAREST)
            cv2.putText(bgr, lbl, (4,18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (255,235,59), 1, cv2.LINE_AA)
            cells.append(bgr)
        strip  = np.hstack(cells)
        opl    = 3.5 + 0.02*(t_bin+0.5)
        bar    = np.zeros((24, strip.shape[1], 3), np.uint8)
        bar[:] = (20,15,35)
        pct    = int(strip.shape[1]*t_bin/max(T-1,1))
        cv2.rectangle(bar,(0,20),(pct,24),(247,195,79),-1)
        cv2.putText(bar,
                    f'Spatial resolution  bin={t_bin:03d}  OPL={opl:.2f}m',
                    (6,14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.33, (200,200,255), 1, cv2.LINE_AA)
        res_frames.append(np.vstack([strip, bar]))

    write_video(f'{OUT}/RV6_resolution_compare.mp4', res_frames)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    RULER = '═' * 55
    print(f'\n{RULER}')
    print('PURE RENDER VISUALIZATION VIDEOS')
    print(f'{RULER}')
    print('No statistics — only light transport renders\n')

    cornell_tr   = render_rv1_cornell()
    nlos_tr      = render_rv2_nlos()
    color_trs    = render_rv3_color()
    wall_trs     = render_rv4_wall()
    render_rv5_wavefront(cornell_tr)
    render_rv6_resolution()

    print(f'\n{RULER}')
    print(f'ALL VIDEOS in {OUT}/')
    print(RULER)

    all_vids = sorted([f for f in os.listdir(OUT)
                       if f.endswith('.mp4')])
    total_mb = 0
    for fname in all_vids:
        sz = os.path.getsize(f'{OUT}/{fname}') / 1e6
        total_mb += sz
        print(f'  {fname:<45} {sz:5.1f} MB')
    print(f'\n  Total: {total_mb:.1f} MB  |  {len(all_vids)} videos')
    print('\nDone.')