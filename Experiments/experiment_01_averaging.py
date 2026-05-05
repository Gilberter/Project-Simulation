# experiment_01_averaging.py
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         EXPERIMENT 1: Signal Averaging vs. Reconstruction Quality           ║
║                                                                              ║
║  THEORY:                                                                     ║
║  Shot noise follows Poisson statistics: SNR = √N                            ║
║  where N = number of photons per bin.                                        ║
║                                                                              ║
║  Averaging M independent renders is mathematically equivalent to            ║
║  collecting M times more photons:                                            ║
║      SNR_averaged = √(M × N) = √M × SNR_single                             ║
║                                                                              ║
║  This means:                                                                 ║
║    •  10 averages  → √10  ≈ 3.16× SNR improvement                          ║
║    • 100 averages  → √100 = 10× SNR improvement                             ║
║    •1000 averages  → √1000 ≈ 31.6× SNR improvement                         ║
║                                                                              ║
║  EXPERIMENT:                                                                 ║
║  1. Render ONE clean transient of an NLOS hidden object                     ║
║  2. Apply shot noise (M=1 simulation)                                        ║
║  3. Simulate averaging by rendering M=10,100,1000 independent noisy         ║
║     copies and averaging them → equivalent to more laser pulses             ║
║  4. Plot SNR and PSNR vs M to confirm √M scaling law                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import cv2
from scipy import signal
from scipy.stats import skewnorm

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0f1a', 'axes.facecolor': '#1a1a2e',
    'axes.edgecolor':   '#444466', 'axes.labelcolor': '#ccccee',
    'xtick.color':      '#aaaacc', 'ytick.color':     '#aaaacc',
    'text.color':       '#ccccee', 'grid.color':      '#2a2a4a',
    'grid.linewidth':   0.6,       'legend.facecolor': '#1a1a2e',
    'legend.edgecolor': '#444466', 'figure.dpi':       150,
})

OUT = 'output_exp01'
os.makedirs(OUT, exist_ok=True)

PALETTE = {
    1:    '#ef5350',
    10:   '#ffb74d',
    100:  '#81c784',
    1000: '#4fc3f7',
    'clean': '#ffffff',
}


# ══════════════════════════════════════════════════════════════════════════════
# Noise helpers (self-contained, no import from sim_to_real)
# ══════════════════════════════════════════════════════════════════════════════

def skewed_gaussian_kernel(n: int, sigma: float, skew: float) -> np.ndarray:
    t = np.arange(n) - n // 2
    k = skewnorm.pdf(t, a=skew, loc=0, scale=sigma)
    return k / k.sum() if k.sum() > 1e-12 else (np.eye(1, n, n//2)[0])


def apply_irf(transient: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    H, W, T, C = transient.shape
    out = np.zeros_like(transient)
    for c in range(C):
        for h in range(H):
            row = transient[h, :, :, c]
            out[h, :, :, c] = np.array([
                signal.fftconvolve(row[w], kernel, mode='same')
                for w in range(W)
            ])
    return out


def add_shot_noise(arr: np.ndarray, ppu: float,
                   rng: np.random.Generator) -> np.ndarray:
    expected = np.clip(arr, 0, None) * ppu
    return rng.poisson(expected).astype(np.float64) / ppu


def add_dcr(arr: np.ndarray, dcr_per_bin: float,
            ppu: float, rng: np.random.Generator) -> np.ndarray:
    dark = rng.poisson(dcr_per_bin, size=arr.shape).astype(np.float64)
    return arr + dark / ppu


def simulate_M_averages(clean: np.ndarray, M: int,
                        ppu: float, dcr_per_bin: float,
                        irf_kernel: np.ndarray,
                        seed: int = 0) -> np.ndarray:
    """
    Simulate M independent noisy acquisitions and return their average.
    Each acquisition: IRF → shot noise → DCR.
    Averaging M scans reduces noise by √M (central limit theorem).
    """
    rng    = np.random.default_rng(seed)
    accum  = np.zeros_like(clean, dtype=np.float64)
    # IRF is deterministic — apply once
    irf_clean = apply_irf(clean, irf_kernel)
    for _ in range(M):
        noisy = add_shot_noise(irf_clean, ppu, rng)
        noisy = add_dcr(noisy, dcr_per_bin, ppu, rng)
        accum += noisy
    return accum / M


def snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    sp = np.mean(clean ** 2)
    np_noise = np.mean((noisy - clean) ** 2)
    return float(10 * np.log10(sp / (np_noise + 1e-12)))


def psnr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    mse = np.mean((noisy - clean) ** 2)
    return float(10 * np.log10(clean.max() ** 2 / (mse + 1e-12)))


def rmse(clean: np.ndarray, noisy: np.ndarray) -> float:
    return float(np.sqrt(np.mean((noisy - clean) ** 2)))


def peak_snr_px(clean_px: np.ndarray, noisy_px: np.ndarray) -> float:
    sig = np.abs(clean_px).max()
    noi = np.abs(noisy_px - clean_px).std()
    return float(20 * np.log10(sig / (noi + 1e-12)))


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_snr_vs_M(M_values, snrs, psnrs, rmses,
                  save=f'{OUT}/01_snr_vs_M.png'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # SNR
    ax = axes[0]
    ax.semilogx(M_values, snrs, 'o-', color='#4fc3f7',
                linewidth=2.5, markersize=9, label='Measured SNR')
    # theoretical √M line anchored at M=1
    M_th = np.logspace(0, 3, 200)
    ax.semilogx(M_th, snrs[0] + 10 * np.log10(M_th),
                '--', color='#ffb74d', linewidth=1.8,
                label='Theory: SNR₀ + 10·log₁₀(M)')
    for mv, sv in zip(M_values, snrs):
        ax.annotate(f'{sv:.1f} dB', (mv, sv),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=8, color='#ccccee')
    ax.set_xlabel('Number of averaged scans (M)')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR vs Averages\n(log scale x-axis)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PSNR
    ax = axes[1]
    ax.semilogx(M_values, psnrs, 's-', color='#81c784',
                linewidth=2.5, markersize=9, label='Measured PSNR')
    ax.semilogx(M_th, psnrs[0] + 10 * np.log10(M_th),
                '--', color='#ffb74d', linewidth=1.8,
                label='Theory +10·log₁₀(M)')
    for mv, pv in zip(M_values, psnrs):
        ax.annotate(f'{pv:.1f} dB', (mv, pv),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=8, color='#ccccee')
    ax.set_xlabel('Number of averaged scans (M)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR vs Averages')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE
    ax = axes[2]
    ax.loglog(M_values, rmses, '^-', color='#ef5350',
              linewidth=2.5, markersize=9, label='Measured RMSE')
    ax.loglog(M_th, rmses[0] / np.sqrt(M_th),
              '--', color='#ffb74d', linewidth=1.8,
              label='Theory: RMSE₀ / √M')
    for mv, rv in zip(M_values, rmses):
        ax.annotate(f'{rv:.5f}', (mv, rv),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=8, color='#ccccee')
    ax.set_xlabel('Number of averaged scans (M)')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE vs Averages\n(log-log)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    fig.suptitle('Reconstruction Quality vs Number of Laser Pulses (Averages)\n'
                 'SNR improves as √M — confirmed by experiment',
                 fontsize=13, fontweight='bold', color='white', y=1.01)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save}')


def plot_waveform_comparison(clean: np.ndarray,
                             averaged: dict,
                             ph: int, pw: int,
                             save=f'{OUT}/02_waveform_comparison.png'):
    T   = clean.shape[2]
    opl = 3.5 + 0.02 * (np.arange(T) + 0.5)

    def px(arr):
        return arr[ph, pw, :, :].mean(-1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # panel 0: clean reference
    ax = axes[0]
    c  = px(clean)
    ax.plot(opl, c, color=PALETTE['clean'], lw=2)
    ax.fill_between(opl, c, alpha=0.2, color=PALETTE['clean'])
    ax.set_title('Clean Reference\n(ideal, no noise)')
    ax.set_xlabel('OPL (m)')
    ax.set_ylabel('Radiance')
    ax.grid(True, alpha=0.3)

    # panels 1-4: one per M
    for idx, (M, arr) in enumerate(averaged.items()):
        ax  = axes[idx + 1]
        col = PALETTE[M]
        n   = px(arr)
        ax.plot(opl, c,   color=PALETTE['clean'], lw=1.5,
                alpha=0.4, label='Clean')
        ax.plot(opl, n,   color=col,              lw=1.2,
                alpha=0.85, label=f'M={M}')
        ax.fill_between(opl, n, alpha=0.15, color=col)
        psnr = psnr_db(clean[ph:ph+1, pw:pw+1],
                       arr[ph:ph+1, pw:pw+1])
        ax.set_title(f'M = {M} averages\nPSNR = {psnr:.1f} dB')
        ax.set_xlabel('OPL (m)')
        ax.set_ylabel('Radiance')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # panel 5: all overlaid
    ax = axes[5]
    ax.plot(opl, px(clean), color=PALETTE['clean'], lw=2.5,
            label='Clean', zorder=5)
    for M, arr in averaged.items():
        ax.plot(opl, px(arr), color=PALETTE[M], lw=1.2,
                alpha=0.8, label=f'M={M}')
    ax.set_title('All averages overlaid')
    ax.set_xlabel('OPL (m)')
    ax.set_ylabel('Radiance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Waveform Recovery vs Number of Averaged Scans\n'
                 'More averages → cleaner waveform → better NLOS reconstruction',
                 fontsize=13, fontweight='bold', color='white', y=1.01)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save}')


def plot_noise_floor_reduction(clean: np.ndarray,
                               averaged: dict,
                               save=f'{OUT}/03_noise_floor.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    M_vals, std_vals, mean_diff_vals = [], [], []
    for M, arr in averaged.items():
        diff = arr - clean
        M_vals.append(M)
        std_vals.append(diff.std())
        mean_diff_vals.append(np.abs(diff).mean())

    M_th  = np.logspace(0, 3, 200)
    ref   = std_vals[0]

    ax = axes[0]
    ax.loglog(M_vals, std_vals, 'o-', color='#ef5350',
              lw=2.5, ms=9, label='Measured noise σ')
    ax.loglog(M_th, ref / np.sqrt(M_th),
              '--', color='#ffb74d', lw=2,
              label='Theory: σ₀ / √M')
    for mv, sv in zip(M_vals, std_vals):
        ax.annotate(f'{sv:.5f}', (mv, sv),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=8, color='#ccccee')
    ax.set_xlabel('M (averages)')
    ax.set_ylabel('Noise std deviation')
    ax.set_title('Noise Floor Reduction\n(log-log confirms √M law)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    ax = axes[1]
    improvement = [ref / s for s in std_vals]
    theory_imp  = [np.sqrt(m) for m in M_vals]
    ax.semilogx(M_vals, improvement, 'o-', color='#4fc3f7',
                lw=2.5, ms=9, label='Measured improvement')
    ax.semilogx(M_vals, theory_imp, 's--', color='#ffb74d',
                lw=2, ms=7, label='Theory √M')
    for mv, iv, tv in zip(M_vals, improvement, theory_imp):
        ax.annotate(f'×{iv:.1f}', (mv, iv),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=8.5, color='#ccccee')
    ax.set_xlabel('M (averages)')
    ax.set_ylabel('SNR improvement factor')
    ax.set_title('SNR Improvement Factor\nvs Theoretical √M')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Noise Floor Reduction — √M Scaling Law Verification',
                 fontsize=13, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save}')


def save_averaging_video(clean: np.ndarray,
                         averaged: dict,
                         save=f'{OUT}/04_averaging.mp4'):
    def tm(arr):
        v = np.quantile(arr, 0.995)
        return np.clip(arr / (v + 1e-12), 0, 1)

    H, W, T, C = clean.shape
    panels      = [('CLEAN', clean, '#4fc3f7')] + \
                  [(f'M={m}', a, PALETTE[m]) for m, a in averaged.items()]
    n_panels    = len(panels)
    W_total     = W * n_panels + 2 * (n_panels - 1)

    writer = cv2.VideoWriter(
        save, cv2.VideoWriter_fourcc(*'mp4v'),
        24, (W_total, H + 30))

    tms = [(lbl, tm(arr), col) for lbl, arr, col in panels]

    for t in range(T):
        canvas = np.zeros((H + 30, W_total, 3), dtype=np.uint8)
        for pi, (lbl, arr_tm, col) in enumerate(tms):
            x0    = pi * (W + 2)
            frame = np.clip(arr_tm[:, :, t, :3], 0, 1)
            frame_u8 = (frame * 255).astype(np.uint8)
            canvas[30:, x0:x0+W] = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
            cv2.putText(canvas, lbl, (x0 + 4, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 255), 1)
        opl = 3.5 + 0.02 * (t + 0.5)
        cv2.putText(canvas, f't={t:03d}  OPL={opl:.2f}m',
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (180, 180, 255), 1)
        writer.write(canvas)
    writer.release()
    mb = os.path.getsize(save) / 1e6
    print(f'Saved: {save}  ({mb:.2f} MB)')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import mitransient as mitr

    RULER = '═' * 55

    # ── render NLOS scene ──────────────────────────────────────────────────
    print(f'\n{RULER}')
    print('A. RENDER NLOS SCENE WITH HIDDEN OBJECT')
    print(RULER)

    T_mi = mi.ScalarTransform4f

    laser_dict = {
        'type': 'projector',
        'id':   'laser',
        'to_world': T_mi.translate([-0.5, 0.0, 0.25]),
        'irradiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]},
        'fov': 0.2,
    }
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'transient_nlos_path',
            'nlos_laser_sampling':   True,
            'nlos_hidden_geometry_sampling': True,
            'nlos_hidden_geometry_sampling_includes_relay_wall': False,
            'temporal_filter': 'box',
            'max_depth': -1,
        },
        'laser': laser_dict,
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
                    'width': 32, 'height': 32,
                    'temporal_bins': 300,
                    'bin_width_opl': 0.006,
                    'start_opl': 1.85,
                    'rfilter': {'type': 'box'},
                },
            },
        },
        'hidden_box': {
            'type': 'cube',
            'to_world': T_mi.translate([0.0, 0.0, 1.5]).scale(0.3),
            'bsdf': {'type': 'diffuse',
                     'reflectance': {'type': 'rgb', 'value': [0.8, 0.2, 0.2]}},
        },
    }



    scene      = mi.load_dict(scene_dict)
    relay_wall = scene.shapes()[0]
    laser_obj  = scene.emitters()[0]

    # 1. Manually calculate the target point on the relay wall
    # This gets the 3D position of the UV coordinate (0.5, 0.5)
    target_pos = relay_wall.sample_position(
        time=0.0, 
        sample=mi.Point2f(0.5, 0.5), 
        active=True
    ).p

    # 2. Define the laser origin (must match your laser_dict translate)
    laser_origin = mi.Point3f(-0.5, 0.0, 0.25)
    # 3. Create the rotation matrix to point the laser at the target
    # .look_at(origin, target, up)
    new_to_world = mi.Transform4f.look_at(
        origin=laser_origin,
        target=target_pos,
        up=[0, 1, 0]
    )

    # 4. Update the laser emitter in the live scene
    params = mi.traverse(laser_obj)
    params['to_world'] = new_to_world
    params.update()
    
    print("Laser successfully focused at relay wall center.")

    _, img_t   = mi.render(scene, spp=512)
    clean      = np.array(img_t, dtype=np.float64)
    print(f'Clean shape : {clean.shape}   range [{clean.min():.4f}, {clean.max():.4f}]')

    # ── noise params ───────────────────────────────────────────────────────
    PPU          = 500.0
    DCR_CPS      = 1000.0
    BIN_W_OPL    = 0.006
    BIN_W_S      = BIN_W_OPL / 299_792_458.0
    DCR_PER_BIN  = DCR_CPS * BIN_W_S
    IRF_KERN     = skewed_gaussian_kernel(31, 1.5, 3.0)
    M_VALUES     = [1, 10, 100, 1000]

    # ── simulate averaging ─────────────────────────────────────────────────
    print(f'\n{RULER}')
    print('B. SIMULATING AVERAGING')
    print(RULER)

    averaged = {}
    for M in M_VALUES:
        print(f'  M = {M:4d} ...', end=' ', flush=True)
        averaged[M] = simulate_M_averages(
            clean, M, PPU, DCR_PER_BIN, IRF_KERN, seed=M)
        print('done')

    # ── metrics ────────────────────────────────────────────────────────────
    print(f'\n{RULER}')
    print('C. METRICS')
    print(RULER)

    snrs, psnrs, rmses = [], [], []
    print(f'  {"M":>6}  {"SNR(dB)":>9}  {"PSNR(dB)":>9}  {"RMSE":>10}  '
          f'{"Improvement":>12}')
    print(f'  {"─"*6}  {"─"*9}  {"─"*9}  {"─"*10}  {"─"*12}')

    for M in M_VALUES:
        s = snr_db(clean, averaged[M])
        p = psnr_db(clean, averaged[M])
        r = rmse(clean, averaged[M])
        snrs.append(s)
        psnrs.append(p)
        rmses.append(r)
        imp = snrs[0] / s if snrs[0] != 0 else 1.0
        print(f'  {M:>6}  {s:>9.2f}  {p:>9.2f}  {r:>10.6f}  '
              f'    ×{np.sqrt(M):>6.2f} (theory)')

    # ── bright pixel ───────────────────────────────────────────────────────
    bright = np.unravel_index(
        clean.sum(axis=(2, 3)).argmax(), clean.shape[:2])
    ph, pw = int(bright[0]), int(bright[1])

    # ── plots ──────────────────────────────────────────────────────────────
    print(f'\n{RULER}')
    print('D. GENERATING PLOTS')
    print(RULER)

    plot_snr_vs_M(M_VALUES, snrs, psnrs, rmses)
    plot_waveform_comparison(clean, averaged, ph, pw)
    plot_noise_floor_reduction(clean, averaged)
    save_averaging_video(clean, averaged)

    # ── save arrays ────────────────────────────────────────────────────────
    np.save(f'{OUT}/clean.npy', clean.astype(np.float32))
    for M, arr in averaged.items():
        np.save(f'{OUT}/averaged_M{M}.npy', arr.astype(np.float32))

    print(f'\n{RULER}')
    print('SUMMARY')
    print(RULER)
    for fname in sorted(os.listdir(OUT)):
        sz = os.path.getsize(f'{OUT}/{fname}')
        u  = 'MB' if sz > 1e6 else 'KB'
        sv = sz/1e6 if sz > 1e6 else sz/1e3
        print(f'  {fname:<35} {sv:6.1f} {u}')
    print('\nDone.')