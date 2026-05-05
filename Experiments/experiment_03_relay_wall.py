# experiment_03_relay_wall.py
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         EXPERIMENT 3: Mirror vs Matte Relay Wall                            ║
║                                                                              ║
║  THEORY — BSDF and Scattering:                                              ║
║  The relay wall material determines HOW it scatters the laser light.        ║
║                                                                              ║
║  Diffuse (Lambertian) wall — drywall / matte paint:                         ║
║    • Reflects light EQUALLY in all directions                               ║
║    • Energy is spread across the full hemisphere                            ║
║    • Result: broad, low-peak signal from the wall itself                   ║
║    • Hidden object signal is comparable in magnitude to wall scatter       ║
║                                                                              ║
║  Rough Plastic / Glossy wall — shiny paint:                                 ║
║    • Has a SPECULAR LOBE — most energy reflects in the mirror direction     ║
║    • Creates a massive, sharp "glint" peak at the wall's time-of-flight    ║
║    • This early peak can be 10-100× stronger than hidden object signal     ║
║    • Result: hidden object is buried beneath the glint artifact            ║
║                                                                              ║
║  METRIC:                                                                    ║
║  We measure the "Wall-to-Object Signal Ratio" (WOSR):                      ║
║      WOSR = peak_wall_bin / peak_object_bin                                 ║
║  High WOSR → hard reconstruction problem                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import cv2
from scipy import signal as scipy_signal
from scipy.stats import skewnorm

plt.rcParams.update({
    'figure.facecolor': '#0f0f1a', 'axes.facecolor': '#1a1a2e',
    'axes.edgecolor':   '#444466', 'axes.labelcolor': '#ccccee',
    'xtick.color':      '#aaaacc', 'ytick.color':     '#aaaacc',
    'text.color':       '#ccccee', 'grid.color':      '#2a2a4a',
    'grid.linewidth':   0.6,       'legend.facecolor': '#1a1a2e',
    'legend.edgecolor': '#444466', 'figure.dpi':       150,
})

OUT = 'output_exp03'
os.makedirs(OUT, exist_ok=True)

MATERIALS = {
    'Diffuse\n(matte drywall)': {
        'bsdf': {'type': 'diffuse',
                 'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}},
        'color': '#4fc3f7',
        'desc':  'Lambertian scatter — energy spread uniformly',
    },
    'Rough Plastic\n(shiny paint)': {
        'bsdf': {'type': 'roughplastic',
                 'distribution': 'ggx',
                 'alpha':        0.1,
                 'int_ior':      'polypropylene'},
        'color': '#ffb74d',
        'desc':  'Specular lobe — creates strong early glint peak',
    },
    'Smooth Plastic\n(glossy)': {
        'bsdf': {'type': 'roughplastic',
                 'distribution': 'ggx',
                 'alpha':        0.01,
                 'int_ior':      'polypropylene'},
        'color': '#ef5350',
        'desc':  'Nearly mirror — extreme glint overwhelms hidden object',
    },
}


def skewed_gaussian_kernel(n, sigma, skew):
    t = np.arange(n) - n // 2
    k = skewnorm.pdf(t, a=skew, loc=0, scale=sigma)
    return k / k.sum() if k.sum() > 1e-12 else np.eye(1, n, n//2)[0]


def apply_shot_and_dcr(arr, ppu, dcr_per_bin, seed=0):
    rng      = np.random.default_rng(seed)
    expected = np.clip(arr, 0, None) * ppu
    counts   = rng.poisson(expected).astype(np.float64)
    dark     = rng.poisson(dcr_per_bin, size=arr.shape).astype(np.float64)
    return (counts + dark) / ppu


def make_nlos_scene(wall_bsdf: dict):
    import mitsuba as mi
    T = mi.ScalarTransform4f
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
            'irradiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]},
            'fov': 0.2,
        },
        'relay_wall': {
            'type': 'rectangle',
            'id':   'relay_wall',
            'bsdf': wall_bsdf,
            'sensor': {
                'type': 'nlos_capture_meter',
                'sensor_origin': [-0.5, 0.0, 0.25],
                'sampler': {'type': 'independent',
                            'sample_count': 512, 'seed': 0},
                'film': {
                    'type': 'transient_hdr_film',
                    'width': 16, 'height': 16,
                    'temporal_bins': 300,
                    'bin_width_opl': 0.006,
                    'start_opl':     1.85,
                    'rfilter': {'type': 'box'},
                },
            },
        },
        'hidden_object': {
            'type': 'cube',
            'to_world': T.translate([0.0, 0.0, 1.5]).scale(0.3),
            'bsdf': {'type': 'diffuse',
                     'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}},
        },
    }


# ── PLOTS ────────────────────────────────────────────────────────────────────

def plot_bsdf_diagram(save=f'{OUT}/01_bsdf_diagram.png'):
    """Visual explanation of diffuse vs specular scattering."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (name, cfg) in zip(axes, MATERIALS.items()):
        col = cfg['color']
        # Draw wall
        ax.fill_between([-1.2, 1.2], [0, 0], [-0.15, -0.15],
                        color='#444466', alpha=0.8)
        ax.text(0, -0.08, 'Relay Wall', ha='center',
                fontsize=9, color='white')

        # Incoming laser
        ax.annotate('', xy=(0, 0), xytext=(-0.8, 0.8),
                    arrowprops=dict(arrowstyle='->', color='#ffeb3b',
                                   lw=2.5))
        ax.text(-0.85, 0.85, 'Laser', color='#ffeb3b', fontsize=8)

        # Scattered rays
        n_name = name.split('\n')[0]
        if 'Diffuse' in n_name:
            # Lambertian: uniform hemisphere
            angles = np.linspace(10, 170, 9)
            for ang in angles:
                rad = np.deg2rad(ang)
                length = 0.7
                dx, dy = length*np.cos(rad), length*np.sin(rad)
                ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->',
                                            color=col, lw=1.2,
                                            alpha=0.7))
            ax.text(0, 1.05, 'Uniform hemisphere\n(Lambertian)',
                    ha='center', color=col, fontsize=9)
        elif 'Rough' in n_name:
            # Rough: moderate specular lobe
            angles = [60, 75, 90, 105, 120]
            lengths = [0.3, 0.6, 0.9, 0.6, 0.3]
            for ang, ln in zip(angles, lengths):
                rad = np.deg2rad(ang)
                dx, dy = ln*np.cos(rad), ln*np.sin(rad)
                ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->',
                                            color=col, lw=1.5+ln,
                                            alpha=0.8))
            ax.text(0, 1.05, 'Moderate specular lobe\n(rough plastic)',
                    ha='center', color=col, fontsize=9)
        else:
            # Smooth: sharp mirror reflection
            ax.annotate('', xy=(0.8, 0.8), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->',
                                        color=col, lw=4.0))
            ax.text(0, 1.05, 'Mirror reflection\n(smooth plastic)',
                    ha='center', color=col, fontsize=9)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.25, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(name.replace('\n', ' '), color=col)
        ax.text(0, -0.22, cfg['desc'], ha='center',
                fontsize=7.5, color='#aaaacc')

    fig.suptitle('BSDF Scattering Diagrams\n'
                 'Specular walls create a "glint" that overwhelms hidden object signal',
                 fontsize=13, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save}')


def plot_waveform_comparison(results: dict, DCR_FLOOR: float,
                             save=f'{OUT}/02_waveform_comparison.png'):
    T   = list(results.values())[0]['clean'].shape[2]
    opl = 1.85 + 0.006 * (np.arange(T) + 0.5)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for col_idx, (name, data) in enumerate(results.items()):
        clean  = data['clean'].mean(axis=(0, 1))   # (T, 3) — spatial mean
        noisy  = data['noisy'].mean(axis=(0, 1))
        color  = MATERIALS[name]['color']

        # Row 0: clean signal
        ax = axes[0, col_idx]
        total_c = clean.mean(-1)
        ax.plot(opl, total_c, color=color, lw=2.0)
        ax.fill_between(opl, total_c, alpha=0.25, color=color)
        ax.axhline(DCR_FLOOR, color='#ce93d8', lw=1.0,
                   linestyle=':', label='DCR floor')
        peak_val = total_c.max()
        peak_opl = opl[total_c.argmax()]
        ax.axvline(peak_opl, color='white', lw=1.0,
                   linestyle='--', alpha=0.5)
        ax.text(peak_opl, peak_val * 0.8,
                f' peak={peak_val:.4f}\n OPL={peak_opl:.3f}m',
                fontsize=7.5, color='white')
        ax.set_title(f'{name.replace(chr(10)," ")} — CLEAN',
                     color=color)
        ax.set_xlabel('OPL (m)')
        ax.set_ylabel('Radiance')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Row 1: noisy signal
        ax = axes[1, col_idx]
        total_n = noisy.mean(-1)
        ax.plot(opl, total_n, color=color, lw=0.9, alpha=0.85)
        ax.fill_between(opl, total_n, alpha=0.15, color=color)
        ax.axhline(DCR_FLOOR, color='#ce93d8', lw=1.2,
                   linestyle=':', label='DCR floor')
        ax.fill_between(opl, DCR_FLOOR,
                        color='#ce93d8', alpha=0.08)
        ax.set_title(f'{name.replace(chr(10)," ")} — SPAD noisy',
                     color=color)
        ax.set_xlabel('OPL (m)')
        ax.set_ylabel('Radiance')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Relay Wall Material: Waveform Comparison\n'
                 'Top=clean render  ·  Bottom=SPAD noisy output  ·  '
                 'Purple=DCR floor',
                 fontsize=13, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save}')


def plot_wosr_analysis(results: dict, DCR_FLOOR: float,
                       save=f'{OUT}/03_wosr_analysis.png'):
    """Wall-to-Object Signal Ratio analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names  = [n.replace('\n', ' ') for n in results.keys()]
    colors = [MATERIALS[n]['color'] for n in results.keys()]
    wosrs  = []
    peaks  = []

    for name, data in results.items():
        profile = data['clean'].mean(axis=(0, 1, -1))   # (T,)
        T       = len(profile)
        # wall arrives in first ~50 bins, object after
        wall_peak   = profile[:80].max()
        object_peak = profile[80:].max()
        wosr        = wall_peak / (object_peak + 1e-12)
        wosrs.append(wosr)
        peaks.append((wall_peak, object_peak))

    # Bar chart: wall peak vs object peak
    ax   = axes[0]
    x    = np.arange(len(names))
    w    = 0.35
    wall_p   = [p[0] for p in peaks]
    object_p = [p[1] for p in peaks]
    bars1 = ax.bar(x - w/2, wall_p,   w, color='#ffb74d',
                   alpha=0.85, label='Wall peak', edgecolor='none')
    bars2 = ax.bar(x + w/2, object_p, w, color='#4fc3f7',
                   alpha=0.85, label='Object peak', edgecolor='none')
    ax.axhline(DCR_FLOOR, color='#ce93d8', lw=1.5,
               linestyle='--', label='DCR floor')
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + DCR_FLOOR * 0.5,
                f'{bar.get_height():.4f}',
                ha='center', fontsize=7.5, color='#ffb74d')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + DCR_FLOOR * 0.5,
                f'{bar.get_height():.4f}',
                ha='center', fontsize=7.5, color='#4fc3f7')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Peak radiance')
    ax.set_title('Wall Peak vs Object Peak\nper Material')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # WOSR bar
    ax = axes[1]
    bar_cols = [MATERIALS[n]['color'] for n in results.keys()]
    bars = ax.bar(x, wosrs, color=bar_cols, alpha=0.85, edgecolor='none')
    for bar, wv in zip(bars, wosrs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(wosrs)*0.02,
                f'WOSR={wv:.1f}×',
                ha='center', fontsize=9, fontweight='bold',
                color='white')
    ax.axhline(1, color='#81c784', lw=1.5, linestyle='--',
               label='WOSR=1 (equal wall/object)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Wall-to-Object Signal Ratio (WOSR)')
    ax.set_title('WOSR — Reconstruction Difficulty\n'
                 'Higher = harder to see hidden object')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Wall-to-Object Signal Ratio (WOSR) Analysis\n'
                 'Specular walls create massive early peaks that bury hidden object signals',
                 fontsize=13, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save}')


def plot_spatial_peak_maps(results: dict,
                           save=f'{OUT}/04_peak_maps.png'):
    """Show where the peak signal occurs spatially for each material."""
    n_mats = len(results)
    fig, axes = plt.subplots(2, n_mats, figsize=(6*n_mats, 10))

    for col, (name, data) in enumerate(results.items()):
        clean  = data['clean']              # (H, W, T, C)
        H, W   = clean.shape[:2]
        prof   = clean.mean(axis=-1)        # (H, W, T)
        color  = MATERIALS[name]['color']

        # integrated signal map
        integ = prof.sum(axis=2)            # (H, W)
        im0   = axes[0, col].imshow(integ, cmap='plasma', aspect='auto')
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)
        axes[0, col].set_title(f'{name.replace(chr(10)," ")}\nIntegrated signal',
                               color=color)
        axes[0, col].axis('off')

        # peak-time map
        peak_t = prof.argmax(axis=2).astype(float)
        im1    = axes[1, col].imshow(peak_t, cmap='viridis', aspect='auto')
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046, label='Peak bin')
        axes[1, col].set_title(f'{name.replace(chr(10)," ")}\nPeak time bin',
                               color=color)
        axes[1, col].axis('off')

    fig.suptitle('Spatial Maps: Integrated Signal & Peak Time per Material\n'
                 'Specular walls shift the peak time distribution significantly',
                 fontsize=13, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save}')


def save_wall_video(results: dict,
                    save=f'{OUT}/05_wall_comparison.mp4'):
    def tm(arr):
        v = np.quantile(arr, 0.995)
        return np.clip(arr / (v + 1e-12), 0, 1)

    items   = list(results.items())
    H, W, T = items[0][1]['clean'].shape[:3]
    # 2 rows (clean / noisy) × n_materials columns
    n       = len(items)
    W_tot   = W * n + 2 * (n-1)
    H_tot   = H * 2 + 30 + 4

    writer  = cv2.VideoWriter(
        save, cv2.VideoWriter_fourcc(*'mp4v'),
        24, (W_tot, H_tot))

    tms = [(name, tm(d['clean']), tm(d['noisy']))
           for name, d in items]

    for t in range(T):
        canvas = np.zeros((H_tot, W_tot, 3), dtype=np.uint8)
        for pi, (name, c_tm, n_tm) in enumerate(tms):
            x0 = pi * (W + 2)
            for ri, (arr_tm, row_label) in enumerate(
                    [(c_tm, 'CLEAN'), (n_tm, 'SPAD')]):
                y0    = 30 + ri * (H + 2)
                frame = np.clip(arr_tm[:, :, t, :3], 0, 1)
                canvas[y0:y0+H, x0:x0+W] = cv2.cvtColor(
                    (frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.putText(canvas, row_label,
                            (x0+3, y0-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                            (180,180,255), 1)
            label = name.split('\n')[0]
            cv2.putText(canvas, label, (x0+3, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (200,200,255), 1)
        opl = 1.85 + 0.006 * (t + 0.5)
        cv2.putText(canvas, f't={t:03d}  OPL={opl:.3f}m',
                    (4, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.30, (180,180,255), 1)
        writer.write(canvas)

    writer.release()
    mb = os.path.getsize(save) / 1e6
    print(f'Saved: {save}  ({mb:.2f} MB)')


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import mitransient as mitr

    RULER   = '═' * 55
    PPU     = 500.0
    DCR_CPS = 1000.0
    BIN_W   = 0.006
    BIN_W_S = BIN_W / 299_792_458.0
    DCR_PB  = DCR_CPS * BIN_W_S
    DCR_FLOOR = DCR_PB / PPU

    results = {}

    for name, cfg in MATERIALS.items():
        print(f'\n{RULER}')
        print(f'RENDERING: {name.replace(chr(10)," ")}')
        print(RULER)

        scene_d = make_nlos_scene(cfg['bsdf'])
        scene   = mi.load_dict(scene_d)
        rw      = [s for s in scene.shapes()
                   if hasattr(s, 'sensor') and s.sensor() is not None]
        rw      = rw[0] if rw else scene.shapes()[0]
        laser_o = scene.emitters()[0]
        mitr.nlos.focus_emitter_at_relay_wall_uv(
            mi.Point2f(0.5, 0.5), rw, laser_o)

        _, img_t  = mi.render(scene, spp=256)
        clean     = np.array(img_t, dtype=np.float64)
        noisy     = apply_shot_and_dcr(clean, PPU, DCR_PB, seed=42)
        results[name] = {'clean': clean, 'noisy': noisy}

        prof      = clean.mean(axis=(0,1,-1))
        wall_pk   = prof[:80].max()
        obj_pk    = prof[80:].max()
        print(f'  Wall peak   : {wall_pk:.5f}')
        print(f'  Object peak : {obj_pk:.5f}')
        print(f'  WOSR        : {wall_pk/(obj_pk+1e-12):.2f}×')

    plot_bsdf_diagram()
    plot_waveform_comparison(results, DCR_FLOOR)
    plot_wosr_analysis(results, DCR_FLOOR)
    plot_spatial_peak_maps(results)
    save_wall_video(results)

    print(f'\n{RULER}')
    print('SUMMARY')
    print(RULER)
    for fname in sorted(os.listdir(OUT)):
        sz = os.path.getsize(f'{OUT}/{fname}')
        u  = 'MB' if sz > 1e6 else 'KB'
        sv = sz/1e6 if sz > 1e6 else sz/1e3
        print(f'  {fname:<35} {sv:6.1f} {u}')
    print('\nDone.')