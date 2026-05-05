import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import mitransient as mitr

import sys
import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 1. SOFTWARE VERSIONS
# ============================================================
print("=" * 50)
print("1. SOFTWARE VERSIONS")
print("=" * 50)
print(f"Python:          {sys.version.split()[0]}")
print(f"Mitsuba:         {mi.__version__}")
print(f"mitransient:     {mitr.__version__}")
print(f"PyTorch:         {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
print(f"NumPy:           {np.__version__}")

# ============================================================
# 2. MITSUBA VARIANTS
# ============================================================
print("\n" + "=" * 50)
print("2. MITSUBA VARIANTS")
print("=" * 50)
print(f"Active variant:  {mi.variant()}")
for v in mi.variants():
    tag = " <-- ACTIVE" if v == mi.variant() else ""
    print(f"   {v}{tag}")

# ============================================================
# 3. MITRANSIENT PLUGIN REGISTRATION
# ============================================================
print("\n" + "=" * 50)
print("3. MITRANSIENT PLUGIN CHECK")
print("=" * 50)

# NOTE: angulararea must be attached to a shape, so we test it
# inside a minimal scene rather than standalone
plugins_to_check = [
    ("transient_path",       "integrator"),
    ("transient_nlos_path",  "integrator"),
    ("transient_prbvolpath", "integrator"),
    ("transient_hdr_film",   "film"),
    ("phasor_hdr_film",      "film"),
]

all_plugins_ok = True
for plugin_name, plugin_type in plugins_to_check:
    try:
        if plugin_type == "integrator":
            obj = mi.load_dict({"type": plugin_name, "max_depth": 2})
        elif plugin_type == "film":
            obj = mi.load_dict({
                "type": plugin_name,
                "width": 32, "height": 32,
                "temporal_bins": 10,
                "bin_width_opl": 0.1,
                "start_opl": 0.0,
                "rfilter": {"type": "box"},
                **({"wl_mean": 10.0, "wl_sigma": 10.0} 
                   if plugin_name == "phasor_hdr_film" else {})
            })
        print(f"   [OK]  {plugin_name} ({plugin_type})")
    except Exception as e:
        print(f"   [FAIL] {plugin_name} ({plugin_type}): {e}")
        all_plugins_ok = False

# Test angulararea properly: inside a scene with a parent shape
try:
    test_scene = mi.load_dict({
        'type': 'scene',
        'light_shape': {
            'type': 'rectangle',
            'emitter': {           # angulararea as child of a shape
                'type': 'angulararea',
                'cutoff_angle': 45.0,
                'beam_width': 30.0,
                'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
            }
        }
    })
    print(f"   [OK]  angulararea (emitter, attached to shape)")
except Exception as e:
    print(f"   [FAIL] angulararea (emitter): {e}")
    all_plugins_ok = False

# nlos_capture_meter also needs a parent shape - skip deep instantiation
print(f"   [SKIP] nlos_capture_meter (sensor) - requires parent shape, tested in render")

print(f"\nAll plugins OK: {all_plugins_ok}")

# ============================================================
# 4. RENDER: USE mitr.cornell_box() — the correct built-in scene
#    This already has the right integrator, film, lights, etc.
# ============================================================
print("\n" + "=" * 50)
print("4. CORNELL BOX RENDER (spp=64, using mitr.cornell_box())")
print("=" * 50)

# mitr.cornell_box() returns a fully configured dict with:
#   - integrator: transient_path  (correct underscore name)
#   - sensor: perspective + transient_hdr_film
#   - all geometry and lights (NO angulararea, so no shape-parent issue)
scene = mi.load_dict(mitr.cornell_box())
print("Scene loaded OK")

img_steady, img_transient = mi.render(scene, spp=64)

steady_np    = np.array(img_steady)
transient_np = np.array(img_transient)

print(f"Steady    shape: {img_steady.shape}    "
      f"range: [{steady_np.min():.4f}, {steady_np.max():.4f}]")
print(f"Transient shape: {img_transient.shape}  "
      f"range: [{transient_np.min():.4f}, {transient_np.max():.4f}]")
print(f"Has NaN - steady: {np.isnan(steady_np).any()}, "
      f"transient: {np.isnan(transient_np).any()}")

assert steady_np.max() > 0,    "Steady image is all zeros!"
assert transient_np.max() > 0, "Transient image is all zeros!"
print("Content check passed.")

# ============================================================
# 5. TONEMAPPING
# ============================================================
print("\n" + "=" * 50)
print("5. TONEMAPPING")
print("=" * 50)

transient_tonemapped = mitr.vis.tonemap_transient(transient_np)
print(f"Tonemapped shape: {transient_tonemapped.shape}")
print(f"Tonemapped range: [{transient_tonemapped.min():.4f}, "
      f"{transient_tonemapped.max():.4f}]")

# ============================================================
# 6. SAVE STATIC FRAMES
# ============================================================
print("\n" + "=" * 50)
print("6. SAVE STATIC FRAMES")
print("=" * 50)

os.makedirs("output_frames", exist_ok=True)
T = transient_tonemapped.shape[2]

# Save early / mid / late frames + a contact sheet of all three
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

frame_indices = {"early": T // 4, "mid": T // 2, "late": 3 * T // 4}
for ax, (label, idx) in zip(axes[:3], frame_indices.items()):
    frame = np.clip(transient_tonemapped[:, :, idx, :3], 0, 1)
    path  = f"output_frames/frame_{label}_{idx:03d}.png"
    plt.imsave(path, frame)
    ax.imshow(frame)
    ax.set_title(f"{label} (t={idx}/{T})")
    ax.axis('off')
    print(f"Saved: {path}")

# Steady image on the 4th panel
steady_clipped = np.clip(steady_np / steady_np.max(), 0, 1)
axes[3].imshow(steady_clipped)
axes[3].set_title("Steady state")
axes[3].axis('off')
plt.imsave("output_frames/steady.png", steady_clipped)
print("Saved: output_frames/steady.png")

plt.tight_layout()
plt.savefig("output_frames/contact_sheet.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output_frames/contact_sheet.png")

# ============================================================
# 7. SAVE VIDEO (MP4 via OpenCV)
# ============================================================
print("\n" + "=" * 50)
print("7. SAVE VIDEO (MP4)")
print("=" * 50)

import cv2

H, W, T, C = transient_tonemapped.shape
out_path = "output_frames/transient_quick.mp4"
fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
writer   = cv2.VideoWriter(out_path, fourcc, 24, (W, H))

for t in range(T):
    frame     = np.clip(transient_tonemapped[:, :, t, :3], 0, 1)
    frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    writer.write(frame_bgr)

writer.release()
size_kb = os.path.getsize(out_path) / 1024
print(f"Video saved: {out_path}  ({size_kb:.1f} KB, {T} frames @ 24fps)")
assert size_kb > 1, "Video file suspiciously small!"
print("Video size check passed.")

# ============================================================
# 8. HIGH QUALITY RENDER (spp=512)
# ============================================================
print("\n" + "=" * 50)
print("8. HIGH QUALITY RENDER (spp=512)")
print("=" * 50)

scene_hq = mi.load_dict(mitr.cornell_box())
img_steady_hq, img_transient_hq = mi.render(scene_hq, spp=512)

transient_hq_np    = np.array(img_transient_hq)
transient_hq_toned = mitr.vis.tonemap_transient(transient_hq_np)

H_hq, W_hq = transient_hq_toned.shape[:2]
out_path_hq = "output_frames/transient_hq.mp4"
writer_hq   = cv2.VideoWriter(out_path_hq, fourcc, 24, (W_hq, H_hq))

for t in range(transient_hq_toned.shape[2]):
    frame     = np.clip(transient_hq_toned[:, :, t, :3], 0, 1)
    frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    writer_hq.write(frame_bgr)

writer_hq.release()
size_hq = os.path.getsize(out_path_hq) / 1024
print(f"HQ video saved: {out_path_hq}  ({size_hq:.1f} KB)")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Variant used:      {mi.variant()}")
print(f"Steady shape:      {img_steady.shape}")
print(f"Transient shape:   {img_transient.shape}")
print(f"Time bins:         {T}")
print(f"Outputs:")
print(f"  output_frames/steady.png")
print(f"  output_frames/frame_early/mid/late_*.png")
print(f"  output_frames/contact_sheet.png")
print(f"  output_frames/transient_quick.mp4")
print(f"  output_frames/transient_hq.mp4")
print("\nAll checks complete!")




import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import mitransient as mitr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

scene = mi.load_dict(mitr.cornell_box())
img_steady, img_transient = mi.render(scene, spp=1024)

transient_np = np.array(img_transient)
steady_np    = np.array(img_steady)

# --- What does each time bin represent? ---
# Each bin = 0.02 optical path length units
# bin_width_opl = 0.02, start_opl = 3.5, temporal_bins = 300
bin_width = 0.02
start_opl = 3.5
speed_of_light = mitr.speed_of_light  # 299792458 m/s

bin_centers_opl  = start_opl + bin_width * (np.arange(300) + 0.5)
bin_centers_time = bin_centers_opl / speed_of_light  # in seconds

print(f"Time axis spans: {bin_centers_time[0]*1e9:.3f} ns  to  "
      f"{bin_centers_time[-1]*1e9:.3f} ns")
print(f"Each bin = {bin_width / speed_of_light * 1e12:.3f} ps")

# --- Plot the radiance over time (averaged over all pixels) ---
# transient_np shape: (H, W, T, C)
mean_over_pixels = transient_np.mean(axis=(0, 1))  # (T, 3)
total_radiance   = mean_over_pixels.sum(axis=-1)    # (T,)

plt.figure(figsize=(10, 4))
plt.plot(bin_centers_opl, total_radiance)
plt.xlabel("Optical path length (m)")
plt.ylabel("Mean radiance (sum RGB)")
plt.title("Light arrival over time — Cornell Box")
plt.grid(True)
plt.savefig("output_frames/time_profile.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output_frames/time_profile.png")

# --- Which pixels receive light earliest? ---
# For each pixel, find the first time bin with significant signal
threshold = transient_np.max() * 0.01  # 1% of max
first_arrival = np.argmax(transient_np.sum(axis=-1) > threshold, axis=2)
# pixels that never exceed threshold get 0 — mask them
has_signal = transient_np.sum(axis=(-1, -2)) > threshold
first_arrival_masked = np.where(has_signal, first_arrival, np.nan)

plt.figure(figsize=(6, 6))
plt.imshow(first_arrival_masked, cmap='plasma')
plt.colorbar(label='First arrival bin')
plt.title("First light arrival per pixel")
plt.axis('off')
plt.savefig("output_frames/first_arrival_map.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output_frames/first_arrival_map.png")


import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import mitransient as mitr
import numpy as np

# --- NLOS scene: relay wall + hidden Z-shaped object ---
# This simulates a camera that can "see around corners"
# by bouncing laser light off a relay wall

T  = mi.ScalarTransform4f

relay_wall_sensor = {
    'type': 'nlos_capture_meter',
    'sensor_origin': [-0.5, 0.0, 0.25],  # where the sensor physically sits
    'sampler': {
        'type': 'independent',
        'sample_count': 1024,
        'seed': 0,
    },
    'film': {
        'type': 'transient_hdr_film',
        'width':  16,   # scan grid on the relay wall
        'height': 16,
        'temporal_bins': 300,
        'bin_width_opl': 0.006,
        'start_opl': 1.85,
        'rfilter': {'type': 'box'},
    },
}

laser = {
    'type': 'projector',
    'id': 'laser',
    'to_world': T.translate([-0.5, 0.0, 0.25]),  # laser origin (same as sensor for confocal)
    'irradiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]},
    'fov': 0.2,
}

scene_dict = {
    'type': 'scene',
    'integrator': {
        'type': 'transient_nlos_path',
        'nlos_laser_sampling':    True,
        'nlos_hidden_geometry_sampling': True,
        'nlos_hidden_geometry_sampling_includes_relay_wall': False,
        'temporal_filter': 'box',
        'max_depth': -1,
    },
    'laser': laser,
    'relay_wall': {
        'type': 'rectangle',
        'id': 'relay_wall',
        'bsdf': {'type': 'diffuse',
                 'reflectance': {'type': 'rgb', 'value': [1, 1, 1]}},
        'sensor': relay_wall_sensor,
    },
    # Hidden object behind the wall — the camera cannot directly see this
    'hidden_box': {
        'type': 'cube',
        'to_world': T.translate([0.0, 0.0, 1.5]).scale(0.3),
        'bsdf': {'type': 'diffuse',
                 'reflectance': {'type': 'rgb', 'value': [0.8, 0.2, 0.2]}},
    },
}

scene       = mi.load_dict(scene_dict)
relay_wall  = scene.shapes()[1]   # the rectangle with the sensor
laser_obj   = scene.emitters()[0]

# Point laser at center of relay wall
mitr.nlos.focus_emitter_at_relay_wall_uv(
    mi.Point2f(0.5, 0.5), relay_wall, laser_obj
)

img_steady, img_transient = mi.render(scene, spp=512)

transient_np = np.array(img_transient)
print(f"NLOS transient shape: {transient_np.shape}")
# (W, H, T, C)  — note: NLOS sensor returns (scan_x, scan_y, time, channels)

tonemapped = mitr.vis.tonemap_transient(transient_np)

import cv2, os
os.makedirs("output_nlos", exist_ok=True)

H, W, T_bins, C = tonemapped.shape
writer = cv2.VideoWriter(
    "output_nlos/nlos.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'), 24, (W, H)
)
for t in range(T_bins):
    frame = np.clip(tonemapped[:, :, t, :3], 0, 1)
    writer.write(cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
writer.release()
print("Saved: output_nlos/nlos.mp4")


import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import mitransient as mitr
import drjit as dr
import numpy as np

# Render the cornell box, then optimize the light position
# to match a target transient image

scene = mi.load_dict(mitr.cornell_box())

# Get all differentiable scene parameters
params = mi.traverse(scene)
print("Differentiable parameters available:")
for k in params.keys():
    print(f"  {k}")

# Pick one to optimize — e.g. the emitter's radiance
key = 'light.emitter.radiance.value'
print(f"\nInitial value of '{key}': {params[key]}")

# Render reference (ground truth)
img_steady_ref, img_transient_ref = mi.render(scene, spp=128)

# Perturb the parameter
params[key] = params[key] * 0.5   # halve the light
params.update()

# Render perturbed
img_steady_perturbed, _ = mi.render(scene, spp=128)

# Compute simple L2 loss on steady image
loss = dr.mean(dr.sqr(img_steady_perturbed - img_steady_ref))
print(f"\nL2 loss after perturbation: {loss[0]:.6f}")

# Compute gradients
dr.backward(loss)
grad = dr.grad(params[key])
print(f"Gradient w.r.t. radiance: {grad}")
# Positive gradient means increasing radiance reduces the loss
# (i.e. we should go back toward the original value)