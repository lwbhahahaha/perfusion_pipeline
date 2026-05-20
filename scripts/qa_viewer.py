#!/usr/bin/env python
"""
QA viewer: dump axial PNGs at one slice for each contrast frame, plus a
mid-slice figure with V1/V2/perf_map side-by-side.

Run after step 3 (frames) and optionally step 4 (perfusion):
    python qa_viewer.py [--slice 250] [--frames-only]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PIPELINE_DIR = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline"
FRAMES_DIR = os.path.join(PIPELINE_DIR, "frames")
OUT_DIR    = os.path.join(PIPELINE_DIR, "output")
QA_DIR     = os.path.join(OUT_DIR, "qa")
os.makedirs(QA_DIR, exist_ok=True)

NX, NY, NZ = 1600, 1400, 500

ap = argparse.ArgumentParser()
ap.add_argument("--slice", type=int, default=200,
                help="Z-axis slice index for axial view (0..499)")
ap.add_argument("--frames-only", action="store_true",
                help="Skip perfusion map (run before step 4)")
args = ap.parse_args()
SLICE = args.slice

def load_int16_slice(path, z):
    """Load only one z-slice from x-fastest raw int16 (1600*1400*500)."""
    bytes_per_slice = NX * NY * 2
    with open(path, "rb") as f:
        f.seek(z * bytes_per_slice)
        buf = f.read(bytes_per_slice)
    arr = np.frombuffer(buf, dtype=np.int16).reshape((NY, NX), order="F")
    return arr.copy()

# ── Frame strip ──
frame_files = sorted(f for f in os.listdir(FRAMES_DIR)
                     if f.startswith("frame_") and f.endswith(".raw"))
n_frames = len(frame_files)
print(f"[qa] Found {n_frames} frames in {FRAMES_DIR}")

cols = 7
rows = (n_frames + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3),
                         constrained_layout=True)
for i, fn in enumerate(frame_files):
    ax = axes.flat[i]
    s = load_int16_slice(os.path.join(FRAMES_DIR, fn), SLICE)
    im = ax.imshow(s, cmap="gray", vmin=-200, vmax=300)
    ax.set_title(fn.replace("frame_", "t=").replace(".raw", "s"), fontsize=10)
    ax.axis("off")
for j in range(n_frames, rows*cols):
    axes.flat[j].axis("off")
plt.suptitle(f"Axial slice z={SLICE} across contrast frames", fontsize=14)
strip_path = os.path.join(QA_DIR, f"frame_strip_z{SLICE}.png")
plt.savefig(strip_path, dpi=120)
plt.close()
print(f"[qa] Wrote {strip_path}")

# ── V1 vs V2 vs perf side-by-side ──
v1_slice = load_int16_slice(os.path.join(FRAMES_DIR, "frame_00.raw"), SLICE)

# Pick V2 = peak frame for slow territory (~t=20s)
v2_path = os.path.join(FRAMES_DIR, "frame_20.raw")
if os.path.isfile(v2_path):
    v2_slice = load_int16_slice(v2_path, SLICE)
else:
    v2_slice = v1_slice
    print(f"[qa] frame_20.raw not found, using V1 as V2")

if not args.frames_only and os.path.isfile(os.path.join(OUT_DIR, "perfusion_map.npy")):
    perf = np.load(os.path.join(OUT_DIR, "perfusion_map.npy"))
    # perf shape was set in step 4 — should be (NX, NY, NZ) order=F
    print(f"[qa] Perfusion map shape: {perf.shape}")
    if perf.shape == (NX, NY, NZ):
        perf_slice = perf[:, :, SLICE].T   # to match imshow conv
    elif perf.shape == (NZ, NY, NX):
        perf_slice = perf[SLICE]
    else:
        perf_slice = None
        print(f"[qa] Unexpected perf shape, skipping")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(v1_slice, cmap="gray", vmin=-200, vmax=300)
    axes[0].set_title(f"V1 (t=0s, baseline)\nz={SLICE}"); axes[0].axis("off")
    axes[1].imshow(v2_slice, cmap="gray", vmin=-200, vmax=300)
    axes[1].set_title(f"V2 (t=20s, post-peak)\nz={SLICE}"); axes[1].axis("off")
    if perf_slice is not None:
        masked = np.ma.masked_invalid(perf_slice)
        axes[2].imshow(v2_slice, cmap="gray", vmin=-200, vmax=300)
        axes[2].imshow(masked, cmap="jet", vmin=0, vmax=3, alpha=0.7)
        axes[2].set_title("Perfusion map (mL/min/g)\noverlay on V2"); axes[2].axis("off")
    plt.tight_layout()
    summary_path = os.path.join(QA_DIR, f"v1_v2_perf_z{SLICE}.png")
    plt.savefig(summary_path, dpi=140)
    plt.close()
    print(f"[qa] Wrote {summary_path}")
else:
    print(f"[qa] No perfusion_map.npy yet (run step 4 first), skipping V1/V2/perf panel")

print("\n[qa] Done.")
