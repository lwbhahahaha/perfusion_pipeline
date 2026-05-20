#!/usr/bin/env python
"""Heart-cropped QA — frame strip + V1/V2/perf panel cropped to heart bbox."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

PIPELINE_DIR = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline"
FRAMES_DIR = os.path.join(PIPELINE_DIR, "frames")
OUT_DIR    = os.path.join(PIPELINE_DIR, "output")
QA_DIR     = os.path.join(OUT_DIR, "qa"); os.makedirs(QA_DIR, exist_ok=True)
NX, NY, NZ = 1600, 1400, 500

# Heart bbox from probe (label 15 LV myo): z=[35,419] y=[317,775] x=[831,1216]
X0, X1 = 800, 1280
Y0, Y1 = 280, 820
SLICE = 200   # axial within heart range
PHANTOM_LABELS_PATH = "/home/molloi-lab/smb_mount/shared_drive/Shu Nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"

def load_int16_slice(path, z, x0=X0, x1=X1, y0=Y0, y1=Y1):
    """One z-slab (Fortran/x-fastest within the slab) → 2-D (y, x) imshow array.
    The slab buffer is laid out [x=0..NX, y=0,...,x=0..NX, y=1,...]; reshape
    to (NY, NX) with order="C" gives arr[j, i] = val(i, j), exactly what
    imshow expects (j → row → y-axis, i → column → x-axis)."""
    bytes_per_slice = NX * NY * 2
    with open(path, "rb") as f:
        f.seek(z * bytes_per_slice)
        buf = f.read(bytes_per_slice)
    arr = np.frombuffer(buf, dtype=np.int16).reshape((NY, NX), order="C")
    return arr[y0:y1, x0:x1].copy()

def load_uint8_slice(path, z, x0=X0, x1=X1, y0=Y0, y1=Y1):
    bytes_per_slice = NX * NY
    with open(path, "rb") as f:
        f.seek(z * bytes_per_slice)
        buf = f.read(bytes_per_slice)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((NY, NX), order="C")
    return arr[y0:y1, x0:x1].copy()

# ── Frame strip (heart cropped) ──
frame_files = sorted(f for f in os.listdir(FRAMES_DIR)
                     if f.startswith("frame_") and f.endswith(".raw"))
n_frames = len(frame_files)
cols, rows = 7, (n_frames + 6) // 7
fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5),
                         constrained_layout=True)
for i, fn in enumerate(frame_files):
    ax = axes.flat[i]
    s = load_int16_slice(os.path.join(FRAMES_DIR, fn), SLICE)
    ax.imshow(s, cmap="gray", vmin=-100, vmax=300)
    ax.set_title(fn.replace("frame_", "t=").replace(".raw", "s"), fontsize=10)
    ax.axis("off")
for j in range(n_frames, rows*cols):
    axes.flat[j].axis("off")
plt.suptitle(f"Heart axial slice z={SLICE} (x[{X0}:{X1}], y[{Y0}:{Y1}]) — contrast bolus passing", fontsize=14)
strip_path = os.path.join(QA_DIR, f"heart_strip_z{SLICE}.png")
plt.savefig(strip_path, dpi=120); plt.close()
print(f"[qa] {strip_path}")

# ── V1 / V2 / Perfusion panel cropped to heart ──
v1 = load_int16_slice(os.path.join(FRAMES_DIR, "frame_00.raw"), SLICE)
v2 = load_int16_slice(os.path.join(FRAMES_DIR, "frame_10.raw"), SLICE)
labels = load_uint8_slice(PHANTOM_LABELS_PATH, SLICE)
myo_mask = (labels >= 15) & (labels <= 18)

perf_npy = os.path.join(OUT_DIR, "perfusion_map.npy")
if os.path.isfile(perf_npy):
    perf = np.load(perf_npy)            # (1600, 1400, 500) Float32
    perf_slice = perf[X0:X1, Y0:Y1, SLICE].T   # (Y, X) for imshow
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for ax, img, title, vmin, vmax in zip(
        axes[:3],
        [v1, v2, v2 - v1],
        ["V1 (t=0s)", "V2 (t=10s, peak)", "ΔHU = V2 − V1 (HU)"],
        [-100, -100, 0],
        [300, 400, 200],
    ):
        ax.imshow(img, cmap="gray" if vmin < 0 else "hot", vmin=vmin, vmax=vmax)
        ax.set_title(title); ax.axis("off")
    # Perfusion overlay on V2
    masked = np.ma.masked_invalid(perf_slice)
    axes[3].imshow(v2, cmap="gray", vmin=-100, vmax=400)
    im = axes[3].imshow(masked, cmap="jet", vmin=0, vmax=4, alpha=0.7)
    plt.colorbar(im, ax=axes[3], shrink=0.8, label="mL/min/g")
    axes[3].set_title("Perfusion (mL/min/g)\noverlay on V2"); axes[3].axis("off")
    plt.tight_layout()
    panel_path = os.path.join(QA_DIR, f"heart_v1v2perf_z{SLICE}.png")
    plt.savefig(panel_path, dpi=140); plt.close()
    print(f"[qa] {panel_path}")

    # Histogram of perfusion in myocardium
    perf_full = perf
    finite = np.isfinite(perf_full) & (perf_full > 0)
    samples = perf_full[finite].ravel()
    if len(samples) > 1_000_000:
        idx = np.random.RandomState(42).choice(len(samples), 1_000_000, replace=False)
        samples = samples[idx]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.clip(samples, 0, 10), bins=100, color="steelblue", edgecolor="navy")
    ax.axvline(np.median(samples), color="red", linestyle="--", label=f"median={np.median(samples):.2f}")
    ax.axvline(samples.mean(),    color="orange", linestyle="--", label=f"mean={samples.mean():.2f}")
    ax.set_xlabel("Perfusion (mL/min/g)"); ax.set_ylabel("voxel count")
    ax.set_title("Myocardium perfusion histogram (V2=10s)")
    ax.legend(); plt.tight_layout()
    hist_path = os.path.join(QA_DIR, "perfusion_histogram.png")
    plt.savefig(hist_path, dpi=140); plt.close()
    print(f"[qa] {hist_path}")

print("\n[qa] Done.")
