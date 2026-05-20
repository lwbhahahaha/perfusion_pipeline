#!/usr/bin/env python
"""
RSNA-style figure: perfusion map mosaic + AIF curve + territory map.

Three figures, all heart-cropped:
  fig_A_perfusion_mosaic.png  — 2x4 axial slices spanning heart, perfusion
                                 overlay + ΔHU + V2 + per-tree territory
  fig_B_aif_curves.png        — AIF C(t) (clinical gamma) + sampled mean
                                 LV-blood-pool / mean-myo / per-tree-myo
                                 ΔHU(t) curves over the 31 frames
  fig_C_territory_perfusion.png — territory map (LAD/LCX/RCA color) + per-
                                  territory perfusion histogram + axial slice
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import gridspec
import toml
from scipy.integrate import trapezoid

PIPELINE_DIR = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline"
INTER_DIR  = os.path.join(PIPELINE_DIR, "intermediate")
FRAMES_DIR = os.path.join(PIPELINE_DIR, "frames")
OUT_DIR    = os.path.join(PIPELINE_DIR, "output")
RSNA_DIR   = os.path.join(OUT_DIR, "rsna_figs"); os.makedirs(RSNA_DIR, exist_ok=True)

PHANTOM_LABELS_PATH = "/home/molloi-lab/smb_mount/shared_drive/Shu Nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"
NX, NY, NZ = 1600, 1400, 500
# Heart bbox (label 15 LV myo): z=[35,419] y=[317,775] x=[831,1216]
X0, X1 = 800, 1280
Y0, Y1 = 280, 820
SLICES_Z = [120, 160, 200, 240, 280, 320, 360]   # 7 axial slices spanning heart

meta = toml.load(os.path.join(INTER_DIR, "metadata.toml"))
AIF = meta["aif_clinical"]
HU_PER_MG_ML = float(meta["hu_per_mg_ml_iodine"])
N_FRAMES = int(meta["frames"]["n_frames"])

def gamma_aif(t):
    if t <= AIF["t0_s"]: return 0.0
    tp = (t - AIF["t0_s"]) / (AIF["tmax_s"] - AIF["t0_s"])
    return 0.0 if tp <= 0 else AIF["amplitude_mg_ml"] * tp**AIF["alpha"] * np.exp(AIF["alpha"] * (1.0 - tp))

# ──────────────────────────────────────────────────────────────────────────
# Helpers — read one z-slab from a Fortran-order phantom raw (x-fastest)
# Reshape (NY, NX) order='C' gives arr[j, i] = val(i, j) → imshow row=y, col=x.
# ──────────────────────────────────────────────────────────────────────────
def read_int16_slab(path, z):
    with open(path, "rb") as f:
        f.seek(z * NX * NY * 2)
        return np.frombuffer(f.read(NX * NY * 2), dtype=np.int16).reshape((NY, NX), order="C")

def read_uint8_slab(path, z):
    with open(path, "rb") as f:
        f.seek(z * NX * NY)
        return np.frombuffer(f.read(NX * NY), dtype=np.uint8).reshape((NY, NX), order="C")

def crop(slab):
    return slab[Y0:Y1, X0:X1].copy()

# ──────────────────────────────────────────────────────────────────────────
# Fig A — perfusion mosaic
# ──────────────────────────────────────────────────────────────────────────
print("[rsna] Loading perfusion_map.npy …")
perf = np.load(os.path.join(OUT_DIR, "perfusion_map.npy"))   # (NX, NY, NZ) F-saved by np.save
print(f"[rsna]   shape={perf.shape} dtype={perf.dtype}")

# perf saved as np.save which keeps numpy array shape but np.save uses C-order by default;
# our perf was created with shape (NX, NY, NZ) so np.save preserves that. Then perf[X0:X1, Y0:Y1, z]
# gives (X-range, Y-range) — needs .T to (Y, X) for imshow.

n_slices = len(SLICES_Z)
fig = plt.figure(figsize=(n_slices * 3.2, 9.5))
gs = gridspec.GridSpec(3, n_slices, figure=fig, height_ratios=[1, 1, 1], hspace=0.10, wspace=0.05)

for k, z in enumerate(SLICES_Z):
    v2 = crop(read_int16_slab(os.path.join(FRAMES_DIR, "frame_10.raw"), z))
    v1 = crop(read_int16_slab(os.path.join(FRAMES_DIR, "frame_00.raw"), z))
    dhu = (v2.astype(np.int32) - v1.astype(np.int32)).astype(np.float32)
    perf_slice = perf[X0:X1, Y0:Y1, z].T

    # row 0: V2 (post-contrast)
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(v2, cmap="gray", vmin=-100, vmax=400)
    ax.set_title(f"z={z}", fontsize=11)
    if k == 0: ax.set_ylabel("V2 (t=10s)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # row 1: ΔHU map
    ax = fig.add_subplot(gs[1, k])
    im = ax.imshow(dhu, cmap="hot", vmin=0, vmax=200)
    if k == 0: ax.set_ylabel("ΔHU = V2 − V1", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    if k == n_slices - 1:
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("HU", fontsize=9)

    # row 2: Perfusion overlay
    ax = fig.add_subplot(gs[2, k])
    ax.imshow(v2, cmap="gray", vmin=-100, vmax=400)
    masked = np.ma.masked_invalid(perf_slice)
    im = ax.imshow(masked, cmap="jet", vmin=0, vmax=4, alpha=0.75)
    if k == 0: ax.set_ylabel("Perfusion (mL/min/g)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    if k == n_slices - 1:
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("mL/min/g", fontsize=9)

fig.suptitle("Simulated CT myocardial perfusion mosaic — XCAT vmale50 + 6 µm tree + Pries-Poiseuille\n"
             "AIF clinical gamma (8 mg/mL @ 8 s), V2 = t=10 s peak", fontsize=12, y=0.98)
fig_a_path = os.path.join(RSNA_DIR, "fig_A_perfusion_mosaic.png")
fig.savefig(fig_a_path, dpi=140, bbox_inches="tight"); plt.close()
print(f"[rsna] Wrote {fig_a_path}")

# ──────────────────────────────────────────────────────────────────────────
# Fig B — AIF + tissue time curves
# ──────────────────────────────────────────────────────────────────────────
print("[rsna] Sampling tissue time curves over 31 frames …")
labels = np.fromfile(PHANTOM_LABELS_PATH, dtype=np.uint8)
labels_3d = labels.reshape((NX, NY, NZ), order="F")
myo = (labels_3d >= 15) & (labels_3d <= 18)
lv  = (labels_3d == 19)
del labels, labels_3d

tid = np.fromfile(os.path.join(INTER_DIR, "myo_tree_id_patched.raw"),
                  dtype=np.uint8).reshape((NX, NY, NZ), order="F")
masks = {"LV pool":     lv,
         "myo (all)":   myo,
         "myo (LAD)":   myo & (tid == 1),
         "myo (LCX)":   myo & (tid == 2),
         "myo (RCA)":   myo & (tid == 3)}
del lv, tid

# Mean ΔHU per frame for each mask. Read frame_00 once, then each frame, subtract mean over mask.
print("[rsna]   reading frame 00 …")
v0 = np.fromfile(os.path.join(FRAMES_DIR, "frame_00.raw"),
                 dtype=np.int16).reshape((NX, NY, NZ), order="F").astype(np.int32)
mean_v0 = {name: float(v0[m].mean()) for name, m in masks.items()}
del v0

times = np.arange(N_FRAMES, dtype=float)
mean_v_curves = {name: np.zeros(N_FRAMES) for name in masks}
for fi in range(N_FRAMES):
    vt = np.fromfile(os.path.join(FRAMES_DIR, f"frame_{fi:02d}.raw"),
                     dtype=np.int16).reshape((NX, NY, NZ), order="F").astype(np.int32)
    for name, m in masks.items():
        mean_v_curves[name][fi] = float(vt[m].mean()) - mean_v0[name]
    print(f"[rsna]   frame {fi}: LV ΔHU={mean_v_curves['LV pool'][fi]:.1f}  myo ΔHU={mean_v_curves['myo (all)'][fi]:.1f}")
    del vt

# Theoretical AIF
t_fine = np.linspace(0, N_FRAMES - 1, 1001)
c_fine = np.array([gamma_aif(t) for t in t_fine])
aif_hu = HU_PER_MG_ML * c_fine

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(t_fine, aif_hu, color="black", linestyle="--", linewidth=1.4,
        label="theoretical AIF (HU)")
# LV pool tracks AIF closely. Use a thick navy solid line.
ax.plot(times, mean_v_curves["LV pool"], color="#000080", linestyle='-',
        linewidth=2.2, marker='o', markersize=6, label="LV pool")
# Combined myocardium curve as the headline (thick solid black, no marker overlap).
ax.plot(times, mean_v_curves["myo (all)"], color="black", linestyle='-',
        linewidth=2.5, label="myocardium (all)")
# Per-territory curves overlap exactly (uniform-arrival simulation reflects
# autoregulation in healthy myocardium). Plot each with a distinguishing dash
# pattern + thin line + own marker shape so all three are individually visible.
ax.plot(times, mean_v_curves["myo (LAD)"], color="#1f77ff", linestyle='--',
        linewidth=1.4, marker='v', markersize=7, alpha=0.85, label="LAD territory")
ax.plot(times, mean_v_curves["myo (LCX)"], color="#e3342f", linestyle='-.',
        linewidth=1.4, marker='^', markersize=7, alpha=0.85, label="LCX territory")
ax.plot(times, mean_v_curves["myo (RCA)"], color="#22aa44", linestyle=':',
        linewidth=1.6, marker='D', markersize=6, alpha=0.85, label="RCA territory")

ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("ΔHU (mean over region)", fontsize=12)
ax.set_title("Time-density curves — clinical gamma AIF, ECV-scaled myocardial response\n"
             "(LAD/LCX/RCA overlap: uniform-arrival autoregulation, no stenosis)",
             fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
fig_b_path = os.path.join(RSNA_DIR, "fig_B_time_curves.png")
fig.savefig(fig_b_path, dpi=140, bbox_inches="tight"); plt.close()
print(f"[rsna] Wrote {fig_b_path}")

# ──────────────────────────────────────────────────────────────────────────
# Fig C — territory map + per-tree perfusion histograms
# ──────────────────────────────────────────────────────────────────────────
print("[rsna] Building territory + per-tree perfusion histogram …")
tid = np.fromfile(os.path.join(INTER_DIR, "myo_tree_id_patched.raw"),
                  dtype=np.uint8).reshape((NX, NY, NZ), order="F")

fig = plt.figure(figsize=(13, 6))
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.4], wspace=0.20)

# Left: territory mid-axial slice (color by tree)
ax = fig.add_subplot(gs[0, 0])
z_mid = 200
v2 = crop(read_int16_slab(os.path.join(FRAMES_DIR, "frame_10.raw"), z_mid))
tid_slice = tid[X0:X1, Y0:Y1, z_mid].T
ax.imshow(v2, cmap="gray", vmin=-100, vmax=400)
# Color overlay per tree id
overlay = np.zeros(tid_slice.shape + (4,), dtype=float)
overlay[tid_slice == 1] = (0.12, 0.47, 1.0, 0.7)   # LAD blue
overlay[tid_slice == 2] = (0.89, 0.20, 0.18, 0.7)  # LCX red
overlay[tid_slice == 3] = (0.13, 0.67, 0.27, 0.7)  # RCA green
ax.imshow(overlay)
ax.set_title(f"Territory map (z={z_mid})", fontsize=12); ax.axis("off")
# Custom legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#1f77ff", label="LAD"),
                   Patch(facecolor="#e3342f", label="LCX"),
                   Patch(facecolor="#22aa44", label="RCA")],
          loc="lower right", fontsize=10)

# Right: per-territory perfusion histogram. With uniform arrival the three
# territories collapse onto identical distributions; we use histtype='step'
# (line outlines, no fill) and three distinct line styles so all three curves
# are individually visible even when they overlap perfectly.
ax = fig.add_subplot(gs[0, 1])
n_bins = 80; bin_max = 4.0
styles = [("LAD", 1, "#1f77ff", '-',  2.6),
          ("LCX", 2, "#e3342f", '--', 2.0),
          ("RCA", 3, "#22aa44", ':',  2.2)]
max_y = 0
for name, t_id, color, ls, lw in styles:
    sel = (tid == t_id)
    samp = perf[sel]
    finite = samp[np.isfinite(samp) & (samp > 0)]
    if len(finite) > 1_000_000:
        finite = np.random.RandomState(42).choice(finite, 1_000_000, replace=False)
    counts, edges, _ = ax.hist(np.clip(finite, 0, bin_max), bins=n_bins,
                                histtype='step', linewidth=lw, linestyle=ls,
                                color=color,
                                label=f"{name} (n={len(samp)/1e6:.1f}M, μ={samp[np.isfinite(samp)].mean():.2f} mL/min/g)")
    if counts.max() > max_y: max_y = counts.max()

ax.set_xlim(0, bin_max)
ax.set_ylim(0, max_y * 1.10)
ax.set_xlabel("Perfusion (mL/min/g)", fontsize=12)
ax.set_ylabel("voxel count", fontsize=12)
ax.set_title("Per-territory perfusion histogram (V2=10 s)\n"
             "(LAD/LCX/RCA overlap at baseline — uniform autoregulation)",
             fontsize=11)
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
fig_c_path = os.path.join(RSNA_DIR, "fig_C_territory.png")
fig.savefig(fig_c_path, dpi=140, bbox_inches="tight"); plt.close()
print(f"[rsna] Wrote {fig_c_path}")

print("\n[rsna] Done.")
print(f"  Figures: {RSNA_DIR}/")
print("    fig_A_perfusion_mosaic.png")
print("    fig_B_time_curves.png")
print("    fig_C_territory.png")
