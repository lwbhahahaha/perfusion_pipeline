#!/usr/bin/env python
"""
Step 4b: max-projection perfusion.

Single-volume CT perfusion picks a single V2 time, but with our wide
arrival-time distribution (median 1.08s, p95 15.4s) no single time
captures every myocardium voxel at its peak. This script instead loops
over all 31 frames and stores, per-voxel:
    - V2_max[i,j,k] = max(frame_t[i,j,k]) over all t ∈ [0, 30s]
    - t_peak[i,j,k] = arg max time
Then runs the Mullani-Gould formula using V2_max as the "V2" image and
the AIF AUC integrated up to the actual t_peak per voxel ("voxel-specific
AUC"). This recovers the per-voxel peak enhancement regardless of the
voxel's individual transit time.
"""

import os, sys, numpy as np, toml
from scipy.integrate import trapezoid

PIPELINE_DIR = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline"
INTER_DIR  = os.path.join(PIPELINE_DIR, "intermediate")
FRAMES_DIR = os.path.join(PIPELINE_DIR, "frames")
OUT_DIR    = os.path.join(PIPELINE_DIR, "output")
PHANTOM_LABELS_PATH = "/home/molloi-lab/smb_mount/shared_drive/Shu Nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"

NX, NY, NZ = 1600, 1400, 500
VOXEL_MM   = 0.2
TISSUE_RHO = 1.053

meta = toml.load(os.path.join(INTER_DIR, "metadata.toml"))
AIF = meta["aif_clinical"]
HU_PER_MG_ML = float(meta["hu_per_mg_ml_iodine"])
N_FRAMES = int(meta["frames"]["n_frames"])
DT = float(meta["frames"]["dt_s"])

def gamma_aif(t):
    if t <= AIF["t0_s"]: return 0.0
    tp = (t - AIF["t0_s"]) / (AIF["tmax_s"] - AIF["t0_s"])
    return 0.0 if tp <= 0 else AIF["amplitude_mg_ml"] * tp**AIF["alpha"] * np.exp(AIF["alpha"] * (1.0 - tp))

# ── Load V1 + myo mask ──
def load_int16(path):
    return np.fromfile(path, dtype=np.int16).reshape((NX, NY, NZ), order="F")

print("[step4b] Loading V1 (frame_00) and myo mask …")
V1 = load_int16(os.path.join(FRAMES_DIR, "frame_00.raw"))
labels = np.fromfile(PHANTOM_LABELS_PATH, dtype=np.uint8).reshape((NX, NY, NZ), order="F")
myo_mask = ((labels >= 15) & (labels <= 18))
n_myo = int(myo_mask.sum())
print(f"[step4b]   myo voxels = {n_myo}")
del labels

# ── Sweep frames; track per-voxel max ΔHU and its time ──
print(f"[step4b] Scanning {N_FRAMES} frames for per-voxel max ΔHU …")
max_dhu = np.zeros((NX, NY, NZ), dtype=np.int16)        # peak ΔHU value
t_peak  = np.zeros((NX, NY, NZ), dtype=np.uint8)        # frame index of peak
v1_int = V1.astype(np.int32)
for fi in range(N_FRAMES):
    fpath = os.path.join(FRAMES_DIR, f"frame_{fi:02d}.raw")
    Vt = load_int16(fpath).astype(np.int32)
    delta = Vt - v1_int
    upd = delta > max_dhu.astype(np.int32)
    max_dhu[upd] = delta[upd].astype(np.int16)
    t_peak[upd]  = fi
    print(f"[step4b]   t={fi}s  ΔHU_max(myo)={int(delta[myo_mask].max())}  ΔHU_mean(myo)={float(delta[myo_mask].mean()):.1f}")
    del Vt, delta, upd
del v1_int, V1

# ── Per-voxel AIF AUC integrated up to that voxel's t_peak ──
print("[step4b] Building per-voxel AIF-AUC table by t_peak …")
aif_auc_table = np.zeros(N_FRAMES, dtype=np.float64)
t_fine = np.linspace(0, (N_FRAMES - 1) * DT, 5001)
c_fine = np.array([gamma_aif(t) for t in t_fine])
for fi in range(N_FRAMES):
    mask_t = t_fine <= fi * DT
    if mask_t.sum() < 2:
        aif_auc_table[fi] = 0.0
    else:
        aif_auc_table[fi] = HU_PER_MG_ML * trapezoid(c_fine[mask_t], t_fine[mask_t])
print(f"[step4b]   AIF AUC at t=10: {aif_auc_table[10]:.1f} HU·s")
print(f"[step4b]   AIF AUC at t=20: {aif_auc_table[20]:.1f} HU·s")
print(f"[step4b]   AIF AUC at t=30: {aif_auc_table[30]:.1f} HU·s")

# ── Mullani-Gould per voxel using max-ΔHU and per-voxel AUC ──
print("[step4b] Computing per-voxel perfusion (max-projection) …")
voxel_vol_cm3 = VOXEL_MM**3 / 1000.0
organ_mass = n_myo * TISSUE_RHO * voxel_vol_cm3
print(f"[step4b]   organ mass = {organ_mass:.2f} g")

# perf_voxel = (ΔHU_voxel × voxel_volume) / AIF_AUC(t_peak_voxel) × 60 / mass
# Equivalent voxel-wise formula:
#   flow_voxel [mL/min] = 60 × ΔHU_voxel × voxel_volume_cm³ / AIF_AUC_at_peak [HU·s]
# Then divide by mass for perfusion mL/min/g.
perf_map = np.full((NX, NY, NZ), np.nan, dtype=np.float32)
flow_map = np.full((NX, NY, NZ), np.nan, dtype=np.float32)

# Vectorize: aif_auc per voxel from t_peak lookup
t_peak_int = t_peak[myo_mask].astype(np.int32)
auc_voxel = aif_auc_table[t_peak_int]
auc_voxel = np.where(auc_voxel > 0, auc_voxel, np.inf)
delta_hu_voxel = max_dhu[myo_mask].astype(np.float32)
flow_voxel = 60.0 * delta_hu_voxel * voxel_vol_cm3 / auc_voxel        # mL/min per voxel
perf_voxel = flow_voxel / (organ_mass / n_myo)                         # divide by per-voxel mass equiv

# Actually wait — perfusion should be flow per gram. Each voxel has mass voxel_volume*rho.
# So perf_voxel = flow_voxel / (voxel_volume_cm3 * tissue_rho)
voxel_mass_g = voxel_vol_cm3 * TISSUE_RHO
perf_voxel = flow_voxel / voxel_mass_g                                  # mL/min/g per voxel

flow_map[myo_mask] = flow_voxel
perf_map[myo_mask] = perf_voxel

mean_perf = float(np.nanmean(perf_map))
median_perf = float(np.nanmedian(perf_map[myo_mask]))
total_flow = float(np.nansum(flow_map))   # sum of flows (because each voxel is 1 voxel of organ)

print("\n[step4b] === MAX-PROJECTION PERFUSION RESULT ===")
print(f"  organ mass:       {organ_mass:.2f} g")
print(f"  mean ΔHU_max:     {float(np.nanmean(max_dhu[myo_mask])):.2f} HU")
print(f"  median ΔHU_max:   {float(np.nanmedian(max_dhu[myo_mask])):.2f} HU")
print(f"  total flow:       {total_flow:.2f} mL/min")
print(f"  perfusion mean:   {mean_perf:.4f} mL/min/g")
print(f"  perfusion median: {median_perf:.4f} mL/min/g")
print(f"  perfusion p10:    {float(np.nanpercentile(perf_map[myo_mask], 10)):.3f}")
print(f"  perfusion p90:    {float(np.nanpercentile(perf_map[myo_mask], 90)):.3f}")

# Save
np.save(os.path.join(OUT_DIR, "perfusion_map_maxproj.npy"), perf_map)
np.save(os.path.join(OUT_DIR, "flow_map_maxproj.npy"), flow_map)
np.save(os.path.join(OUT_DIR, "max_dhu_map.npy"), max_dhu)
np.save(os.path.join(OUT_DIR, "t_peak_map.npy"), t_peak)
with open(os.path.join(OUT_DIR, "summary_maxproj.txt"), "w") as f:
    f.write(f"organ_mass_g          = {organ_mass:.4f}\n")
    f.write(f"mean_dHU_max          = {float(np.nanmean(max_dhu[myo_mask])):.4f}\n")
    f.write(f"total_flow_mlmin      = {total_flow:.4f}\n")
    f.write(f"perfusion_mean_mlming = {mean_perf:.6f}\n")
    f.write(f"perfusion_median      = {median_perf:.6f}\n")
print(f"\n[step4b] Saved perfusion_map_maxproj.npy + flow_map_maxproj.npy + max_dhu_map.npy + t_peak_map.npy")
print("[step4b] Done.")
