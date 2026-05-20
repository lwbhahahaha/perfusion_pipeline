#!/usr/bin/env python
"""
Step 4: feed simulated raw frames into PerfusionImaging.compute_organ_metrics
to produce a final myocardial perfusion map (mL/min/g).

Uses the V1 = baseline frame (t=0, no contrast) and V2 = peak frame
(default t=10 s) approach of single_volume.ipynb. The AIF AUC is computed
analytically from the clinical gamma parameters in metadata.toml — we don't
need to fit gamma_plot since we know the input bolus shape exactly.

Output:
    output/perfusion_map.npy    Float32 perfusion map (mL/min/g)
    output/flow_map.npy         Float32 flow map (mL/min)
    output/summary.txt          Scalar perfusion stats
"""

import os
import sys
import argparse
import numpy as np
from scipy.integrate import trapezoid
import toml


# ─────────────────────────────────────────────────────────────────
# Inlined perfusion formula (mathematically identical to
# PerfusionImaging.tool.compute_organ_metrics; we don't import the
# package because it pulls in SimpleITK + antspyx as transitive deps).
# Reference: single_volume.ipynb in PerfusionImaging-main/.
# ─────────────────────────────────────────────────────────────────
def compute_organ_metrics_inline(v2, mask, v1, input_conc, voxel_size_mm,
                                 tissue_rho=1.053):
    """Single-volume CT perfusion (Mullani-Gould mass-balance).

    Parameters
    ----------
    v2 : ndarray (Int16 or Float32) — V2 image in HU
    mask : ndarray (bool or 0/1) — myocardium mask, same shape as v2
    v1 : float — pre-contrast baseline HU inside the mask
    input_conc : float — AIF AUC in HU·s
    voxel_size_mm : tuple (sx, sy, sz) — voxel spacing in mm
    tissue_rho : float — tissue density g/cm³ (myocardium ≈ 1.053)

    Returns dict with the same keys as PerfusionImaging.tool.compute_organ_metrics
    """
    v2 = v2.astype(np.float64)
    mask = mask.astype(bool)
    n_mask = int(mask.sum())
    voxel_vol_cm3 = voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2] / 1000.0
    organ_mass = n_mask * tissue_rho * voxel_vol_cm3   # g
    organ_vol_inplane = voxel_vol_cm3                  # per-voxel volume cm³

    # V1 array: scalar v1 inside mask, V2 values outside (will be NaNed)
    v1_arr = v2.copy()
    v1_arr[mask] = v1
    # NaN outside mask for both
    v1_arr[~mask] = np.nan
    v2_nan = v2.copy()
    v2_nan[~mask] = np.nan

    delta_hu = np.mean(v2_nan[mask]) - np.mean(v1_arr[mask])    # baseline-subtracted enhancement
    v1_mass = np.sum(v1_arr[mask]) * organ_vol_inplane          # HU·cm³
    v2_mass = np.sum(v2_nan[mask]) * organ_vol_inplane          # HU·cm³

    flow = (60.0 / input_conc) * (v2_mass - v1_mass)            # mL/min

    # Voxel-wise flow distribution proportional to local enhancement
    flow_map = (v2_nan - v1_arr) / delta_hu * flow
    flow_std = float(np.std(flow_map[mask]))
    perf_map = flow_map / organ_mass                            # mL/min/g
    perf_std = float(np.std(perf_map[mask]))
    perf     = flow / organ_mass                                # mL/min/g (mean)

    return dict(organ_mass=organ_mass, delta_hu=float(delta_hu),
                organ_vol_inplane=organ_vol_inplane,
                v1_mass=float(v1_mass), v2_mass=float(v2_mass),
                flow=float(flow), flow_map=flow_map.astype(np.float32),
                flow_std=flow_std,
                perf_map=perf_map.astype(np.float32), perf_std=perf_std,
                perf=float(perf))

PHANTOM_LABELS_PATH = "/home/molloi-lab/smb_mount/shared_drive/Shu Nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"
PIPELINE_DIR = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline"
INTER_DIR    = os.path.join(PIPELINE_DIR, "intermediate")
FRAMES_DIR   = os.path.join(PIPELINE_DIR, "frames")
OUT_DIR      = os.path.join(PIPELINE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# Phantom dims (x, y, z) per the .raw layout — Fortran order from XCAT
NX, NY, NZ = 1600, 1400, 500
VOXEL_MM = 0.2  # = 0.02 cm
TISSUE_RHO = 1.053  # g/cm³ for myocardium

# ── Load metadata ──
meta = toml.load(os.path.join(INTER_DIR, "metadata.toml"))
AIF = meta["aif_clinical"]
HU_PER_MG_ML = float(meta["hu_per_mg_ml_iodine"])
DT_S = float(meta["frames"]["dt_s"])
T_END_S = float(meta["frames"]["t_end_s"])
N_FRAMES = int(meta["frames"]["n_frames"])

print(f"[step4] AIF: amp={AIF['amplitude_mg_ml']} mg/mL, "
      f"t0={AIF['t0_s']}s, tmax={AIF['tmax_s']}s, alpha={AIF['alpha']}")
print(f"[step4] Frames: 0..{T_END_S}s @ {DT_S}s ({N_FRAMES} files)")

def gamma_aif(t):
    """Clinical injected bolus gamma-variate, units mg/mL."""
    if t <= AIF["t0_s"]:
        return 0.0
    tp = (t - AIF["t0_s"]) / (AIF["tmax_s"] - AIF["t0_s"])
    if tp <= 0:
        return 0.0
    return AIF["amplitude_mg_ml"] * tp**AIF["alpha"] * np.exp(
        AIF["alpha"] * (1.0 - tp))

# ── CLI: pick V2 time ──
ap = argparse.ArgumentParser()
ap.add_argument("--v2_t", type=int, default=20,
                help="V2 frame time in seconds (default 20, well past AIF peak+disp for slow territory)")
args = ap.parse_args()
V2_T = args.v2_t
assert 0 <= V2_T < N_FRAMES, f"v2_t must be 0..{N_FRAMES-1}"
print(f"[step4] V1 = frame_00.raw (t=0s, baseline)")
print(f"[step4] V2 = frame_{V2_T:02d}.raw (t={V2_T}s, contrast peak)")

# ── Load V1 (baseline), V2 (peak) ──
def load_raw_int16(path):
    """Load a 1600x1400x500 Int16 raw, return (nz, ny, nx) C-order array (ANTs convention)."""
    arr = np.fromfile(path, dtype=np.int16)
    arr = arr.reshape((NX, NY, NZ), order="F")  # XCAT raw is x-fastest
    return arr  # shape (NX, NY, NZ)

V1 = load_raw_int16(os.path.join(FRAMES_DIR, "frame_00.raw"))
V2 = load_raw_int16(os.path.join(FRAMES_DIR, f"frame_{V2_T:02d}.raw"))
print(f"[step4]   V1 shape={V1.shape} dtype={V1.dtype} HU range [{V1.min()}, {V1.max()}]")
print(f"[step4]   V2 shape={V2.shape} dtype={V2.dtype} HU range [{V2.min()}, {V2.max()}]")

# ── Build myocardium mask from XCAT labels (15..18) ──
labels = np.fromfile(PHANTOM_LABELS_PATH, dtype=np.uint8).reshape(
    (NX, NY, NZ), order="F")
myo_mask = ((labels >= 15) & (labels <= 18)).astype(np.uint8)
n_myo = int(myo_mask.sum())
mass_g = n_myo * TISSUE_RHO * (VOXEL_MM**3) / 1000.0  # mm³ → cm³, then g
print(f"[step4]   myocardium mask: {n_myo} voxels = {mass_g:.1f} g (rho=1.053)")
del labels  # free

# ── Compute AUC of AIF analytically ──
# Integrate gamma(t) from 0 to V2_T (mg·s/mL), then × 25 → HU·s
t_fine = np.linspace(0.0, float(V2_T), 5001)
c_fine = np.array([gamma_aif(t) for t in t_fine])
auc_mg_s_per_ml = float(trapezoid(c_fine, t_fine))
auc_hu_s = HU_PER_MG_ML * auc_mg_s_per_ml
print(f"[step4]   AIF AUC(0..{V2_T}s) = {auc_mg_s_per_ml:.2f} mg·s/mL = {auc_hu_s:.1f} HU·s")
print(f"[step4]   AIF peak C(t={V2_T}) = {gamma_aif(V2_T):.2f} mg/mL = {HU_PER_MG_ML*gamma_aif(V2_T):.1f} HU above baseline")

# ── V1 mean HU in mask (used as the pre-contrast myocardial baseline) ──
v1_mean_hu = float(V1[myo_mask.astype(bool)].mean())
v2_mean_hu = float(V2[myo_mask.astype(bool)].mean())
print(f"[step4]   V1 mean HU in myo = {v1_mean_hu:.2f}")
print(f"[step4]   V2 mean HU in myo = {v2_mean_hu:.2f}  (ΔHU = {v2_mean_hu-v1_mean_hu:.2f})")

# ── Run inlined Mullani-Gould perfusion formula ──
print("\n[step4] Computing single-volume perfusion …")
del V1   # free V1 raw, only need mean scalar v1_mean_hu
result = compute_organ_metrics_inline(
    V2, myo_mask, v1_mean_hu, auc_hu_s,
    voxel_size_mm=(VOXEL_MM, VOXEL_MM, VOXEL_MM),
    tissue_rho=TISSUE_RHO)
del V2

# ── Report ──
print("\n[step4] === PERFUSION RESULT ===")
print(f"  organ mass:       {result['organ_mass']:.2f} g")
print(f"  delta HU (mean):  {result['delta_hu']:.2f} HU")
print(f"  flow:             {result['flow']:.2f} mL/min")
print(f"  flow std:         {result['flow_std']:.2f}")
print(f"  perfusion (mean): {result['perf']:.4f} mL/min/g")
print(f"  perfusion std:    {result['perf_std']:.4f}")

# Compare with FlowContrastSim natural-flow numbers (from CLAUDE.md memory)
nat_flow_mlmin = 523.7 + 271.7 + 378.3   # LAD + LCX + RCA, natural Pries+Poiseuille @ 100→15 mmHg
print(f"\n[step4] Cross-check vs FlowContrastSim natural per-tree sums:")
print(f"  Σ root_flow (Poiseuille forward) = {nat_flow_mlmin:.1f} mL/min")
print(f"  perfusion_code flow (single-volume) = {result['flow']:.1f} mL/min")
print(f"  ratio = {result['flow']/nat_flow_mlmin:.2f}")
print(f"  (mismatch indicates AIF mass-balance vs Poiseuille model differ;")
print(f"   close to 1.0 means the perfusion bridge is internally consistent.)")

# ── Save outputs ──
print("\n[step4] Saving outputs …")
np.save(os.path.join(OUT_DIR, "perfusion_map.npy"), result["perf_map"].astype(np.float32))
np.save(os.path.join(OUT_DIR, "flow_map.npy"),      result["flow_map"].astype(np.float32))
with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write(f"V2_t                = {V2_T}s\n")
    f.write(f"AIF_AUC_HUs         = {auc_hu_s:.4f}\n")
    f.write(f"V1_mean_HU          = {v1_mean_hu:.4f}\n")
    f.write(f"V2_mean_HU          = {v2_mean_hu:.4f}\n")
    f.write(f"organ_mass_g        = {result['organ_mass']:.4f}\n")
    f.write(f"delta_HU_mean       = {result['delta_hu']:.4f}\n")
    f.write(f"flow_mlmin          = {result['flow']:.4f}\n")
    f.write(f"flow_std            = {result['flow_std']:.4f}\n")
    f.write(f"perfusion_mean_mlmin_g = {result['perf']:.6f}\n")
    f.write(f"perfusion_std       = {result['perf_std']:.6f}\n")
print(f"[step4]   {OUT_DIR}/perfusion_map.npy ({result['perf_map'].nbytes/1e9:.2f} GB)")
print(f"[step4]   {OUT_DIR}/flow_map.npy")
print(f"[step4]   {OUT_DIR}/summary.txt")
print("\n[step4] Done.")
