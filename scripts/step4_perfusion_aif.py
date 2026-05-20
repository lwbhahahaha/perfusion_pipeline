#!/usr/bin/env python3
"""
Step 4 (AIF variant): feed simulated raw frames into the Mullani-Gould single-
volume perfusion formula, using a REAL patient AIF (from step0_prepare_aif.py
and step3_voxelize_frames_aif.jl).

AUC is integrated numerically from the real AIF curve, replacing the
analytical-gamma AUC used in step4_perfusion.py.

Output:
    output/perfusion_map.npy    Float32 perfusion map (mL/min/g)
    output/flow_map.npy         Float32 flow map (mL/min)
    output/summary.txt          Scalar perfusion stats
"""

import os
import sys
import argparse
import numpy as np
try:
    import tomllib  # Python 3.11+
    def _load_toml(path):
        with open(path, "rb") as f:
            return tomllib.load(f)
except ImportError:
    try:
        import tomli
        def _load_toml(path):
            with open(path, "rb") as f:
                return tomli.load(f)
    except ImportError:
        import toml
        def _load_toml(path):
            return toml.load(path)


def compute_organ_metrics_inline(v2, mask, v1, input_conc, voxel_size_mm,
                                 tissue_rho=1.053):
    """Single-volume CT perfusion (Mullani-Gould mass-balance).
    Identical math to step4_perfusion.py; only AUC source differs upstream.
    """
    v2 = v2.astype(np.float64)
    mask = mask.astype(bool)
    n_mask = int(mask.sum())
    voxel_vol_cm3 = voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2] / 1000.0
    organ_mass = n_mask * tissue_rho * voxel_vol_cm3   # g
    organ_vol_inplane = voxel_vol_cm3                  # per-voxel volume cm³

    v1_arr = v2.copy()
    v1_arr[mask] = v1
    v1_arr[~mask] = np.nan
    v2_nan = v2.copy()
    v2_nan[~mask] = np.nan

    delta_hu = np.mean(v2_nan[mask]) - np.mean(v1_arr[mask])
    v1_mass = np.sum(v1_arr[mask]) * organ_vol_inplane
    v2_mass = np.sum(v2_nan[mask]) * organ_vol_inplane

    flow = (60.0 / input_conc) * (v2_mass - v1_mass)

    flow_map = (v2_nan - v1_arr) / delta_hu * flow
    flow_std = float(np.std(flow_map[mask]))
    perf_map = flow_map / organ_mass
    perf_std = float(np.std(perf_map[mask]))
    perf     = flow / organ_mass

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

NX, NY, NZ = 1600, 1400, 500
VOXEL_MM = 0.2
TISSUE_RHO = 1.053


def load_aif_csv(path):
    """Load CSV produced by step0_prepare_aif.py.
    Returns (t_s, C_mg_per_mL) as 1-D numpy arrays sorted by time."""
    t_list, c_list = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("time_s"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            t_list.append(float(parts[0]))
            c_list.append(float(parts[1]))
    t = np.asarray(t_list, dtype=np.float64)
    c = np.asarray(c_list, dtype=np.float64)
    order = np.argsort(t)
    return t[order], c[order]


def aif_auc_hu_s(t, c_mg_per_ml, hu_per_mg_ml, t_max_s):
    """AUC of AIF up to t_max_s, in HU·s.
    Numerical trapezoidal integration on a fine grid (10 ms)."""
    trapz_fn = getattr(np, "trapezoid", np.trapz)
    # Clip integration window to [0, min(t_end_of_aif, t_max_s)]
    t_hi = min(float(t[-1]), float(t_max_s))
    if t_hi <= 0:
        return 0.0
    t_fine = np.linspace(0.0, t_hi, max(int(t_hi / 0.01), 2))
    c_fine = np.interp(t_fine, t, c_mg_per_ml, left=0.0, right=c_mg_per_ml[-1])
    auc_mg_s_per_ml = float(trapz_fn(c_fine, t_fine))
    return auc_mg_s_per_ml * hu_per_mg_ml, auc_mg_s_per_ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aif_csv",
                    help="real-AIF CSV from step0_prepare_aif.py")
    ap.add_argument("--v2_t", type=int, default=20,
                    help="V2 frame time in seconds (default 20)")
    ap.add_argument("--out_suffix", type=str, default="",
                    help="suffix appended to output filenames (e.g., '_subj002')")
    args = ap.parse_args()

    meta = _load_toml(os.path.join(INTER_DIR, "metadata.toml"))
    HU_PER_MG_ML = float(meta["hu_per_mg_ml_iodine"])
    DT_S = float(meta["frames"]["dt_s"])
    T_END_S = float(meta["frames"]["t_end_s"])
    N_FRAMES = int(meta["frames"]["n_frames"])

    print(f"[step4-aif] aif_csv = {args.aif_csv}")
    if not os.path.isfile(args.aif_csv):
        print(f"ERROR: AIF CSV not found: {args.aif_csv}", file=sys.stderr)
        sys.exit(1)

    V2_T = args.v2_t
    assert 0 <= V2_T < N_FRAMES, f"v2_t must be 0..{N_FRAMES-1}"

    t_aif, c_aif = load_aif_csv(args.aif_csv)
    print(f"[step4-aif] AIF: {len(t_aif)} samples, t ∈ [{t_aif[0]:.3f}, {t_aif[-1]:.3f}] s, "
          f"C ∈ [{c_aif.min():.4f}, {c_aif.max():.4f}] mg/mL")
    print(f"[step4-aif] HU_per_mg_ml = {HU_PER_MG_ML}")
    print(f"[step4-aif] V1 = frame_00.raw (t=0s, baseline)")
    print(f"[step4-aif] V2 = frame_{V2_T:02d}.raw (t={V2_T}s, contrast peak)")

    auc_hu_s, auc_mg_s_per_ml = aif_auc_hu_s(t_aif, c_aif, HU_PER_MG_ML, float(V2_T))
    print(f"[step4-aif] AIF AUC(0..{V2_T}s) = {auc_mg_s_per_ml:.3f} mg·s/mL = {auc_hu_s:.2f} HU·s")
    c_at_v2 = float(np.interp(V2_T, t_aif, c_aif, left=0.0, right=c_aif[-1]))
    print(f"[step4-aif] AIF C(t={V2_T}s) = {c_at_v2:.4f} mg/mL = {HU_PER_MG_ML*c_at_v2:.2f} HU above baseline")

    # ── Load V1 (baseline), V2 (peak) ──
    def load_raw_int16(path):
        arr = np.fromfile(path, dtype=np.int16)
        return arr.reshape((NX, NY, NZ), order="F")

    V1 = load_raw_int16(os.path.join(FRAMES_DIR, "frame_00.raw"))
    V2 = load_raw_int16(os.path.join(FRAMES_DIR, f"frame_{V2_T:02d}.raw"))
    print(f"[step4-aif]   V1 HU range [{V1.min()}, {V1.max()}]")
    print(f"[step4-aif]   V2 HU range [{V2.min()}, {V2.max()}]")

    # ── Build myocardium mask from XCAT labels (15..18) ──
    labels = np.fromfile(PHANTOM_LABELS_PATH, dtype=np.uint8).reshape(
        (NX, NY, NZ), order="F")
    myo_mask = ((labels >= 15) & (labels <= 18)).astype(np.uint8)
    n_myo = int(myo_mask.sum())
    mass_g = n_myo * TISSUE_RHO * (VOXEL_MM**3) / 1000.0
    print(f"[step4-aif]   myocardium mask: {n_myo} voxels = {mass_g:.1f} g (rho=1.053)")
    del labels

    v1_mean_hu = float(V1[myo_mask.astype(bool)].mean())
    v2_mean_hu = float(V2[myo_mask.astype(bool)].mean())
    print(f"[step4-aif]   V1 mean HU in myo = {v1_mean_hu:.2f}")
    print(f"[step4-aif]   V2 mean HU in myo = {v2_mean_hu:.2f}  (ΔHU = {v2_mean_hu-v1_mean_hu:.2f})")

    print("\n[step4-aif] Computing single-volume perfusion …")
    del V1
    result = compute_organ_metrics_inline(
        V2, myo_mask, v1_mean_hu, auc_hu_s,
        voxel_size_mm=(VOXEL_MM, VOXEL_MM, VOXEL_MM),
        tissue_rho=TISSUE_RHO)
    del V2

    print("\n[step4-aif] === PERFUSION RESULT ===")
    print(f"  organ mass:       {result['organ_mass']:.2f} g")
    print(f"  delta HU (mean):  {result['delta_hu']:.2f} HU")
    print(f"  flow:             {result['flow']:.2f} mL/min")
    print(f"  flow std:         {result['flow_std']:.2f}")
    print(f"  perfusion (mean): {result['perf']:.4f} mL/min/g")
    print(f"  perfusion std:    {result['perf_std']:.4f}")

    # Sanity check vs clinical range
    clinical_rest_range = (0.5, 1.5)
    clinical_stress_range = (2.0, 5.0)
    print(f"\n[step4-aif] Clinical sanity ranges (mL/min/g):")
    print(f"  REST:   {clinical_rest_range[0]:.2f}–{clinical_rest_range[1]:.2f}")
    print(f"  STRESS: {clinical_stress_range[0]:.2f}–{clinical_stress_range[1]:.2f}")
    perf = result['perf']
    in_rest = clinical_rest_range[0] <= perf <= clinical_rest_range[1]
    in_stress = clinical_stress_range[0] <= perf <= clinical_stress_range[1]
    if in_rest:
        print(f"  ★ {perf:.3f} is within REST range — consistent with healthy at-rest CT perfusion")
    elif in_stress:
        print(f"  ★ {perf:.3f} is within STRESS range — consistent with hyperemic CT perfusion")
    else:
        print(f"  ⚠ {perf:.3f} is outside both clinical ranges; inspect inputs")

    # ── Save outputs ──
    suffix = args.out_suffix
    print("\n[step4-aif] Saving outputs …")
    p_path = os.path.join(OUT_DIR, f"perfusion_map{suffix}.npy")
    f_path = os.path.join(OUT_DIR, f"flow_map{suffix}.npy")
    s_path = os.path.join(OUT_DIR, f"summary{suffix}.txt")
    np.save(p_path, result["perf_map"].astype(np.float32))
    np.save(f_path, result["flow_map"].astype(np.float32))
    with open(s_path, "w") as f:
        f.write(f"# Step4-AIF perfusion summary\n")
        f.write(f"aif_csv             = {args.aif_csv}\n")
        f.write(f"V2_t                = {V2_T}s\n")
        f.write(f"AIF_AUC_HUs         = {auc_hu_s:.4f}\n")
        f.write(f"AIF_AUC_mg_s_per_mL = {auc_mg_s_per_ml:.4f}\n")
        f.write(f"V1_mean_HU          = {v1_mean_hu:.4f}\n")
        f.write(f"V2_mean_HU          = {v2_mean_hu:.4f}\n")
        f.write(f"organ_mass_g        = {result['organ_mass']:.4f}\n")
        f.write(f"delta_HU_mean       = {result['delta_hu']:.4f}\n")
        f.write(f"flow_mlmin          = {result['flow']:.4f}\n")
        f.write(f"flow_std            = {result['flow_std']:.4f}\n")
        f.write(f"perfusion_mean_mlmin_g = {result['perf']:.6f}\n")
        f.write(f"perfusion_std       = {result['perf_std']:.6f}\n")
    print(f"[step4-aif]   {p_path} ({result['perf_map'].nbytes/1e9:.2f} GB)")
    print(f"[step4-aif]   {f_path}")
    print(f"[step4-aif]   {s_path}")
    print("\n[step4-aif] Done.")


if __name__ == "__main__":
    main()
