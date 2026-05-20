#!/usr/bin/env python3
"""
Step 5 (Path B): compute perfusion map from BasisSim-reconstructed V1 + V2
Float32 HU volumes (512×512×250) instead of the deterministic phantom frames
of step4. This captures full CT physics noise (photon stats + recon noise +
beam hardening + scatter) so the resulting perfusion map has the same noise
character as a real CTP scan.

Key transforms:
  * BasisSim reverses phantom Y and Z when ingesting (see run_cta_sim_param.jl)
    and then downsamples by 2 (every-other voxel). We re-apply the same
    transforms to the myocardium-label mask so it aligns with the recon image.
  * Recon FOV (35×35×5 cm) is larger than the BasisSim-internal phantom extent
    (16×14×5 cm after downsample). Recon voxels outside the phantom map to
    not-myocardium.
  * AIF AUC is integrated up to V2 time from the real patient AIF CSV.

Output (in OUT_DIR):
  perfusion_map_basissim_subject002_rest.npy   Float32 (512×512×250)
  flow_map_basissim_subject002_rest.npy
  summary_basissim_subject002_rest.txt

Usage:
  python3 step5_perfusion_from_recon.py  AIF_CSV  V1_DIR  V2_DIR  [--v2_t 21]
                                         [--recon fbp|hir]
                                         [--out_suffix _subject002_rest]
"""

import argparse
import os
import sys
import numpy as np
try:
    import tomllib
    def _load_toml(p):
        with open(p, "rb") as f: return tomllib.load(f)
except ImportError:
    import tomli
    def _load_toml(p):
        with open(p, "rb") as f: return tomli.load(f)

# ── Phantom geometry (original, before BasisSim transforms) ──────────────────
PHANTOM_LABELS = "/home/molloi-lab/smb_mount/shared_drive/Shu Nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"
NX, NY, NZ = 1600, 1400, 500
PHANTOM_VOXEL_MM = 0.2
DOWNSAMPLE = 2  # matches run_cta_sim_param.jl DOWNSAMPLE_FACTOR

# BasisSim-internal phantom after reverse+downsample: 800×700×250 at 0.2 mm
BS_NX, BS_NY, BS_NZ = NX // DOWNSAMPLE, NY // DOWNSAMPLE, NZ // DOWNSAMPLE
BS_VOXEL_MM = PHANTOM_VOXEL_MM  # voxel size is preserved across downsample in this pipeline
BS_PHYS_X_MM = BS_NX * BS_VOXEL_MM
BS_PHYS_Y_MM = BS_NY * BS_VOXEL_MM
BS_PHYS_Z_MM = BS_NZ * BS_VOXEL_MM

TISSUE_RHO = 1.053  # g/cm³ for myocardium
HU_PER_MG_ML_IODINE_100KVP = 35.0


def load_recon_f32(recon_dir, kind="fbp"):
    """Load BasisSim recon HU volume + metadata."""
    meta = _load_toml(os.path.join(recon_dir, "recon_meta.toml"))
    fn = f"recon_{kind}_hu_f32.raw"
    path = os.path.join(recon_dir, fn)
    nx, ny, nz = [int(v) for v in meta["recon"]["shape"]]
    arr = np.fromfile(path, dtype=np.float32).reshape((nx, ny, nz), order="F")
    return arr, meta["recon"]


def build_myo_mask_at_recon_resolution(recon_shape, recon_voxel_mm, recon_z_cm):
    """
    Resample myocardium mask from original phantom space to recon space.

    Original phantom: (1600, 1400, 500) at 0.2 mm voxel, labels at 15..18 = myo.
    BasisSim transforms: reverse(dims=(2,3)) → downsample by 2.
    Recon: matches recon_shape at recon_voxel_mm, centered at origin (per
    BS.Phantom default; phantom extent 16×14×5 cm sits inside recon 35×35×5 cm).
    """
    nx_r, ny_r, nz_r = recon_shape
    rdx, rdy, rdz = recon_voxel_mm   # mm

    print(f"[step5] loading phantom labels ({NX}×{NY}×{NZ}) …")
    labels = np.fromfile(PHANTOM_LABELS, dtype=np.uint8).reshape((NX, NY, NZ), order="F")
    myo_orig = ((labels >= 15) & (labels <= 18)).astype(np.uint8)
    n_myo_orig = int(myo_orig.sum())
    print(f"[step5]   original myo voxels: {n_myo_orig}")

    # Apply BasisSim transforms in F-order array space
    myo_rev = myo_orig[:, ::-1, ::-1]
    # downsample_labeled in run_cta_sim takes phantom[(i-1)*factor + h] with h=2,
    # i.e. 0-indexed in Python: phantom[2*i + 1]. For factor=2 we keep odd indices.
    myo_ds = myo_rev[1::DOWNSAMPLE, 1::DOWNSAMPLE, 1::DOWNSAMPLE]
    assert myo_ds.shape == (BS_NX, BS_NY, BS_NZ), \
        f"downsample shape {myo_ds.shape} != expected {(BS_NX, BS_NY, BS_NZ)}"
    n_myo_ds = int(myo_ds.sum())
    print(f"[step5]   BS-phantom myo voxels (after reverse+downsample): {n_myo_ds}")

    # Build recon-grid coords in BS-phantom voxel space.
    # Both grids centered at origin (BS.Phantom default).
    ph_cx, ph_cy, ph_cz = BS_NX / 2 - 0.5, BS_NY / 2 - 0.5, BS_NZ / 2 - 0.5
    r_cx, r_cy, r_cz = nx_r / 2 - 0.5, ny_r / 2 - 0.5, nz_r / 2 - 0.5
    s_x = rdx / BS_VOXEL_MM
    s_y = rdy / BS_VOXEL_MM
    s_z = rdz / BS_VOXEL_MM
    print(f"[step5]   recon voxel scale (recon/BS): x={s_x:.4f} y={s_y:.4f} z={s_z:.4f}")

    # Vectorized resample: for each recon voxel, compute corresponding BS voxel,
    # then nearest-neighbor lookup. Build with 1D coords (memory-efficient).
    # ph_i = (i_r - r_cx) * s_x + ph_cx  → 1D shape (nx_r,)
    ph_i_1d = (np.arange(nx_r) - r_cx) * s_x + ph_cx
    ph_j_1d = (np.arange(ny_r) - r_cy) * s_y + ph_cy
    ph_k_1d = (np.arange(nz_r) - r_cz) * s_z + ph_cz
    ii = np.rint(ph_i_1d).astype(np.int32)
    jj = np.rint(ph_j_1d).astype(np.int32)
    kk = np.rint(ph_k_1d).astype(np.int32)
    # Bound flags per axis
    i_ok = (ii >= 0) & (ii < BS_NX)
    j_ok = (jj >= 0) & (jj < BS_NY)
    k_ok = (kk >= 0) & (kk < BS_NZ)
    # Clamp to safe lookup; will mask later
    ii_c = np.clip(ii, 0, BS_NX - 1)
    jj_c = np.clip(jj, 0, BS_NY - 1)
    kk_c = np.clip(kk, 0, BS_NZ - 1)

    # Lookup: myo_ds[ii_c[:, None, None], jj_c[None, :, None], kk_c[None, None, :]]
    print("[step5]   sampling myo mask onto recon grid …")
    myo_recon = myo_ds[ii_c[:, None, None], jj_c[None, :, None], kk_c[None, None, :]]
    valid = i_ok[:, None, None] & j_ok[None, :, None] & k_ok[None, None, :]
    myo_recon = myo_recon & valid.astype(np.uint8)
    print(f"[step5]   recon myo voxels: {int(myo_recon.sum())}")
    return myo_recon.astype(bool)


def aif_auc_hu_s(aif_csv, t_max_s, hu_per_mg_ml):
    """Load CSV (time_s, C_mg_per_mL) and integrate up to t_max_s."""
    ts, cs = [], []
    with open(aif_csv) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("time_s"):
                continue
            parts = s.split(",")
            if len(parts) < 2: continue
            ts.append(float(parts[0])); cs.append(float(parts[1]))
    t = np.asarray(ts, dtype=np.float64)
    c = np.asarray(cs, dtype=np.float64)
    order = np.argsort(t)
    t = t[order]; c = c[order]
    trapz_fn = getattr(np, "trapezoid", np.trapz)
    t_hi = min(float(t[-1]), float(t_max_s))
    t_fine = np.linspace(0.0, t_hi, max(int(t_hi / 0.01), 2))
    c_fine = np.interp(t_fine, t, c, left=0.0, right=c[-1])
    auc_mg = float(trapz_fn(c_fine, t_fine))
    return auc_mg * hu_per_mg_ml, auc_mg, t, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aif_csv")
    ap.add_argument("v1_dir", help="BasisSim out dir for V1 baseline")
    ap.add_argument("v2_dir", help="BasisSim out dir for V2 peak")
    ap.add_argument("--v2_t", type=float, default=21.0)
    ap.add_argument("--recon", choices=["fbp", "hir"], default="fbp")
    ap.add_argument("--out_suffix", default="_subject002_rest_basissim")
    ap.add_argument("--out_dir",
                    default="/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline/output")
    args = ap.parse_args()

    print(f"[step5] AIF CSV  = {args.aif_csv}")
    print(f"[step5] V1 dir   = {args.v1_dir}")
    print(f"[step5] V2 dir   = {args.v2_dir}")
    print(f"[step5] recon    = {args.recon}")
    print(f"[step5] V2 time  = {args.v2_t}s")
    print(f"[step5] HU/mg/mL = {HU_PER_MG_ML_IODINE_100KVP} (100 kVp)")

    # Load V1 and V2 recon (Float32 HU)
    V1, meta = load_recon_f32(args.v1_dir, args.recon)
    V2, _    = load_recon_f32(args.v2_dir, args.recon)
    print(f"[step5] V1 shape={V1.shape}  HU range [{V1.min():.1f}, {V1.max():.1f}]")
    print(f"[step5] V2 shape={V2.shape}  HU range [{V2.min():.1f}, {V2.max():.1f}]")
    nx_r, ny_r, nz_r = V1.shape
    voxel_size_mm = tuple(float(v) for v in meta["voxel_size_mm"])
    recon_z_cm = float(meta["z_cm"])
    print(f"[step5] recon voxel size = {voxel_size_mm} mm")

    # Build myocardium mask at recon resolution
    myo_mask = build_myo_mask_at_recon_resolution(
        V1.shape, voxel_size_mm, recon_z_cm)
    n_myo = int(myo_mask.sum())
    voxel_vol_cm3 = (voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2]) / 1000.0
    organ_mass_g = n_myo * TISSUE_RHO * voxel_vol_cm3
    print(f"[step5] myo voxels at recon res: {n_myo}  ({organ_mass_g:.1f} g, ρ=1.053)")

    if n_myo < 1000:
        print("ERROR: myocardium mask is too small at recon resolution; check coord transform")
        sys.exit(1)

    # AIF AUC up to V2 time
    auc_hu_s, auc_mg, t_aif, c_aif = aif_auc_hu_s(args.aif_csv, args.v2_t, HU_PER_MG_ML_IODINE_100KVP)
    c_at_v2 = float(np.interp(args.v2_t, t_aif, c_aif, left=0.0, right=c_aif[-1]))
    print(f"[step5] AIF AUC(0..{args.v2_t}s) = {auc_mg:.3f} mg·s/mL = {auc_hu_s:.2f} HU·s")
    print(f"[step5] AIF C(t={args.v2_t}s) = {c_at_v2:.4f} mg/mL ≈ {c_at_v2 * HU_PER_MG_ML_IODINE_100KVP:.1f} HU enhancement")

    # Mullani-Gould single-volume perfusion (same math as step4)
    V1_f = V1.astype(np.float64)
    V2_f = V2.astype(np.float64)
    v1_mean_hu = float(V1_f[myo_mask].mean())
    v2_mean_hu = float(V2_f[myo_mask].mean())
    delta_hu = v2_mean_hu - v1_mean_hu
    v1_mass = (V1_f * myo_mask).sum() * voxel_vol_cm3
    v2_mass = (V2_f * myo_mask).sum() * voxel_vol_cm3
    flow_total = (60.0 / auc_hu_s) * (v2_mass - v1_mass)  # mL/min total

    # Voxel-wise flow ∝ local ΔHU
    diff = V2_f - V1_f
    flow_map = np.zeros_like(diff, dtype=np.float32)
    if abs(delta_hu) > 1e-6:
        flow_map[myo_mask] = (diff[myo_mask] / delta_hu * flow_total).astype(np.float32)
    perf_map = (flow_map / organ_mass_g).astype(np.float32)  # mL/min/g
    perf_inside = perf_map[myo_mask]
    perf_mean = float(perf_inside.mean())
    perf_std  = float(perf_inside.std())

    print("\n[step5] === BASISSIM PERFUSION RESULT ===")
    print(f"  V1 mean HU (myo): {v1_mean_hu:.2f}")
    print(f"  V2 mean HU (myo): {v2_mean_hu:.2f}")
    print(f"  ΔHU mean:          {delta_hu:.2f}")
    print(f"  organ mass:        {organ_mass_g:.2f} g")
    print(f"  flow (total):      {flow_total:.2f} mL/min")
    print(f"  perfusion (mean):  {perf_mean:.4f} mL/min/g")
    print(f"  perfusion (std):   {perf_std:.4f}  (per-voxel spread)")
    print(f"  perfusion p5..p95: {np.percentile(perf_inside, 5):.4f} .. "
          f"{np.percentile(perf_inside, 95):.4f} mL/min/g")
    print(f"  perfusion median:  {np.median(perf_inside):.4f}")

    # Clinical sanity
    rest_lo, rest_hi = 0.5, 1.5
    if rest_lo <= perf_mean <= rest_hi:
        print(f"  ★ mean {perf_mean:.3f} within REST range [{rest_lo}, {rest_hi}] ✓")
    else:
        print(f"  ⚠ mean {perf_mean:.3f} OUTSIDE REST range [{rest_lo}, {rest_hi}]")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    suf = args.out_suffix
    p_path = os.path.join(args.out_dir, f"perfusion_map{suf}.npy")
    f_path = os.path.join(args.out_dir, f"flow_map{suf}.npy")
    m_path = os.path.join(args.out_dir, f"myo_mask{suf}.npy")
    s_path = os.path.join(args.out_dir, f"summary{suf}.txt")
    np.save(p_path, perf_map)
    np.save(f_path, flow_map)
    np.save(m_path, myo_mask.astype(np.uint8))
    with open(s_path, "w") as f:
        f.write(f"# Path B (BasisSim) perfusion summary\n")
        f.write(f"aif_csv     = {args.aif_csv}\n")
        f.write(f"v1_dir      = {args.v1_dir}\n")
        f.write(f"v2_dir      = {args.v2_dir}\n")
        f.write(f"recon_kind  = {args.recon}\n")
        f.write(f"v2_t        = {args.v2_t}\n")
        f.write(f"AIF_AUC_HUs = {auc_hu_s:.4f}\n")
        f.write(f"V1_mean_HU  = {v1_mean_hu:.4f}\n")
        f.write(f"V2_mean_HU  = {v2_mean_hu:.4f}\n")
        f.write(f"delta_HU    = {delta_hu:.4f}\n")
        f.write(f"organ_mass_g = {organ_mass_g:.4f}\n")
        f.write(f"flow_mlmin  = {flow_total:.4f}\n")
        f.write(f"perfusion_mean = {perf_mean:.6f}\n")
        f.write(f"perfusion_std  = {perf_std:.6f}\n")
        f.write(f"perfusion_p5   = {np.percentile(perf_inside, 5):.6f}\n")
        f.write(f"perfusion_p95  = {np.percentile(perf_inside, 95):.6f}\n")
        f.write(f"perfusion_median = {np.median(perf_inside):.6f}\n")
        f.write(f"n_myo_recon_voxels = {n_myo}\n")
        f.write(f"recon_shape = {V1.shape}\n")
        f.write(f"recon_voxel_size_mm = {voxel_size_mm}\n")
    print(f"\n[step5] saved:")
    print(f"  {p_path}")
    print(f"  {f_path}")
    print(f"  {m_path}")
    print(f"  {s_path}")


if __name__ == "__main__":
    main()
