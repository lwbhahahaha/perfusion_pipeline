#!/usr/bin/env python3
"""
Build an opaque "inner-blood-pool" surface mesh in BasisSim recon coordinates,
so the perfusion 3D viewer can occlude points sitting behind the heart chambers
when viewed from outside.

Input  : XCAT phantom labels (1600x1400x500 UInt8)
         + aorta lumen mask (UInt8, same shape)
Output : chamber_mesh_recon.npz with
           verts   (N, 3)  float32  mm  in recon coord frame
           faces   (M, 3)  int32     triangle indices

Pipeline (matches step5 / run_cta_sim_param.jl conventions):
  1. labels[chamber + aorta]  →  bool mask in phantom space
  2. reverse(dims=2,3) + downsample by 2     →  BS phantom space (800,700,250)
  3. resample to recon grid (512,512,250 @ FOV 35 cm × z 5 cm centered)
  4. marching_cubes(level=0.5)  →  verts, faces
  5. verts (i,j,k) → mm using recon voxel size (0.6836, 0.6836, 0.2)

Usage:
    python3 build_chamber_mesh.py  [--include-aorta] [--include-coronary-trunk]
"""
import argparse
import os
import numpy as np
from skimage.measure import marching_cubes

PHANTOM_LABELS = "/home/molloi-lab/smb_mount/shared_drive/Shu Nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"
AORTA_MASK = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline/intermediate/aorta_lumen_mask.raw"
OUT_PATH = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline/intermediate/chamber_mesh_recon.npz"

NX, NY, NZ = 1600, 1400, 500
DOWNSAMPLE = 2
BS_NX, BS_NY, BS_NZ = NX // DOWNSAMPLE, NY // DOWNSAMPLE, NZ // DOWNSAMPLE
BS_VOXEL_MM = 0.2  # = source voxel size, preserved across downsample (every-other)

RECON_SHAPE = (512, 512, 250)
RECON_FOV_CM = 35.0
RECON_Z_CM = 5.0
RECON_VOXEL_MM = (RECON_FOV_CM * 10.0 / RECON_SHAPE[0],
                  RECON_FOV_CM * 10.0 / RECON_SHAPE[1],
                  RECON_Z_CM   * 10.0 / RECON_SHAPE[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-aorta", action="store_true",
                    help="also include aorta lumen mask")
    ap.add_argument("--include-coronary-trunk", action="store_true",
                    help="also include XCAT label 26 (coronary trunk)")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    print("[mesh] loading phantom labels …")
    labels = np.fromfile(PHANTOM_LABELS, dtype=np.uint8).reshape((NX, NY, NZ), order="F")
    print(f"[mesh]   shape={labels.shape}  dtype={labels.dtype}")

    # Build blood pool mask: chamber blood + (optional aorta) + (optional coronary trunk)
    print("[mesh] assembling blood-pool mask:  LV(19) + RV(20) + LA(21) + RA(22) …")
    chamber_mask = (labels == 19) | (labels == 20) | (labels == 21) | (labels == 22)
    print(f"[mesh]   chambers (19,20,21,22): {int(chamber_mask.sum())} voxels")

    if args.include_aorta:
        if not os.path.isfile(AORTA_MASK):
            print(f"[mesh] WARN: --include-aorta requested but mask not found at {AORTA_MASK}")
        else:
            print("[mesh] adding aorta lumen mask …")
            aorta = np.fromfile(AORTA_MASK, dtype=np.uint8).reshape((NX, NY, NZ), order="F")
            chamber_mask |= (aorta > 0)
            print(f"[mesh]   chambers + aorta: {int(chamber_mask.sum())} voxels")

    if args.include_coronary_trunk:
        print("[mesh] adding coronary trunk (label 26) …")
        chamber_mask |= (labels == 26)
        print(f"[mesh]   + trunk: {int(chamber_mask.sum())} voxels")

    del labels

    # ── BasisSim phantom transforms: reverse y,z then downsample ──
    print("[mesh] applying BasisSim phantom transforms (reverse y,z + ds by 2) …")
    rev = chamber_mask[:, ::-1, ::-1]
    bs_phantom = rev[1::DOWNSAMPLE, 1::DOWNSAMPLE, 1::DOWNSAMPLE]
    del rev, chamber_mask
    print(f"[mesh]   BS-phantom shape: {bs_phantom.shape}  nnz={int(bs_phantom.sum())}")

    # ── Resample to recon grid (matches step5 build_myo_mask_at_recon_resolution) ──
    nx_r, ny_r, nz_r = RECON_SHAPE
    rdx, rdy, rdz = RECON_VOXEL_MM
    ph_cx = BS_NX / 2 - 0.5; ph_cy = BS_NY / 2 - 0.5; ph_cz = BS_NZ / 2 - 0.5
    r_cx  = nx_r / 2 - 0.5;   r_cy  = ny_r / 2 - 0.5;   r_cz  = nz_r / 2 - 0.5
    s_x = rdx / BS_VOXEL_MM
    s_y = rdy / BS_VOXEL_MM
    s_z = rdz / BS_VOXEL_MM

    ph_i_1d = (np.arange(nx_r) - r_cx) * s_x + ph_cx
    ph_j_1d = (np.arange(ny_r) - r_cy) * s_y + ph_cy
    ph_k_1d = (np.arange(nz_r) - r_cz) * s_z + ph_cz
    ii = np.clip(np.rint(ph_i_1d).astype(np.int32), 0, BS_NX - 1)
    jj = np.clip(np.rint(ph_j_1d).astype(np.int32), 0, BS_NY - 1)
    kk = np.clip(np.rint(ph_k_1d).astype(np.int32), 0, BS_NZ - 1)
    i_ok = (ph_i_1d >= 0) & (ph_i_1d < BS_NX)
    j_ok = (ph_j_1d >= 0) & (ph_j_1d < BS_NY)
    k_ok = (ph_k_1d >= 0) & (ph_k_1d < BS_NZ)

    print(f"[mesh] resampling to recon grid {RECON_SHAPE} @ voxel {RECON_VOXEL_MM} mm …")
    chamber_recon = bs_phantom[ii[:, None, None], jj[None, :, None], kk[None, None, :]]
    valid = i_ok[:, None, None] & j_ok[None, :, None] & k_ok[None, None, :]
    chamber_recon = chamber_recon & valid
    print(f"[mesh]   recon-grid chamber voxels: {int(chamber_recon.sum())}")
    del bs_phantom

    # Pre-process the mask to give marching_cubes a clean, watertight input:
    #   1. binary_closing fills small interior holes / step-edge dips
    #   2. gaussian_filter smooths the level surface so triangles aren't tiny
    # CRITICAL: DO NOT decimate naively (stride-based decimation removes triangles
    # at random and punches holes through the closed surface, defeating opacity).
    from scipy.ndimage import binary_closing, gaussian_filter
    print("[mesh] binary_closing(radius=2) to fill pinholes …")
    closed = binary_closing(chamber_recon, iterations=2)
    chamber_smooth = gaussian_filter(closed.astype(np.float32), sigma=1.0)

    # ── Marching cubes at level=0.5  (watertight closed mesh) ──
    print("[mesh] running marching cubes …")
    verts, faces, normals, _ = marching_cubes(chamber_smooth, level=0.5)
    print(f"[mesh]   verts: {verts.shape}  faces: {faces.shape}  (no decimation — keep watertight)")

    # verts indices (i,j,k) → mm in recon coords
    verts_mm = verts.astype(np.float32) * np.array(RECON_VOXEL_MM, dtype=np.float32)

    np.savez_compressed(args.out,
                        verts_mm=verts_mm,
                        faces=faces.astype(np.int32),
                        recon_shape=np.array(RECON_SHAPE, dtype=np.int32),
                        voxel_mm=np.array(RECON_VOXEL_MM, dtype=np.float32))
    print(f"[mesh] wrote {args.out} ({os.path.getsize(args.out)/1024**2:.2f} MB)")


if __name__ == "__main__":
    main()
