#!/usr/bin/env python3
"""
Extract coronary tree segments with diameter ≥ MIN_DIAM_UM from each tree CSV
and project endpoints into BasisSim recon mm coords (the same frame as the
perfusion viewer).

Output:
    tree_overlay.npz  with arrays
        x  (2N+gaps,)  float32   x_mm of segment start/end with NaN separators
        y  (...)
        z  (...)
        diameter_um   (N,)       per-segment (for optional width scaling)
        tree_id       (N,)       0=LAD, 1=LCX, 2=RCA

Coord transform chain (NRB tree cm → viewer recon mm):
    1. NRB cm     + NRB_OFFSET                       → phantom world cm
    2. /0.02 cm/voxel                                → phantom voxel idx
    3. reverse(y), reverse(z)                        → BS phantom orientation
    4. (idx-1)/2                                     → BS phantom downsampled idx
    5. centered scale + offset to recon (512^2x250) → recon voxel idx
    6. * recon voxel mm (0.6836,0.6836,0.2)         → mm in viewer coords
"""
import argparse
import os
import sys
import numpy as np

NRB_OFFSET = np.array([2.1443, -9.5553, -20.0068], dtype=np.float64)
PH_DIMS = (1600, 1400, 500)
PH_VOXEL_CM = 0.02
DOWNSAMPLE = 2
BS_DIMS = (800, 700, 250)
BS_VOXEL_MM = 0.2
RECON_DIMS = (512, 512, 250)
RECON_VOXEL_MM = np.array([0.6836, 0.6836, 0.2], dtype=np.float64)

BS_CENTER = np.array([(BS_DIMS[0]-1)/2, (BS_DIMS[1]-1)/2, (BS_DIMS[2]-1)/2])
R_CENTER = np.array([(RECON_DIMS[0]-1)/2, (RECON_DIMS[1]-1)/2, (RECON_DIMS[2]-1)/2])
SCALE_BS_TO_R = BS_VOXEL_MM / RECON_VOXEL_MM   # element-wise


def nrb_to_recon_mm(pts_cm):
    """pts_cm: (N, 3) NRB coordinates in cm.
    Returns: (N, 3) mm in viewer coords (matches scatter myo + chamber mesh)."""
    p = pts_cm + NRB_OFFSET  # NRB → phantom world cm
    # phantom voxel idx
    p = p / PH_VOXEL_CM
    # reverse y, z (BasisSim convention)
    p[:, 1] = (PH_DIMS[1] - 1) - p[:, 1]
    p[:, 2] = (PH_DIMS[2] - 1) - p[:, 2]
    # downsample: bs_idx = (orig - 1) / 2  (every-other-starting-at-1)
    p = (p - 1.0) / DOWNSAMPLE
    # BS → recon (both centered at iso)
    p = (p - BS_CENTER) * SCALE_BS_TO_R + R_CENTER
    # recon idx → mm in viewer coords
    p = p * RECON_VOXEL_MM
    return p.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree_dir", help="dir with lad_segments.csv etc.")
    ap.add_argument("out_npz", help="output .npz")
    ap.add_argument("--min_diam_um", type=float, default=500.0)
    args = ap.parse_args()

    csvs = []
    for name in ("lad", "lcx", "rca"):
        p = os.path.join(args.tree_dir, f"{name}_segments.csv")
        if os.path.isfile(p):
            csvs.append((name, p))
    if not csvs:
        print(f"ERROR: no *_segments.csv in {args.tree_dir}", file=sys.stderr)
        sys.exit(1)

    all_pts = []  # list of (start_cm, end_cm) for each kept segment
    all_diams = []
    all_tree_id = []
    tree_id_map = {"lad": 0, "lcx": 1, "rca": 2}

    for name, csv_path in csvs:
        tid = tree_id_map[name]
        print(f"[tree] streaming {csv_path} (≥{args.min_diam_um} μm) …")
        n_kept = 0
        n_total = 0
        # Read CSV manually for speed (file is multi-GB; only need 6 endpoint cols + diam)
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
            i_x1 = header.index("x1_cm");  i_y1 = header.index("y1_cm");  i_z1 = header.index("z1_cm")
            i_x2 = header.index("x2_cm");  i_y2 = header.index("y2_cm");  i_z2 = header.index("z2_cm")
            i_d  = header.index("diameter_um")
            for line in f:
                if not line.strip(): continue
                n_total += 1
                cols = line.split(",")
                try:
                    d = float(cols[i_d])
                except (ValueError, IndexError):
                    continue
                if d < args.min_diam_um:
                    continue
                try:
                    x1 = float(cols[i_x1]); y1 = float(cols[i_y1]); z1 = float(cols[i_z1])
                    x2 = float(cols[i_x2]); y2 = float(cols[i_y2]); z2 = float(cols[i_z2])
                except (ValueError, IndexError):
                    continue
                all_pts.append([x1, y1, z1, x2, y2, z2])
                all_diams.append(d)
                all_tree_id.append(tid)
                n_kept += 1
        print(f"[tree]   {name}: kept {n_kept} / {n_total} segs")

    if not all_pts:
        print("ERROR: no segments matched the diameter cutoff", file=sys.stderr)
        sys.exit(1)

    pts = np.asarray(all_pts, dtype=np.float64)  # (N, 6) start/end
    diams = np.asarray(all_diams, dtype=np.float32)
    tids  = np.asarray(all_tree_id, dtype=np.uint8)
    N = pts.shape[0]
    print(f"[tree] total kept: {N} segments  diameter [{diams.min():.1f}, {diams.max():.1f}] μm")

    # Transform start + end points
    starts_mm = nrb_to_recon_mm(pts[:, :3])
    ends_mm   = nrb_to_recon_mm(pts[:, 3:6])

    # Build Plotly-style flat arrays: [x1, x2, nan, x1', x2', nan, ...]
    # 3*N entries total.
    flat_x = np.empty(3 * N, dtype=np.float32)
    flat_y = np.empty(3 * N, dtype=np.float32)
    flat_z = np.empty(3 * N, dtype=np.float32)
    flat_x[0::3] = starts_mm[:, 0];  flat_x[1::3] = ends_mm[:, 0];  flat_x[2::3] = np.nan
    flat_y[0::3] = starts_mm[:, 1];  flat_y[1::3] = ends_mm[:, 1];  flat_y[2::3] = np.nan
    flat_z[0::3] = starts_mm[:, 2];  flat_z[1::3] = ends_mm[:, 2];  flat_z[2::3] = np.nan

    np.savez_compressed(args.out_npz,
                        # Flat NaN-separated arrays (kept for the legacy lines viewer)
                        x=flat_x, y=flat_y, z=flat_z,
                        # Per-segment start/end in viewer mm (used by cylinder-mesh viewer)
                        starts_mm=starts_mm,
                        ends_mm=ends_mm,
                        diameter_um=diams, tree_id=tids,
                        min_diam_um=np.float32(args.min_diam_um))
    print(f"[tree] wrote {args.out_npz}  ({os.path.getsize(args.out_npz)/1024**2:.2f} MB)")
    print(f"  x range: {np.nanmin(flat_x):.2f} .. {np.nanmax(flat_x):.2f} mm")
    print(f"  y range: {np.nanmin(flat_y):.2f} .. {np.nanmax(flat_y):.2f} mm")
    print(f"  z range: {np.nanmin(flat_z):.2f} .. {np.nanmax(flat_z):.2f} mm")


if __name__ == "__main__":
    main()
