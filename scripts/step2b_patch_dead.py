#!/usr/bin/env python
"""
Step 2b: patch myocardium voxels whose nearest terminal has zero flow
(arrival = Inf) by reassigning them the arrival time of the *nearest
voxel with a live terminal*. This is a post-hoc spatial interpolation
that respects the underlying Pries+Poiseuille hemodynamics — we don't
modify any flow values, only redirect dead-territory voxels to a
neighboring live capillary territory (which is what blood would do in
reality once a particular path is starved).

Without this fix, ~70 % of myocardium ends up unenhanced because the
6 µm capillary end-points pick up disproportionate viscous resistance
under no-calibration Pries 1992, leaving many leaves with zero flow.

Reads:
    intermediate/myo_arrival.raw  (Float32, Inf where dead)
    intermediate/myo_tree_id.raw  (UInt8)
    intermediate/myo_dist.raw     (Float32)
    vmale50_act_1.raw             (UInt8)

Writes (patched, *_patched.raw):
    intermediate/myo_arrival_patched.raw  (Float32, no Inf in myo)
    intermediate/myo_tree_id_patched.raw  (UInt8, inherits from neighbor)
    intermediate/myo_dist_patched.raw
"""

import os, numpy as np
from scipy.spatial import cKDTree

PIPELINE_DIR = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline"
INTER_DIR = os.path.join(PIPELINE_DIR, "intermediate")
PHANTOM_LABELS_PATH = "/home/molloi-lab/smb_mount/shared_drive/Shu Nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"
NX, NY, NZ = 1600, 1400, 500

print("[step2b] Loading arrays …")
arr = np.fromfile(os.path.join(INTER_DIR, "myo_arrival.raw"),
                  dtype=np.float32).reshape((NX, NY, NZ), order="F")
tid = np.fromfile(os.path.join(INTER_DIR, "myo_tree_id.raw"),
                  dtype=np.uint8).reshape((NX, NY, NZ), order="F")
dist = np.fromfile(os.path.join(INTER_DIR, "myo_dist.raw"),
                   dtype=np.float32).reshape((NX, NY, NZ), order="F")
labels = np.fromfile(PHANTOM_LABELS_PATH,
                    dtype=np.uint8).reshape((NX, NY, NZ), order="F")
myo = (labels >= 15) & (labels <= 18)
n_myo = int(myo.sum())
print(f"[step2b]   myo voxels = {n_myo}")

live_mask = myo & np.isfinite(arr)
dead_mask = myo & np.isinf(arr)
n_live = int(live_mask.sum())
n_dead = int(dead_mask.sum())
print(f"[step2b]   live voxels (finite arrival): {n_live} ({100*n_live/n_myo:.1f}%)")
print(f"[step2b]   dead voxels (Inf arrival):    {n_dead} ({100*n_dead/n_myo:.1f}%)")
if n_dead == 0:
    print("[step2b] No dead voxels — nothing to patch."); raise SystemExit(0)

# Voxel coordinates (in voxel indices; KDTree on integer coords is fine)
print("[step2b] Building KDTree of live voxel coordinates …")
live_idx = np.where(live_mask)
live_coords = np.stack(live_idx, axis=1).astype(np.float32)         # (N_live, 3)
live_arrivals = arr[live_idx]
live_tids = tid[live_idx]
kd = cKDTree(live_coords)

dead_idx = np.where(dead_mask)
dead_coords = np.stack(dead_idx, axis=1).astype(np.float32)         # (N_dead, 3)

print(f"[step2b] Querying {n_dead} dead voxels for nearest live neighbor …")
dists_idx, nearest = kd.query(dead_coords, k=1, workers=-1)
inherit_arrival = live_arrivals[nearest]
inherit_tid     = live_tids[nearest]

# Stats on inheritance distance
voxel_size_cm = 0.02
dists_cm = dists_idx * voxel_size_cm
print(f"[step2b]   inheritance distance (cm): median={np.median(dists_cm):.3f}  p95={np.percentile(dists_cm, 95):.3f}  max={dists_cm.max():.3f}")

# Apply
arr_patched = arr.copy()
tid_patched = tid.copy()
dist_patched = dist.copy()
arr_patched[dead_idx] = inherit_arrival
tid_patched[dead_idx] = inherit_tid
# Keep original terminal distance (so we can tell patched vs untouched). Or set to nearest_live distance.
# For simplicity, keep dist as-is (tells distance to original nearest terminal — which had no flow).

# Save — IMPORTANT: numpy ndarray.tofile() always writes in C-order regardless
# of array memory layout, but our raw files use Fortran (x-fastest) order to
# match XCAT phantom binary convention. Force F-order via tobytes(order='F').
print("[step2b] Saving patched arrays (F-order) …")
def write_f_order(arr, path):
    with open(path, "wb") as f:
        f.write(arr.tobytes(order="F"))
write_f_order(arr_patched,  os.path.join(INTER_DIR, "myo_arrival_patched.raw"))
write_f_order(tid_patched,  os.path.join(INTER_DIR, "myo_tree_id_patched.raw"))
write_f_order(dist_patched, os.path.join(INTER_DIR, "myo_dist_patched.raw"))

# Verify
arr_p_myo = arr_patched[myo]
n_inf_after = int(np.isinf(arr_p_myo).sum())
finite_after = arr_p_myo[np.isfinite(arr_p_myo)]
print(f"[step2b]   after patch: Inf in myo = {n_inf_after} (should be 0)")
print(f"[step2b]   arrival times (s): median={np.median(finite_after):.2f}  p95={np.percentile(finite_after, 95):.2f}  max={finite_after.max():.2f}")
print("\n[step2b] Done.")
