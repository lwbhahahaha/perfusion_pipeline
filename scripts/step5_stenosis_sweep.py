#!/usr/bin/env python
"""
Step 5: LAD-stenosis perfusion sweep.

Reuses the baseline V1 (frame_00) and V2 (frame_10) from step 3 and applies
post-hoc per-territory ΔHU scaling by Q_stenosed / Q_baseline (taken from
FlowContrastSim's lad_stenosis_sweep.jl output). For stenosis at a given
percentage, only LAD-owned myocardium voxels (territory map from step 2)
have their ΔHU reduced — LCX/RCA territories are unchanged. The pipeline
then runs the same single-volume Mullani-Gould perfusion formula and
records per-territory perfusion stats.

Output:
    output/stenosis_sweep.csv       (stenosis%, Q_LAD_ratio, perf_LAD, perf_LCX, perf_RCA)
    output/rsna_figs/fig_D_gould_perfusion.png  — Gould curve in perfusion-domain
    output/qa/perf_overlay_st{pct:02d}.png — per-stenosis QA mid-slice
"""
import os, sys, numpy as np, toml, csv
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

PIPELINE_DIR = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline"
INTER_DIR  = os.path.join(PIPELINE_DIR, "intermediate")
FRAMES_DIR = os.path.join(PIPELINE_DIR, "frames")
OUT_DIR    = os.path.join(PIPELINE_DIR, "output")
RSNA_DIR   = os.path.join(OUT_DIR, "rsna_figs"); os.makedirs(RSNA_DIR, exist_ok=True)
QA_DIR     = os.path.join(OUT_DIR, "qa"); os.makedirs(QA_DIR, exist_ok=True)
PHANTOM_LABELS_PATH = "/home/molloi-lab/smb_mount/shared_drive/Shu Nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"
LAD_STENOSIS_CSV = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/FlowContrastSim.jl/scripts/lad_stenosis.csv"

NX, NY, NZ = 1600, 1400, 500
VOXEL_MM = 0.2
TISSUE_RHO = 1.053
V2_T = 10  # peak frame index

meta = toml.load(os.path.join(INTER_DIR, "metadata.toml"))
AIF = meta["aif_clinical"]
HU_PER_MG_ML = float(meta["hu_per_mg_ml_iodine"])

def gamma_aif(t):
    if t <= AIF["t0_s"]: return 0.0
    tp = (t - AIF["t0_s"]) / (AIF["tmax_s"] - AIF["t0_s"])
    return 0.0 if tp <= 0 else AIF["amplitude_mg_ml"] * tp**AIF["alpha"] * np.exp(AIF["alpha"] * (1.0 - tp))

# AIF AUC up to V2
t_fine = np.linspace(0, V2_T, 5001)
auc_aif_hu_s = HU_PER_MG_ML * float(trapezoid([gamma_aif(t) for t in t_fine], t_fine))
print(f"[step5] AIF AUC(0..{V2_T}s) = {auc_aif_hu_s:.1f} HU·s")

# ── Load LAD stenosis Q-ratio table from FCS output ──
# CSV columns: stenosis_pct, proximal_diameter_um, root_flow_mlmin
# We want Q_ratio = Q_at_stenosis / Q_baseline
qrows = []
with open(LAD_STENOSIS_CSV) as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        qrows.append((float(r["stenosis_pct"]), float(r["root_flow_mlmin"])))
qrows.sort(key=lambda r: r[0])
q_baseline = next(q for st, q in qrows if st == 0.0)
Q_RATIO = {st: q / q_baseline for st, q in qrows}
print(f"[step5] LAD Q ratios:")
for st, q in qrows:
    print(f"  {st:5.0f}% st → Q={q:7.1f} mL/min  ratio={q/q_baseline:.3f}")

# ── Load baseline V1, V2, masks ──
def load_int16(path):
    return np.fromfile(path, dtype=np.int16).reshape((NX, NY, NZ), order="F")
def load_uint8(path):
    return np.fromfile(path, dtype=np.uint8).reshape((NX, NY, NZ), order="F")
def load_float32(path):
    return np.fromfile(path, dtype=np.float32).reshape((NX, NY, NZ), order="F")

print("[step5] Loading V1 (frame_00), V2 (frame_10), labels, tree_id …")
V1 = load_int16(os.path.join(FRAMES_DIR, "frame_00.raw")).astype(np.float32)
V2 = load_int16(os.path.join(FRAMES_DIR, f"frame_{V2_T:02d}.raw")).astype(np.float32)
labels = load_uint8(PHANTOM_LABELS_PATH)
myo_mask = (labels >= 15) & (labels <= 18)
tid = load_uint8(os.path.join(INTER_DIR, "myo_tree_id_patched.raw"))
del labels

n_myo = int(myo_mask.sum())
mass_g = n_myo * TISSUE_RHO * (VOXEL_MM**3) / 1000.0
print(f"[step5]   myo voxels = {n_myo}, mass = {mass_g:.1f} g")
v1_mean_hu = float(V1[myo_mask].mean())
print(f"[step5]   V1 mean HU = {v1_mean_hu:.2f}")

# Per-territory mass (from step 2 stats)
masks_per_tree = {1: (tid == 1) & myo_mask,
                  2: (tid == 2) & myo_mask,
                  3: (tid == 3) & myo_mask}
mass_per_tree = {tid_n: int(mask.sum()) * TISSUE_RHO * (VOXEL_MM**3) / 1000.0
                 for tid_n, mask in masks_per_tree.items()}
print(f"[step5]   per-tree mass: LAD={mass_per_tree[1]:.1f} g  LCX={mass_per_tree[2]:.1f} g  RCA={mass_per_tree[3]:.1f} g")

# ── Loop over stenosis levels ──
def perfusion_inline(v2, mask, v1_scalar, auc_hu_s):
    n_mask = int(mask.sum())
    voxel_vol = VOXEL_MM**3 / 1000.0
    organ_mass = n_mask * TISSUE_RHO * voxel_vol
    v1_arr = np.full_like(v2, np.nan, dtype=np.float64)
    v1_arr[mask] = v1_scalar
    v2_nan = np.full_like(v2, np.nan, dtype=np.float64)
    v2_nan[mask] = v2[mask]
    delta_hu = float(np.nanmean(v2_nan[mask]) - v1_scalar)
    v1_mass = float(np.nansum(v1_arr[mask]) * voxel_vol)
    v2_mass = float(np.nansum(v2_nan[mask]) * voxel_vol)
    flow = (60.0 / auc_hu_s) * (v2_mass - v1_mass)
    perf = flow / organ_mass
    flow_map = (v2_nan - v1_arr) / delta_hu * flow if abs(delta_hu) > 1e-9 else np.zeros_like(v2_nan)
    perf_map = flow_map / organ_mass
    return dict(flow=flow, perf=perf, organ_mass=organ_mass, delta_hu=delta_hu,
                perf_map=perf_map.astype(np.float32))

results = []  # list of (stenosis%, Q_ratio, perf_LAD, perf_LCX, perf_RCA, perf_total)
for st_pct, q_ratio in sorted(Q_RATIO.items()):
    if st_pct not in (0.0, 30.0, 50.0, 70.0, 90.0):
        continue   # only headline points
    print(f"\n[step5] === Stenosis {st_pct:.0f}%  Q_LAD_ratio = {q_ratio:.3f} ===")

    # Build stenosed V2 by scaling LAD-territory ΔHU
    V2_st = V2.copy()
    delta_baseline = V2 - V1   # ΔHU at V2 time
    lad_sel = masks_per_tree[1]
    V2_st[lad_sel] = V1[lad_sel] + q_ratio * delta_baseline[lad_sel]

    # Whole-myocardium perfusion
    res_all = perfusion_inline(V2_st, myo_mask, v1_mean_hu, auc_aif_hu_s)
    print(f"  whole myo: ΔHU={res_all['delta_hu']:.2f} HU  flow={res_all['flow']:.1f} mL/min  perf={res_all['perf']:.3f} mL/min/g")

    # Per-territory perfusion (within each tree's mask)
    perf_per = {}
    for tid_n, name in [(1,"LAD"),(2,"LCX"),(3,"RCA")]:
        sel = masks_per_tree[tid_n]
        res = perfusion_inline(V2_st, sel, v1_mean_hu, auc_aif_hu_s)
        perf_per[name] = res['perf']
        print(f"  {name}: mass={res['organ_mass']:.1f} g  ΔHU={res['delta_hu']:.2f}  flow={res['flow']:.1f}  perf={res['perf']:.3f}")
    results.append((st_pct, q_ratio, perf_per["LAD"], perf_per["LCX"], perf_per["RCA"], res_all["perf"]))

    # Save mid-slice QA
    SLICE = 200
    X0,X1,Y0,Y1 = 800,1280,280,820
    perf_full = res_all['perf_map']
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    v2_slice = V2_st[X0:X1, Y0:Y1, SLICE].T
    axes[0].imshow(v2_slice, cmap="gray", vmin=-100, vmax=400)
    axes[0].set_title(f"V2 (t={V2_T}s)  stenosis={st_pct:.0f}%")
    axes[0].axis("off")
    perf_slice = perf_full[X0:X1, Y0:Y1, SLICE].T
    axes[1].imshow(v2_slice, cmap="gray", vmin=-100, vmax=400)
    masked = np.ma.masked_invalid(perf_slice)
    im = axes[1].imshow(masked, cmap="jet", vmin=0, vmax=4, alpha=0.75)
    plt.colorbar(im, ax=axes[1], shrink=0.7, label="mL/min/g")
    axes[1].set_title(f"Perfusion  stenosis={st_pct:.0f}%   whole-myo={res_all['perf']:.2f} mL/min/g")
    axes[1].axis("off")
    plt.tight_layout()
    qa_path = os.path.join(QA_DIR, f"perf_overlay_st{int(st_pct):02d}.png")
    plt.savefig(qa_path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  → {qa_path}")
    del V2_st

# ── Save CSV + Gould-curve fig ──
csv_out = os.path.join(OUT_DIR, "stenosis_sweep.csv")
with open(csv_out, "w") as f:
    w = csv.writer(f)
    w.writerow(["stenosis_pct","Q_LAD_ratio","perf_LAD","perf_LCX","perf_RCA","perf_whole_myo"])
    for r in results:
        w.writerow([f"{r[0]:.0f}", f"{r[1]:.4f}",
                    f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}", f"{r[5]:.4f}"])
print(f"\n[step5] Wrote {csv_out}")

st_pcts = [r[0] for r in results]
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(st_pcts, [r[2] for r in results], marker='o', linewidth=2, color="#1f77ff", label="LAD territory")
ax.plot(st_pcts, [r[3] for r in results], marker='o', linewidth=2, color="#e3342f", label="LCX territory (unchanged)")
ax.plot(st_pcts, [r[4] for r in results], marker='o', linewidth=2, color="#22aa44", label="RCA territory (unchanged)")
ax.plot(st_pcts, [r[5] for r in results], marker='s', linewidth=2, color="black", linestyle="--", label="whole myocardium")
# Reference Gould curve (Q ratio)
ax.plot(st_pcts, [r[1] * results[0][2] for r in results], marker='x', linestyle=':', color="gray", alpha=0.6, label=f"LAD Q-ratio × baseline")
ax.set_xlabel("LAD proximal stenosis (%)", fontsize=12)
ax.set_ylabel("Mean territory perfusion (mL/min/g)", fontsize=12)
ax.set_title("Gould curve in perfusion domain — LAD-territory perfusion vs stenosis severity", fontsize=12)
ax.grid(alpha=0.3); ax.legend(loc="upper right")
ax.set_ylim(0, max(r[2] for r in results) * 1.15)
fig_path = os.path.join(RSNA_DIR, "fig_D_gould_perfusion.png")
plt.savefig(fig_path, dpi=140, bbox_inches="tight"); plt.close()
print(f"[step5] Wrote {fig_path}")

# Print summary
print("\n[step5] === STENOSIS SWEEP SUMMARY ===")
print(f"{'st%':>5}  {'Q_ratio':>8}  {'perf_LAD':>10}  {'perf_LCX':>10}  {'perf_RCA':>10}  {'whole':>8}")
for r in results:
    print(f"{r[0]:>5.0f}  {r[1]:>8.3f}  {r[2]:>10.3f}  {r[3]:>10.3f}  {r[4]:>10.3f}  {r[5]:>8.3f}")
print("\n[step5] Done.")
