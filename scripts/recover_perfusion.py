#!/usr/bin/env python3
"""recover_perfusion.py — measure myocardial perfusion from the honest dynamic recon series.

Reads the reconstructed HU volumes produced by run_dynamic_perfusion.sh, maps the
XCAT myocardium (and per-territory tree labels) into recon space, and reports the
max-slope CT-MPI perfusion vs. the known input flow. No contrast is imposed here;
the myocardial enhancement is whatever the recon actually contains.

Env:
  OUT    output dir from run_dynamic_perfusion.sh (has recon_*s/, intermediate/)
  XCAT   vmale50 UInt8 phantom raw
  TID    per-voxel tree-territory raw (UInt8, 1=LAD 2=LCX 3=RCA) in phantom space
  FRAMES space-separated frame times (default "0 12 15 18 21 24")
  INPUT_P  "LAD,LCX,RCA" ground-truth flow, mL/min/g (default 4.20,4.01,3.94)
"""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

NX, NY, NZ = 1600, 1400, 500
DOWN = 2; BSX, BSY, BSZ = 800, 700, 250; BSVOX = 0.2; VX, VZ = 0.6836, 0.2; RHO = 1.053

OUT = os.environ["OUT"]
XCAT = os.environ["XCAT"]
TID = os.environ.get("TID", os.path.join(OUT, "..", "intermediate", "myo_tree_id.raw"))
times = np.array([int(x) for x in os.environ.get("FRAMES", "0 12 15 18 21 24").split()], float)
inP = {i + 1: float(x) for i, x in enumerate(os.environ.get("INPUT_P", "4.20,4.01,3.94").split(","))}
names = {1: "LAD", 2: "LCX", 3: "RCA"}; col = {1: "tab:blue", 2: "tab:red", 3: "tab:green"}

xc = np.fromfile(XCAT, dtype=np.uint8).reshape((NX, NY, NZ), order="F")
tid = np.fromfile(TID, dtype=np.uint8).reshape((NX, NY, NZ), order="F")


def to_recon(m):
    """Map a phantom-space boolean mask into the 512x512x250 recon grid."""
    m = m[:, ::-1, ::-1][1::DOWN, 1::DOWN, 1::DOWN]
    sx, sz = VX / BSVOX, VZ / BSVOX
    phc = [BSX / 2 - .5, BSY / 2 - .5, BSZ / 2 - .5]; rc = [255.5, 255.5, 124.5]
    ii = np.rint((np.arange(512) - rc[0]) * sx + phc[0]).astype(int)
    jj = np.rint((np.arange(512) - rc[1]) * sx + phc[1]).astype(int)
    kk = np.rint((np.arange(250) - rc[2]) * sz + phc[2]).astype(int)
    ok = lambda a, n: (a >= 0) & (a < n)
    r = m[np.clip(ii, 0, BSX - 1)[:, None, None], np.clip(jj, 0, BSY - 1)[None, :, None],
          np.clip(kk, 0, BSZ - 1)[None, None, :]]
    return r & (ok(ii, BSX)[:, None, None] & ok(jj, BSY)[None, :, None] & ok(kk, BSZ)[None, None, :])


lvR = to_recon(xc == 19)
tidR = np.zeros((512, 512, 250), np.uint8)
for t in (1, 2, 3):
    tidR[to_recon(tid == t)] = t

HUlv = []; HUt = {1: [], 2: [], 3: []}
for f in times.astype(int):
    r = np.fromfile(f"{OUT}/recon_{f:02d}s/recon_fbp_hu_f32.raw", dtype=np.float32).reshape((512, 512, 250), order="F")
    HUlv.append(float(r[lvR].mean()))
    for terr in (1, 2, 3):
        HUt[terr].append(float(r[tidR == terr].mean()))
HUlv = np.array(HUlv); dHUaif = HUlv.max() - HUlv[0]

print(f"AIF (LV) HU: {[f'{x:.0f}' for x in HUlv]}")
print(f"\n{'terr':>5} {'peak myo HU':>12} {'recovered P':>12} {'input P':>9}")
recP = {}
for terr in (1, 2, 3):
    C = np.array(HUt[terr]); dC = C - C[0]
    ms = np.gradient(dC, times).max(); recP[terr] = ms / dHUaif * 60 / RHO
    print(f"{names[terr]:>5} {C.max():12.0f} {recP[terr]:12.2f} {inP[terr]:9.2f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.plot(times, HUlv, 'k-o', lw=2, label="AIF (LV blood pool)")
for t in (1, 2, 3):
    a1.plot(times, HUt[t], '-o', color=col[t], label=f"{names[t]} myo (real capillary contrast)")
a1.axhline(85, ls=':', c='gray'); a1.text(0.5, 88, "real-patient myo ~85 HU", fontsize=8, c='gray')
a1.set_xlabel("time (s)"); a1.set_ylabel("HU"); a1.legend(fontsize=8); a1.grid(alpha=.3)
a1.set_title("Dynamic enhancement — real simulated contrast (PDE→voxelize→CT)")
a2.plot([0, 5], [0, 5], 'k--', label="ideal")
for t in (1, 2, 3):
    a2.plot(inP[t], recP[t], 'o', ms=12, color=col[t], label=f"{names[t]}: {inP[t]:.1f}→{recP[t]:.2f}")
a2.set_xlabel("input flow (mL/min/g)"); a2.set_ylabel("recovered (max-slope)")
a2.set_title("CT-MPI flow roll-off"); a2.legend(fontsize=8); a2.grid(alpha=.3); a2.set_xlim(0, 5); a2.set_ylim(0, 5)
plt.tight_layout(); plt.savefig(f"{OUT}/dynamic_perfusion_figure.png", dpi=130)
print(f"saved {OUT}/dynamic_perfusion_figure.png")
