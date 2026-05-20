#!/usr/bin/env python3
"""
Build a lightweight standalone HTML 3D viewer for a perfusion map.

Reads a Float32 perfusion map (1600x1400x500 Fortran-order) + the phantom
labels file, picks a random sample of myocardium voxels, and emits a
self-contained Plotly Scatter3d HTML you can open in any browser.

Voxels are colored by perfusion (mL/min/g). The default downsample (~80k
points) keeps the HTML under ~10 MB.

Usage:
    python3 build_perfusion_viewer.py  PERF_NPY  [--n_points 80000]
                                                 [--out output/perfusion_viewer.html]
"""

import argparse
import os
import numpy as np

PHANTOM_LABELS = "/home/molloi-lab/smb_mount/shared_drive/Shu Nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"
NX, NY, NZ = 1600, 1400, 500
VOXEL_CM = 0.02
# XCAT origin (cm) — same as in apply_contrast_at_peak.jl / step3
ORIGIN = (2.846980, -9.773884, -20.600891)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("perf_npy", help="Float32 perfusion map .npy (1600x1400x500 F-order)")
    ap.add_argument("--n_points", type=int, default=80_000,
                    help="number of myocardium voxels to plot (default 80k)")
    ap.add_argument("--out", type=str, default=None,
                    help="output HTML path (default: alongside perf_npy)")
    ap.add_argument("--show_aorta", action="store_true",
                    help="overlay aorta lumen voxels (red)")
    ap.add_argument("--aorta_mask",
                    default="/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline/intermediate/aorta_lumen_mask.raw",
                    help="aorta lumen mask raw path")
    ap.add_argument("--n_aorta_points", type=int, default=20_000,
                    help="aorta voxel subsample size (default 20k)")
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(args.perf_npy) or ".",
                                "perfusion_viewer.html")

    print(f"[viewer] perfusion: {args.perf_npy}")
    print(f"[viewer] phantom labels: {PHANTOM_LABELS}")
    print(f"[viewer] n_points: {args.n_points}")
    print(f"[viewer] output: {args.out}")

    print("[viewer] loading perfusion map …")
    perf = np.load(args.perf_npy)
    # .npy was saved in C-order with shape (NX, NY, NZ) post-reshape — both
    # axes orders agree with the original raw layout (F-order over NX,NY,NZ).
    print(f"[viewer]   shape={perf.shape}  range=[{np.nanmin(perf):.3f}, {np.nanmax(perf):.3f}]")

    print("[viewer] loading phantom labels …")
    labels = np.fromfile(PHANTOM_LABELS, dtype=np.uint8).reshape(
        (NX, NY, NZ), order="F")
    myo_mask = (labels >= 15) & (labels <= 18)
    n_myo = int(myo_mask.sum())
    print(f"[viewer]   myo voxels: {n_myo}")

    rng = np.random.default_rng(20260518)

    # Sample myocardium voxels
    myo_idx = np.argwhere(myo_mask)
    keep = min(args.n_points, n_myo)
    sel = rng.choice(myo_idx.shape[0], size=keep, replace=False)
    samp = myo_idx[sel]
    xs, ys, zs = samp[:, 0], samp[:, 1], samp[:, 2]
    perf_vals = perf[xs, ys, zs].astype(np.float32)
    print(f"[viewer] sampled {keep} myo voxels  perf range=[{perf_vals.min():.3f}, {perf_vals.max():.3f}] mL/min/g")

    # World coordinates (cm)
    x_cm = xs.astype(np.float32) * VOXEL_CM + ORIGIN[0]
    y_cm = ys.astype(np.float32) * VOXEL_CM + ORIGIN[1]
    z_cm = zs.astype(np.float32) * VOXEL_CM + ORIGIN[2]

    aorta_x = aorta_y = aorta_z = None
    if args.show_aorta and os.path.isfile(args.aorta_mask):
        print("[viewer] loading aorta lumen mask …")
        aorta = np.fromfile(args.aorta_mask, dtype=np.uint8).reshape(
            (NX, NY, NZ), order="F").astype(bool)
        n_aorta = int(aorta.sum())
        aorta_idx = np.argwhere(aorta)
        k_aorta = min(args.n_aorta_points, n_aorta)
        sel_a = rng.choice(aorta_idx.shape[0], size=k_aorta, replace=False)
        a_samp = aorta_idx[sel_a]
        aorta_x = a_samp[:, 0].astype(np.float32) * VOXEL_CM + ORIGIN[0]
        aorta_y = a_samp[:, 1].astype(np.float32) * VOXEL_CM + ORIGIN[1]
        aorta_z = a_samp[:, 2].astype(np.float32) * VOXEL_CM + ORIGIN[2]
        print(f"[viewer]   sampled {k_aorta} aorta voxels")

    print("[viewer] writing HTML …")
    title = f"Perfusion map — {os.path.basename(args.perf_npy)}"

    def jfloats(arr, ndp=4):
        # Compact JSON-safe float array (no NaN/Inf in this data, but guard anyway)
        return "[" + ",".join(f"{x:.{ndp}f}" if np.isfinite(x) else "null" for x in arr) + "]"

    # Color scale: fixed clinical REST range [0, 2] mL/min/g so the viewer is
    # comparable across runs. Healthy REST is ~0.5–1.5; values >2 indicate
    # boundary noise (those few outliers will saturate but the bulk of the map
    # is rendered with full dynamic range across the clinical band).
    cmin = 0.0
    cmax = 2.0
    if float(perf_vals.max()) < 1.0:
        cmax = max(1.0, float(perf_vals.max()) * 1.2)
    n_clipped_hi = int((perf_vals > cmax).sum())
    print(f"[viewer]   color scale: [{cmin:.3f}, {cmax:.3f}] mL/min/g  (clipped {n_clipped_hi} outliers > {cmax:.2f})")

    with open(args.out, "w") as f:
        f.write("<!DOCTYPE html><html><head><meta charset=\"utf-8\">\n")
        f.write(f"<title>{title}</title>\n")
        f.write("<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>\n")
        f.write("<style>body{font-family:system-ui,sans-serif;margin:0;padding:8px;"
                "background:#101015;color:#eee}#plot{height:92vh}"
                "h2{margin:4px 0 8px}p{margin:4px 0;color:#aaa;font-size:12px}</style>\n")
        f.write("</head><body>\n")
        f.write(f"<h2>SUBJECT002 REST perfusion map</h2>\n")
        med = float(np.median(perf_vals))
        mean_v = float(perf_vals.mean())
        f.write(f"<p>{keep} of {n_myo} myocardium voxels (random subsample). "
                f"Color = perfusion (mL/min/g), clinical range [{cmin:.2f}, {cmax:.2f}]. "
                f"Subject mean = {mean_v:.3f}, median = {med:.3f} mL/min/g. "
                f"Healthy REST: 0.5–1.5. "
                f"Voxel size 0.02 cm. Drag to rotate; scroll to zoom.</p>\n")
        f.write("<div id=\"plot\"></div>\n<script>\n")
        f.write("const myoTrace={\n")
        f.write("  type:'scatter3d',mode:'markers',name:'myocardium',\n")
        f.write(f"  x:{jfloats(x_cm)},\n  y:{jfloats(y_cm)},\n  z:{jfloats(z_cm)},\n")
        f.write("  marker:{size:1.6,opacity:0.75,\n")
        f.write(f"    color:{jfloats(perf_vals)},\n")
        f.write(f"    cmin:{cmin:.4f},cmax:{cmax:.4f},\n")
        f.write("    colorscale:[[0,'#101040'],[0.25,'#3050a0'],[0.5,'#10a060'],"
                "[0.75,'#ffb060'],[1,'#ffffe0']],\n")
        f.write("    colorbar:{title:{text:'Perfusion (mL/min/g)'},x:1.02,len:0.7}}};\n")
        if aorta_x is not None:
            f.write("const aortaTrace={\n  type:'scatter3d',mode:'markers',name:'aorta lumen',\n")
            f.write(f"  x:{jfloats(aorta_x)},\n  y:{jfloats(aorta_y)},\n  z:{jfloats(aorta_z)},\n")
            f.write("  marker:{size:1.2,opacity:0.6,color:'#cc2222'}};\n")
            f.write("const traces=[myoTrace,aortaTrace];\n")
        else:
            f.write("const traces=[myoTrace];\n")
        f.write("const layout={paper_bgcolor:'#101015',\n")
        f.write("  scene:{aspectmode:'data',bgcolor:'#161620',\n")
        f.write("    xaxis:{title:{text:'x (cm)'},color:'#ccc',gridcolor:'#333',"
                "backgroundcolor:'#101015'},\n")
        f.write("    yaxis:{title:{text:'y (cm)'},color:'#ccc',gridcolor:'#333',"
                "backgroundcolor:'#101015'},\n")
        f.write("    zaxis:{title:{text:'z (cm)'},color:'#ccc',gridcolor:'#333',"
                "backgroundcolor:'#101015'}},\n")
        f.write("  margin:{l:0,r:0,t:30,b:30},showlegend:true,legend:{font:{color:'#ccc'}}};\n")
        f.write("Plotly.newPlot('plot',traces,layout,{responsive:true});\n")
        f.write("</script></body></html>\n")

    sz = os.path.getsize(args.out)
    print(f"[viewer] wrote {args.out} ({sz/1024**2:.2f} MB)")


if __name__ == "__main__":
    main()
