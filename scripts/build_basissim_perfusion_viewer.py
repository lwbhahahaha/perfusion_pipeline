#!/usr/bin/env python3
"""
Build a lightweight HTML 3D viewer for the Path-B BasisSim perfusion map.

Reads:
  output/perfusion_map_subject002_rest_basissim.npy   (Float32, 512×512×250)
  output/myo_mask_subject002_rest_basissim.npy        (UInt8, same shape)

Outputs an HTML with a Plotly Scatter3d of the myocardium voxels colored by
perfusion (mL/min/g). The mask is computed in recon coordinates by step5.

Usage:
    python3 build_basissim_perfusion_viewer.py  [--n_points 80000]
"""

import argparse
import os
import numpy as np

PIPELINE = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline"
PERF_PATH = os.path.join(PIPELINE, "output", "perfusion_map_subject002_rest_basissim.npy")
MASK_PATH = os.path.join(PIPELINE, "output", "myo_mask_subject002_rest_basissim.npy")
OUT_PATH = os.path.join(PIPELINE, "output", "perfusion_viewer_subject002_rest_basissim.html")
DEFAULT_TITLE = "SUBJECT002 REST — BasisSim perfusion map (path B)"

# Recon geometry (matches step5 reading of recon_meta.toml)
RECON_VOXEL_MM = (0.6836, 0.6836, 0.2)
RECON_SHAPE = (512, 512, 250)
# BasisSim places phantom centered at the iso; iso is at the origin of the
# recon volume — we just plot voxel indices for now (no absolute world coords),
# since the perfusion map is recon-centric.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_points", type=int, default=80_000,
                    help="number of myo voxels to plot")
    ap.add_argument("--perf", type=str, default=PERF_PATH,
                    help="path to perfusion map .npy")
    ap.add_argument("--mask", type=str, default=MASK_PATH,
                    help="path to myo mask .npy")
    ap.add_argument("--out", type=str, default=OUT_PATH)
    ap.add_argument("--title", type=str, default=DEFAULT_TITLE)
    ap.add_argument("--cmax", type=float, default=3.0,
                    help="colorbar maximum mL/min/g (default 3.0 for REST; use 5.0 for STRESS)")
    args = ap.parse_args()

    print(f"[viewer] loading perfusion map: {args.perf}")
    perf = np.load(args.perf)
    print(f"  shape={perf.shape} dtype={perf.dtype} range=[{perf.min():.3f}, {perf.max():.3f}]")
    print(f"[viewer] loading myo mask: {args.mask}")
    mask = np.load(args.mask).astype(bool)
    n_myo = int(mask.sum())
    print(f"  myo voxels in recon: {n_myo}")

    rng = np.random.default_rng(20260518)
    myo_idx = np.argwhere(mask)
    keep = min(args.n_points, n_myo)
    sel = rng.choice(myo_idx.shape[0], size=keep, replace=False)
    samp = myo_idx[sel]
    xs, ys, zs = samp[:, 0], samp[:, 1], samp[:, 2]
    perf_vals = perf[xs, ys, zs].astype(np.float32)

    # World-ish coords for plotting (recon voxel grid in mm)
    x_mm = xs.astype(np.float32) * RECON_VOXEL_MM[0]
    y_mm = ys.astype(np.float32) * RECON_VOXEL_MM[1]
    z_mm = zs.astype(np.float32) * RECON_VOXEL_MM[2]

    cmin, cmax = 0.0, args.cmax  # match the real-scan colorbar
    n_clip_hi = int((perf_vals > cmax).sum())
    n_clip_lo = int((perf_vals < cmin).sum())
    print(f"[viewer] sampled {keep} voxels  perf range=[{perf_vals.min():.3f}, {perf_vals.max():.3f}]")
    print(f"[viewer] color scale [{cmin:.2f}, {cmax:.2f}]  (clipped: {n_clip_hi} above, {n_clip_lo} below)")

    def jfloats(arr, ndp=4):
        return "[" + ",".join(f"{x:.{ndp}f}" if np.isfinite(x) else "null" for x in arr) + "]"

    mean_v = float(perf_vals.mean())
    med_v  = float(np.median(perf_vals))
    p5     = float(np.percentile(perf_vals, 5))
    p95    = float(np.percentile(perf_vals, 95))

    title = args.title
    with open(args.out, "w") as f:
        f.write("<!DOCTYPE html><html><head><meta charset=\"utf-8\">\n")
        f.write(f"<title>{title}</title>\n")
        f.write("<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>\n")
        f.write("<style>body{font-family:system-ui,sans-serif;margin:0;padding:8px;"
                "background:#101015;color:#eee}#plot{height:92vh}"
                "h2{margin:4px 0 8px}p{margin:4px 0;color:#aaa;font-size:12px}</style>\n")
        f.write("</head><body>\n")
        f.write(f"<h2>{title}</h2>\n")
        f.write(f"<p>{keep} of {n_myo} myocardium voxels (random subsample) at recon resolution "
                f"(512×512×250 @ 0.68×0.68×0.2 mm). Color = perfusion (mL/min/g), scale [{cmin:.2f}, {cmax:.2f}]. "
                f"Mean = {mean_v:.3f}, median = {med_v:.3f}, p5–p95 = [{p5:.3f}, {p95:.3f}] mL/min/g. "
                f"Spatial heterogeneity comes from BasisSim CT photon noise + recon (no autoregulation defect). "
                f"Drag to rotate; scroll to zoom.</p>\n")
        f.write("<div id=\"plot\"></div>\n<script>\n")
        f.write("const trace={\n")
        f.write("  type:'scatter3d',mode:'markers',name:'myocardium (path B)',\n")
        f.write(f"  x:{jfloats(x_mm, 2)},\n  y:{jfloats(y_mm, 2)},\n  z:{jfloats(z_mm, 2)},\n")
        f.write("  marker:{size:2.0,opacity:0.80,\n")
        f.write(f"    color:{jfloats(perf_vals, 4)},\n")
        f.write(f"    cmin:{cmin:.4f},cmax:{cmax:.4f},\n")
        # Jet-like colormap matching the user's screenshot of the real scan
        f.write("    colorscale:[[0,'#00007f'],[0.1,'#0000ff'],[0.3,'#00ffff'],"
                "[0.5,'#7fff7f'],[0.7,'#ffff00'],[0.9,'#ff7f00'],[1,'#7f0000']],\n")
        f.write("    colorbar:{title:{text:'Perfusion (mL/min/g)'},x:1.02,len:0.7}}};\n")
        f.write("const layout={paper_bgcolor:'#101015',\n")
        f.write("  scene:{aspectmode:'data',bgcolor:'#161620',\n")
        f.write("    xaxis:{title:{text:'x (mm)'},color:'#ccc',gridcolor:'#333',"
                "backgroundcolor:'#101015'},\n")
        f.write("    yaxis:{title:{text:'y (mm)'},color:'#ccc',gridcolor:'#333',"
                "backgroundcolor:'#101015'},\n")
        f.write("    zaxis:{title:{text:'z (mm)'},color:'#ccc',gridcolor:'#333',"
                "backgroundcolor:'#101015'}},\n")
        f.write("  margin:{l:0,r:0,t:30,b:30},showlegend:false};\n")
        f.write("Plotly.newPlot('plot',[trace],layout,{responsive:true});\n")
        f.write("</script></body></html>\n")

    sz = os.path.getsize(args.out)
    print(f"[viewer] wrote {args.out} ({sz/1024**2:.2f} MB)")


if __name__ == "__main__":
    main()
