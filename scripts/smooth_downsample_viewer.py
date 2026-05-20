#!/usr/bin/env python3
"""
Smooth a perfusion map with a 5x5x5 average kernel, then downsample by 5x
along each axis, and emit a lightweight HTML 3D viewer of the smoothed
result.

Input  (BasisSim recon resolution):
    perfusion_map_subjectXXX_{rest,stress}.npy   shape (512, 512, 250)  Float32
    myo_mask_subjectXXX_{rest,stress}.npy        shape (512, 512, 250)  UInt8

Pipeline:
  1. uniform_filter(perf, size=5)   # 5x5x5 mean smoothing
  2. uniform_filter(mask, size=5)   # smooth mask (gives fractional support)
  3. Downsample by stride=5 → shape (102, 102, 50)
  4. Threshold smoothed mask > 0.5  (majority-myo blocks)
  5. Emit Plotly Scatter3d HTML

The smoothing reduces CT noise, the downsampling makes the viewer
lightweight (≤ 1 MB instead of 2 MB for full-res 80k voxels).

Usage:
    python3 smooth_downsample_viewer.py  PERF_NPY  MASK_NPY  OUT_HTML  TITLE  CMAX
"""
import argparse
import os
import numpy as np
from scipy.ndimage import uniform_filter

RECON_VOXEL_MM = (0.6836, 0.6836, 0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("perf_npy")
    ap.add_argument("mask_npy")
    ap.add_argument("out_html")
    ap.add_argument("title")
    ap.add_argument("cmax", type=float)
    ap.add_argument("--kernel", type=int, default=5, help="averaging kernel size (5 = 5x5x5 mean)")
    ap.add_argument("--n_points", type=int, default=80_000, help="random sample size for display")
    ap.add_argument("--chamber-mesh",
                    default="/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline/intermediate/chamber_mesh_recon.npz",
                    help="path to chamber_mesh_recon.npz; pass empty to disable")
    ap.add_argument("--tree-overlay",
                    default="",
                    help="path to tree_overlay_*.npz (segments ≥ MIN_DIAM); pass empty to disable")
    ap.add_argument("--tree-radius-scale", type=float, default=1.0,
                    help="visual scale multiplier on segment radius (1.0 = true diameter)")
    ap.add_argument("--tree-sides", type=int, default=6,
                    help="number of sides per cylinder polygon (4-8 reasonable)")
    args = ap.parse_args()

    print(f"[smooth] perf = {args.perf_npy}")
    perf = np.load(args.perf_npy)
    mask = np.load(args.mask_npy).astype(np.float32)
    print(f"[smooth]   input shape: {perf.shape}  dtype={perf.dtype}  mask_nnz={int(mask.sum())}")

    # 5x5x5 average smoothing of perfusion (mask-aware: zero out non-myo so noise
    # outside myo doesn't bleed in, then divide by smoothed mask support)
    perf_zeroed = perf.astype(np.float32) * mask
    perf_smooth_num = uniform_filter(perf_zeroed, size=args.kernel, mode="constant")
    mask_smooth = uniform_filter(mask, size=args.kernel, mode="constant")
    # Where the kernel has any myo support, divide to get the mean of myo voxels
    # inside the kernel window. Below a tiny floor we leave perf=0.
    perf_smooth = np.where(mask_smooth > 1e-6, perf_smooth_num / np.maximum(mask_smooth, 1e-6), 0.0)
    print(f"[smooth]   after {args.kernel}x{args.kernel}x{args.kernel} mean kernel: "
          f"perf range [{perf_smooth.min():.3f}, {perf_smooth.max():.3f}]")

    # Stay at the original 512x512x250 grid; just take the smoothed values at the
    # original myo voxels. Then random-subsample for the viewer.
    myo_mask_bool = mask > 0.5
    xs_all, ys_all, zs_all = np.where(myo_mask_bool)
    n_myo = len(xs_all)
    print(f"[smooth]   myo voxels at native resolution: {n_myo}")

    rng = np.random.default_rng(20260518)
    keep = min(args.n_points, n_myo)
    sel = rng.choice(n_myo, size=keep, replace=False)
    xs, ys, zs = xs_all[sel], ys_all[sel], zs_all[sel]
    vals = perf_smooth[xs, ys, zs].astype(np.float32)

    # World coords (mm) — native voxel grid
    x_mm = xs.astype(np.float32) * RECON_VOXEL_MM[0]
    y_mm = ys.astype(np.float32) * RECON_VOXEL_MM[1]
    z_mm = zs.astype(np.float32) * RECON_VOXEL_MM[2]
    print(f"[smooth]   sampled {keep} voxels for display")

    cmin, cmax = 0.0, args.cmax
    n_above = int((vals > cmax).sum())
    n_below = int((vals < cmin).sum())
    print(f"[smooth]   plot {len(vals)} blocks  color [{cmin:.2f}, {cmax:.2f}]  "
          f"clipped {n_above} above / {n_below} below")
    print(f"[smooth]   stats: mean={vals.mean():.3f}  median={np.median(vals):.3f}  "
          f"std={vals.std():.3f}  p5..p95=[{np.percentile(vals,5):.3f}, {np.percentile(vals,95):.3f}]")

    def jfloats(arr, ndp=3):
        return "[" + ",".join(f"{x:.{ndp}f}" if np.isfinite(x) else "null" for x in arr) + "]"

    def jints(arr):
        return "[" + ",".join(str(int(x)) for x in arr) + "]"

    # Load optional chamber mesh
    chamber_data = None
    if args.chamber_mesh and os.path.isfile(args.chamber_mesh):
        try:
            chamber_data = np.load(args.chamber_mesh)
            print(f"[smooth]   chamber mesh: {len(chamber_data['verts_mm'])} verts, "
                  f"{len(chamber_data['faces'])} faces")
        except Exception as e:
            print(f"[smooth]   WARN: could not load chamber mesh: {e}")
            chamber_data = None

    # Load optional tree overlay
    tree_data = None
    if args.tree_overlay and os.path.isfile(args.tree_overlay):
        try:
            tree_data = np.load(args.tree_overlay)
            n_segs = (len(tree_data["x"]) // 3)
            print(f"[smooth]   tree overlay: {n_segs} segments ≥{float(tree_data['min_diam_um']):.0f} μm")
        except Exception as e:
            print(f"[smooth]   WARN: could not load tree overlay: {e}")
            tree_data = None

    with open(args.out_html, "w") as f:
        f.write("<!DOCTYPE html><html><head><meta charset=\"utf-8\">\n")
        f.write(f"<title>{args.title}</title>\n")
        f.write("<script src=\"https://cdn.plot.l y/plotly-2.35.2.min.js\"></script>\n".replace("plot.l y", "plot.ly"))
        f.write("<style>body{font-family:system-ui,sans-serif;margin:0;padding:8px;"
                "background:#101015;color:#eee}#plot{height:92vh}"
                "h2{margin:4px 0 8px}p{margin:4px 0;color:#aaa;font-size:12px}</style>\n")
        f.write("</head><body>\n")
        f.write(f"<h2>{args.title}</h2>\n")
        f.write(f"<p>{args.kernel}×{args.kernel}×{args.kernel} mask-aware mean smoothing at native recon resolution "
                f"(512×512×250), then random subsample {keep} of {n_myo} myo voxels. "
                f"Stats over subsample: mean={vals.mean():.3f}, median={np.median(vals):.3f}, "
                f"p5–p95 [{np.percentile(vals,5):.3f}, {np.percentile(vals,95):.3f}] mL/min/g. "
                f"Color scale [{cmin:.2f}, {cmax:.2f}]. Drag to rotate; scroll to zoom.</p>\n")
        f.write("<div id=\"plot\"></div>\n<script>\n")
        # Chamber mesh (renders FIRST so scatter draws on top; opaque so it
        # occludes any scatter points sitting behind it from the camera).
        traces = []
        if chamber_data is not None:
            v = chamber_data["verts_mm"]
            faces = chamber_data["faces"]
            f.write("const chamberMesh={\n")
            f.write("  type:'mesh3d',name:'chambers + aorta (opaque)',\n")
            f.write(f"  x:{jfloats(v[:,0], 2)},\n  y:{jfloats(v[:,1], 2)},\n  z:{jfloats(v[:,2], 2)},\n")
            f.write(f"  i:{jints(faces[:,0])},\n  j:{jints(faces[:,1])},\n  k:{jints(faces[:,2])},\n")
            f.write("  color:'#4a4a4a',opacity:1.0,flatshading:false,\n")
            f.write("  lighting:{ambient:0.55,diffuse:0.7,specular:0.05,roughness:0.9,fresnel:0.05},\n")
            f.write("  lightposition:{x:1000,y:1000,z:1000},\n")
            f.write("  showscale:false,hoverinfo:'skip'};\n")
            traces.append("chamberMesh")

        f.write("const perfTrace={\n")
        f.write("  type:'scatter3d',mode:'markers',name:'myocardium (smoothed)',\n")
        f.write(f"  x:{jfloats(x_mm, 2)},\n  y:{jfloats(y_mm, 2)},\n  z:{jfloats(z_mm, 2)},\n")
        f.write("  marker:{size:2.0,opacity:0.80,\n")
        f.write(f"    color:{jfloats(vals, 4)},\n")
        f.write(f"    cmin:{cmin:.4f},cmax:{cmax:.4f},\n")
        f.write("    colorscale:[[0,'#00007f'],[0.1,'#0000ff'],[0.3,'#00ffff'],"
                "[0.5,'#7fff7f'],[0.7,'#ffff00'],[0.9,'#ff7f00'],[1,'#7f0000']],\n")
        f.write("    colorbar:{title:{text:'Perfusion (mL/min/g)'},x:1.02,len:0.7}}};\n")
        traces.append("perfTrace")

        # Tree overlay rendered as cylinder MESHES with TRUE diameter per segment.
        # One Mesh3d per tree (LAD/LCX/RCA) so each can be toggled independently.
        tree_trace_indices = {}   # tree_id → trace position in `traces` list
        if tree_data is not None:
            starts = np.asarray(tree_data["starts_mm"], dtype=np.float64)
            ends   = np.asarray(tree_data["ends_mm"], dtype=np.float64)
            tree_ids = np.asarray(tree_data["tree_id"], dtype=np.int32)
            diams   = np.asarray(tree_data["diameter_um"], dtype=np.float64)
            min_d   = float(tree_data["min_diam_um"])
            n_sides = int(args.tree_sides)
            r_scale = float(args.tree_radius_scale)
            COLORS = {0: "#1f77ff", 1: "#e3342f", 2: "#22aa44"}
            NAMES  = {0: "LAD", 1: "LCX", 2: "RCA"}

            angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
            cos_a = np.cos(angles); sin_a = np.sin(angles)

            for tid in (0, 1, 2):
                seg_mask = tree_ids == tid
                if not seg_mask.any():
                    continue
                S = starts[seg_mask]; E = ends[seg_mask]
                R = (diams[seg_mask] / 2.0 / 1000.0) * r_scale   # μm → mm radius (× scale)

                N = len(S)
                vx_all = np.empty(N * 2 * n_sides, dtype=np.float32)
                vy_all = np.empty(N * 2 * n_sides, dtype=np.float32)
                vz_all = np.empty(N * 2 * n_sides, dtype=np.float32)
                fi_all = np.empty(N * 2 * n_sides, dtype=np.int32)
                fj_all = np.empty(N * 2 * n_sides, dtype=np.int32)
                fk_all = np.empty(N * 2 * n_sides, dtype=np.int32)
                vptr = 0; fptr = 0
                for s in range(N):
                    a = S[s]; b = E[s]; r = R[s]
                    axis = b - a
                    L = np.linalg.norm(axis)
                    if L < 1e-9 or r < 1e-9:
                        continue
                    ah = axis / L
                    # Pick a reference vector not parallel to axis
                    ref = np.array([0., 0., 1.]) if abs(ah[2]) < 0.9 else np.array([1., 0., 0.])
                    p1 = np.cross(ah, ref); p1 = p1 / np.linalg.norm(p1)
                    p2 = np.cross(ah, p1)
                    # 2N ring verts: first N at start, next N at end
                    offsets = (cos_a[:, None] * p1 + sin_a[:, None] * p2) * r   # (n_sides, 3)
                    start_ring = a[None, :] + offsets       # (n_sides, 3)
                    end_ring   = b[None, :] + offsets
                    base = vptr
                    vx_all[base:base+n_sides] = start_ring[:, 0]
                    vy_all[base:base+n_sides] = start_ring[:, 1]
                    vz_all[base:base+n_sides] = start_ring[:, 2]
                    vx_all[base+n_sides:base+2*n_sides] = end_ring[:, 0]
                    vy_all[base+n_sides:base+2*n_sides] = end_ring[:, 1]
                    vz_all[base+n_sides:base+2*n_sides] = end_ring[:, 2]
                    # 2*n_sides side triangles per segment
                    for k in range(n_sides):
                        kn = (k + 1) % n_sides
                        # Quad: (k, k+N, kn+N, kn) → 2 tris
                        fi_all[fptr]   = base + k;       fj_all[fptr]   = base + n_sides + k;  fk_all[fptr]   = base + n_sides + kn
                        fi_all[fptr+1] = base + k;       fj_all[fptr+1] = base + n_sides + kn; fk_all[fptr+1] = base + kn
                        fptr += 2
                    vptr += 2 * n_sides

                # Trim to actual used size
                vx_all = vx_all[:vptr]; vy_all = vy_all[:vptr]; vz_all = vz_all[:vptr]
                fi_all = fi_all[:fptr]; fj_all = fj_all[:fptr]; fk_all = fk_all[:fptr]

                f.write(f"const tree_{NAMES[tid]}={{\n")
                f.write("  type:'mesh3d',\n")
                f.write(f"  name:'{NAMES[tid]} (≥{min_d:.0f} μm)',\n")
                f.write(f"  x:{jfloats(vx_all, 2)},\n  y:{jfloats(vy_all, 2)},\n  z:{jfloats(vz_all, 2)},\n")
                f.write(f"  i:{jints(fi_all)},\n  j:{jints(fj_all)},\n  k:{jints(fk_all)},\n")
                f.write(f"  color:'{COLORS[tid]}',opacity:1.0,flatshading:false,\n")
                f.write("  lighting:{ambient:0.55,diffuse:0.7,specular:0.15,roughness:0.7,fresnel:0.1},\n")
                f.write("  lightposition:{x:1000,y:1000,z:1000},\n")
                f.write("  showscale:false,hoverinfo:'skip'};\n")
                tree_trace_indices[tid] = len(traces)
                traces.append(f"tree_{NAMES[tid]}")
        f.write("const layout={paper_bgcolor:'#101015',\n")
        f.write("  scene:{aspectmode:'data',bgcolor:'#161620',\n")
        f.write("    xaxis:{title:{text:'x (mm)'},color:'#ccc',gridcolor:'#333',"
                "backgroundcolor:'#101015'},\n")
        f.write("    yaxis:{title:{text:'y (mm)'},color:'#ccc',gridcolor:'#333',"
                "backgroundcolor:'#101015'},\n")
        f.write("    zaxis:{title:{text:'z (mm)'},color:'#ccc',gridcolor:'#333',"
                "backgroundcolor:'#101015'}},\n")
        f.write("  margin:{l:80,r:0,t:30,b:30},showlegend:false");
        # Toggle buttons for LAD/LCX/RCA: each click toggles the trace's visibility
        # between 'true' and 'legendonly' (which hides it from the scene).
        if tree_trace_indices:
            button_specs = []
            COLORS = {0: "#1f77ff", 1: "#e3342f", 2: "#22aa44"}
            NAMES  = {0: "LAD", 1: "LCX", 2: "RCA"}
            for tid in (0, 1, 2):
                if tid not in tree_trace_indices:
                    continue
                idx = tree_trace_indices[tid]
                # args: when first clicked, hide; args2: when clicked again, show
                button_specs.append(
                    f"{{label:'Toggle {NAMES[tid]}',"
                    f"method:'restyle',"
                    f"args:[{{'visible':'legendonly'}},[{idx}]],"
                    f"args2:[{{'visible':true}},[{idx}]],"
                    f"bgcolor:'{COLORS[tid]}',"
                    f"font:{{color:'#ffffff'}}}}"
                )
            f.write(",\n  updatemenus:[{")
            f.write("type:'buttons',direction:'left',showactive:false,")
            f.write("x:0,xanchor:'left',y:1.05,yanchor:'bottom',")
            f.write("pad:{l:5,r:5,t:5,b:5},bgcolor:'#222',")
            f.write("buttons:[" + ",".join(button_specs) + "]}]")
        f.write("};\n")
        f.write(f"Plotly.newPlot('plot',[{','.join(traces)}],layout,{{responsive:true}});\n")
        f.write("</script></body></html>\n")
    print(f"[smooth] wrote {args.out_html}  ({os.path.getsize(args.out_html)/1024**2:.2f} MB)")


if __name__ == "__main__":
    main()
