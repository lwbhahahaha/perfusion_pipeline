#!/usr/bin/env python3
"""
Surface perfusion viewer — renders the smoothed perfusion map as a continuous
3D mesh surface (Plotly Mesh3d with per-vertex intensity), in the style of a
cortical-surface mapping. Chamber mesh + cylinder tree overlay layers stay the
same as smooth_downsample_viewer.

Pipeline:
  1. 5x5x5 mask-aware mean smooth the perfusion map (same as scatter version).
  2. Gaussian-smooth the myocardium mask (sigma in voxel units) and run
     marching_cubes at level=0.5 → triangular surface (verts in recon voxel
     idx + faces).
  3. Trilinear-sample the smoothed perfusion at every surface vertex.
  4. Plotly Mesh3d with intensity = perfusion, jet-like colorscale.
  5. Optional chamber mesh (opaque, dark gray) underneath and
     LAD/LCX/RCA cylinder tree on top — both unchanged from the scatter
     version.

Usage:
  python3 surface_perfusion_viewer.py  PERF_NPY  MASK_NPY  OUT_HTML  TITLE  CMAX
        [--kernel 5] [--mask-smooth-sigma 1.0]
        [--chamber-mesh <path>] [--tree-overlay <path>]
        [--tree-radius-scale 1.0] [--tree-sides 6]
"""
import argparse
import os
import numpy as np
from scipy.ndimage import (uniform_filter, gaussian_filter, map_coordinates,
                           binary_closing, label, generate_binary_structure,
                           distance_transform_edt)
from skimage.measure import marching_cubes

RECON_VOXEL_MM = (0.6836, 0.6836, 0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("perf_npy")
    ap.add_argument("mask_npy")
    ap.add_argument("out_html")
    ap.add_argument("title")
    ap.add_argument("cmax", type=float)
    ap.add_argument("--kernel", type=int, default=1,
                    help="averaging kernel size for OPTIONAL pre-smoothing of "
                         "perfusion. Default 1 = NO smoothing (raw values are "
                         "sampled mask-aware-trilinear at each vertex). Set >1 "
                         "to re-enable the old uniform_filter behavior.")
    ap.add_argument("--mask-smooth-sigma", type=float, default=1.0,
                    help="Gaussian sigma (voxels) applied to the mask before marching_cubes")
    ap.add_argument("--mask-close-iters", type=int, default=2,
                    help="binary-closing iterations on the mask (fills sub-voxel "
                         "wide gaps without significantly inflating the boundary). "
                         "Stay ≤ 2 if --kernel=1, otherwise vertices may land "
                         "beyond the mask-zeroed perf field and sample 0.")
    ap.add_argument("--mc-level", type=float, default=0.4,
                    help="marching_cubes iso level. Default 0.4 (slightly below "
                         "0.5) so thin myocardium walls — whose Gaussian-smoothed "
                         "value sits at ~0.4–0.5 — still generate triangles")
    ap.add_argument("--mask-fill-max-voxels", type=int, default=300,
                    help="size-thresholded hole fill: any enclosed cavity in the "
                         "mask complement smaller than this is filled. Chambers "
                         "(thousands of voxels) and outside (millions) are skipped.")
    ap.add_argument("--chamber-mesh",
                    default="/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline/intermediate/chamber_mesh_recon.npz")
    ap.add_argument("--tree-overlay", default="")
    ap.add_argument("--tree-radius-scale", type=float, default=1.0)
    ap.add_argument("--tree-sides", type=int, default=6)
    args = ap.parse_args()

    print(f"[surface] perf = {args.perf_npy}")
    perf = np.load(args.perf_npy)
    mask = np.load(args.mask_npy).astype(np.float32)
    print(f"[surface]   input: perf {perf.shape}  mask nnz = {int(mask.sum())}")

    # --- Build the perfusion field for vertex sampling ---
    # The marching-cubes surface may extend a couple voxels past the binary
    # mask boundary (because we morphologically closed the mask + use mc level
    # < 0.5 to bridge sub-voxel tears). At those out-of-mask vertex positions,
    # raw `perf` is 0 (BasisSim only writes inside the mask), which would make
    # the rim look dark.
    #
    # Fix without introducing any spatial smoothing kernel:
    # nearest-neighbor extrapolation via distance_transform_edt. For every
    # out-of-mask voxel, we copy the perfusion value from the closest in-mask
    # voxel. No averaging — exactly the "fill from neighboring pixel" idea.
    mask_bool = mask.astype(bool)
    print(f"[surface]   building EDT-based nearest-neighbor perf extension …")
    _, nn_idx = distance_transform_edt(~mask_bool, return_indices=True)
    perf_filled = perf.astype(np.float32)[nn_idx[0], nn_idx[1], nn_idx[2]]
    if args.kernel > 1:
        # Legacy: optional uniform_filter smoothing on top of the extension.
        perf_smooth = uniform_filter(perf_filled, size=args.kernel, mode="constant")
        print(f"[surface]   + {args.kernel}^3 mean kernel pre-smoothing")
    else:
        perf_smooth = perf_filled
        print(f"[surface]   no pre-smoothing kernel (EDT nearest-neighbor only)")
    print(f"[surface]   perfusion field range [{perf_smooth.min():.2f}, {perf_smooth.max():.2f}]")

    # --- Build myocardium surface mesh ---
    # (1) Mild closing fills 1-2 voxel diagonal gaps in the binary mask.
    mask_bin = (mask > 0.5)
    n0 = int(mask_bin.sum())
    if args.mask_close_iters > 0:
        struct18 = generate_binary_structure(3, 2)  # 18-connected
        mask_bin = binary_closing(mask_bin, structure=struct18, iterations=args.mask_close_iters)
    n1 = int(mask_bin.sum())
    # (2) Size-thresholded fill_holes: label connected components in the
    #     COMPLEMENT of the mask. The huge background is the true outside.
    #     The LV/RV chambers are the next-largest components (~thousands of
    #     voxels). Anything smaller than --mask-fill-max-voxels is a
    #     surface-tear pinhole and we fill it.
    if args.mask_fill_max_voxels > 0:
        struct6 = generate_binary_structure(3, 1)  # 6-connected (strict for holes)
        lbl, n_comp = label(~mask_bin, structure=struct6)
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0  # label 0 is the mask itself; ignore
        fill_count = 0
        fill_voxels = 0
        # Build a boolean mask of components to fill in one vectorized pass
        small_labels = np.where((sizes > 0) & (sizes < args.mask_fill_max_voxels))[0]
        if small_labels.size > 0:
            keep = np.zeros(n_comp + 1, dtype=bool)
            keep[small_labels] = True
            fill_bool = keep[lbl]
            fill_voxels = int(fill_bool.sum())
            fill_count = int(small_labels.size)
            mask_bin |= fill_bool
        print(f"[surface]   filled {fill_count} pinhole components "
              f"({fill_voxels} voxels, each ≤ {args.mask_fill_max_voxels} vox)")
    n2 = int(mask_bin.sum())
    print(f"[surface]   mask: {n0} → close {n1} → fill {n2}  (+{n2-n0} voxels, +{100*(n2-n0)/max(n0,1):.1f}%)")
    # (3) Gaussian smooth + marching cubes on the cleaned mask. Lower level
    #     (0.4 by default) bridges thin walls whose smoothed value sits ~0.4.
    mask_g = gaussian_filter(mask_bin.astype(np.float32), sigma=args.mask_smooth_sigma)
    verts_idx, faces, _, _ = marching_cubes(mask_g, level=args.mc_level)
    verts_mm = verts_idx.astype(np.float32) * np.array(RECON_VOXEL_MM, dtype=np.float32)
    print(f"[surface]   marching_cubes @ level={args.mc_level} → "
          f"{len(verts_mm)} verts, {len(faces)} faces")

    # --- Sample smoothed perfusion at every surface vertex (trilinear) ---
    coords = verts_idx.T  # (3, N)
    perf_vals = map_coordinates(perf_smooth, coords, order=1, mode="constant", cval=0.0).astype(np.float32)
    print(f"[surface]   per-vertex perfusion: mean={perf_vals.mean():.3f}  "
          f"median={np.median(perf_vals):.3f}  std={perf_vals.std():.3f}")
    print(f"[surface]                       p5..p95 = [{np.percentile(perf_vals, 5):.3f}, "
          f"{np.percentile(perf_vals, 95):.3f}]")

    cmin, cmax = 0.0, args.cmax
    n_above = int((perf_vals > cmax).sum())
    print(f"[surface]   color scale [{cmin:.2f}, {cmax:.2f}]  ({n_above} verts clipped above)")

    def jfloats(arr, ndp=3):
        return "[" + ",".join(f"{x:.{ndp}f}" if np.isfinite(x) else "null" for x in arr) + "]"

    def jints(arr):
        return "[" + ",".join(str(int(x)) for x in arr) + "]"

    # --- Optional chamber mesh ---
    chamber_data = None
    if args.chamber_mesh and os.path.isfile(args.chamber_mesh):
        try:
            chamber_data = np.load(args.chamber_mesh)
            print(f"[surface]   chamber mesh: {len(chamber_data['verts_mm'])} verts, "
                  f"{len(chamber_data['faces'])} faces")
        except Exception as e:
            print(f"[surface]   WARN: chamber mesh load failed: {e}")

    # --- Optional cylinder tree overlay ---
    tree_data = None
    if args.tree_overlay and os.path.isfile(args.tree_overlay):
        try:
            tree_data = np.load(args.tree_overlay)
            print(f"[surface]   tree overlay: {len(tree_data['diameter_um'])} segments")
        except Exception as e:
            print(f"[surface]   WARN: tree overlay load failed: {e}")

    # --- HTML output ---
    with open(args.out_html, "w") as f:
        f.write("<!DOCTYPE html><html><head><meta charset=\"utf-8\">\n")
        f.write(f"<title>{args.title}</title>\n")
        f.write("<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>\n")
        f.write("<style>body{font-family:system-ui,sans-serif;margin:0;padding:8px;"
                "background:#101015;color:#eee}#plot{height:78vh}"
                "h2{margin:4px 0 8px}p{margin:4px 0;color:#aaa;font-size:12px}"
                ".camctl{display:grid;grid-template-columns:auto 1fr auto 1fr;"
                "gap:4px 8px;align-items:center;padding:6px 10px;background:#181820;"
                "border:1px solid #2a2a35;border-radius:6px;font-size:11px;"
                "margin:6px 0 4px}"
                ".camctl label{color:#aaa}.camctl input[type=range]{width:100%}"
                ".camctl input[type=number]{width:60px;background:#0d0d12;color:#eee;"
                "border:1px solid #333;padding:2px 4px;font-size:11px}"
                ".camctl-row{display:flex;gap:8px;align-items:center;"
                "padding:4px 10px 8px 10px;font-size:11px}"
                ".camctl-row button{background:#2a2a45;color:#eee;border:1px solid #444;"
                "padding:3px 8px;cursor:pointer;border-radius:3px;font-size:11px}"
                ".camctl-row button:hover{background:#3a3a55}"
                ".camctl-row textarea{flex:1;background:#0d0d12;color:#7df;"
                "border:1px solid #333;font-family:monospace;font-size:10px;"
                "padding:3px;height:36px;resize:vertical}</style>\n")
        f.write("</head><body>\n")
        f.write(f"<h2>{args.title}</h2>\n")
        smooth_desc = (f"EDT nearest-neighbor extension + {args.kernel}³ mean pre-smoothing"
                       if args.kernel > 1 else
                       "EDT nearest-neighbor extension only (no smoothing kernel)")
        f.write(f"<p>Continuous mesh surface (per-vertex perfusion, jet colorscale "
                f"[{cmin:.2f}, {cmax:.2f}] mL/min/g). "
                f"{len(verts_mm)} verts, {len(faces)} faces; "
                f"{smooth_desc}; "
                f"closing×{args.mask_close_iters} + σ={args.mask_smooth_sigma} Gaussian + "
                f"marching_cubes(level={args.mc_level}) on the myocardium mask. "
                f"Drag to rotate; scroll to zoom.</p>\n")
        # Camera control panel: sliders + number inputs + JSON copy/paste so
        # the same view can be applied to multiple viewers for matched
        # side-by-side screenshots.
        f.write("""<div class="camctl">
  <label>Eye x</label><input type="range" id="rEX" min="-3" max="3" step="0.01" value="1.25">
  <input type="number" id="nEX" step="0.01" value="1.25">

  <label>Up x</label><input type="range" id="rUX" min="-1" max="1" step="0.01" value="0">
  <input type="number" id="nUX" step="0.01" value="0">

  <label>Eye y</label><input type="range" id="rEY" min="-3" max="3" step="0.01" value="1.25">
  <input type="number" id="nEY" step="0.01" value="1.25">

  <label>Up y</label><input type="range" id="rUY" min="-1" max="1" step="0.01" value="0">
  <input type="number" id="nUY" step="0.01" value="0">

  <label>Eye z</label><input type="range" id="rEZ" min="-3" max="3" step="0.01" value="1.25">
  <input type="number" id="nEZ" step="0.01" value="1.25">

  <label>Up z</label><input type="range" id="rUZ" min="-1" max="1" step="0.01" value="1">
  <input type="number" id="nUZ" step="0.01" value="1">
</div>
<div class="camctl-row">
  <button onclick="camSnap()">📋 Copy camera (JSON → textarea)</button>
  <button onclick="camApply()">📥 Apply pasted JSON</button>
  <button onclick="camReset()">↺ Reset</button>
  <textarea id="camJson" placeholder="Camera JSON updates live. Copy from one viewer, paste into the next, click Apply."></textarea>
</div>
<div id="plot"></div>
<script>
""")

        traces = []

        # Chamber mesh first (opaque, occludes back of view)
        if chamber_data is not None:
            cv = chamber_data["verts_mm"]
            cf = chamber_data["faces"]
            f.write("const chamberMesh={\n")
            f.write("  type:'mesh3d',name:'chambers + aorta (opaque)',\n")
            f.write(f"  x:{jfloats(cv[:,0], 2)},\n  y:{jfloats(cv[:,1], 2)},\n  z:{jfloats(cv[:,2], 2)},\n")
            f.write(f"  i:{jints(cf[:,0])},\n  j:{jints(cf[:,1])},\n  k:{jints(cf[:,2])},\n")
            f.write("  color:'#4a4a4a',opacity:1.0,flatshading:false,\n")
            # Flat/global lighting: high ambient so back faces don't go dark.
            # Plotly only supports one point light per scene; ambient is the
            # only way to brighten the side opposite the light.
            f.write("  lighting:{ambient:0.9,diffuse:0.35,specular:0.02,roughness:0.95,fresnel:0.05},\n")
            f.write("  lightposition:{x:0,y:0,z:1000000},\n")
            f.write("  showscale:false,hoverinfo:'skip'};\n")
            traces.append("chamberMesh")

        # Perfusion surface (the main change vs scatter version)
        f.write("const perfSurface={\n")
        f.write("  type:'mesh3d',name:'myocardium perfusion',\n")
        f.write(f"  x:{jfloats(verts_mm[:,0], 2)},\n  y:{jfloats(verts_mm[:,1], 2)},\n  z:{jfloats(verts_mm[:,2], 2)},\n")
        f.write(f"  i:{jints(faces[:,0])},\n  j:{jints(faces[:,1])},\n  k:{jints(faces[:,2])},\n")
        f.write(f"  intensity:{jfloats(perf_vals, 3)},\n")
        f.write("  intensitymode:'vertex',\n")
        f.write(f"  cmin:{cmin:.4f},cmax:{cmax:.4f},\n")
        # Jet-like colorscale (matches user's reference cortex image)
        f.write("  colorscale:[[0,'#00007f'],[0.1,'#0000ff'],[0.3,'#00ffff'],"
                "[0.5,'#7fff7f'],[0.7,'#ffff00'],[0.9,'#ff7f00'],[1,'#7f0000']],\n")
        # Colorbar text in white (was gray-on-dark; hard to read)
        f.write("  colorbar:{title:{text:'Perfusion (mL/min/g)',font:{color:'#ffffff'}},"
                "tickfont:{color:'#ffffff'},x:1.02,len:0.7},\n")
        f.write("  opacity:1.0,flatshading:false,\n")
        # Flat/global lighting on the perfusion surface (matches chamber).
        f.write("  lighting:{ambient:0.9,diffuse:0.35,specular:0.05,roughness:0.9,fresnel:0.05},\n")
        f.write("  lightposition:{x:0,y:0,z:1000000}};\n")
        traces.append("perfSurface")

        # Cylinder tree mesh (LAD/LCX/RCA) — same as scatter version
        tree_trace_indices = {}
        if tree_data is not None:
            starts = np.asarray(tree_data["starts_mm"], dtype=np.float64)
            ends = np.asarray(tree_data["ends_mm"], dtype=np.float64)
            tree_ids = np.asarray(tree_data["tree_id"], dtype=np.int32)
            diams = np.asarray(tree_data["diameter_um"], dtype=np.float64)
            min_d = float(tree_data["min_diam_um"])
            n_sides = int(args.tree_sides)
            r_scale = float(args.tree_radius_scale)
            COLORS = {0: "#1f77ff", 1: "#e3342f", 2: "#22aa44"}
            NAMES = {0: "LAD", 1: "LCX", 2: "RCA"}
            angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
            cos_a = np.cos(angles); sin_a = np.sin(angles)
            for tid in (0, 1, 2):
                m = tree_ids == tid
                if not m.any(): continue
                S = starts[m]; E = ends[m]
                R = (diams[m] / 2.0 / 1000.0) * r_scale
                N = len(S)
                vx = np.empty(N * 2 * n_sides, dtype=np.float32)
                vy = np.empty(N * 2 * n_sides, dtype=np.float32)
                vz = np.empty(N * 2 * n_sides, dtype=np.float32)
                fi = np.empty(N * 2 * n_sides, dtype=np.int32)
                fj = np.empty(N * 2 * n_sides, dtype=np.int32)
                fk = np.empty(N * 2 * n_sides, dtype=np.int32)
                vptr = 0; fptr = 0
                for s in range(N):
                    a = S[s]; b = E[s]; r = R[s]
                    axis = b - a; L = np.linalg.norm(axis)
                    if L < 1e-9 or r < 1e-9: continue
                    ah = axis / L
                    ref = np.array([0., 0., 1.]) if abs(ah[2]) < 0.9 else np.array([1., 0., 0.])
                    p1 = np.cross(ah, ref); p1 = p1 / np.linalg.norm(p1)
                    p2 = np.cross(ah, p1)
                    offsets = (cos_a[:, None] * p1 + sin_a[:, None] * p2) * r
                    sr = a[None, :] + offsets; er = b[None, :] + offsets
                    base = vptr
                    vx[base:base+n_sides] = sr[:, 0]; vy[base:base+n_sides] = sr[:, 1]; vz[base:base+n_sides] = sr[:, 2]
                    vx[base+n_sides:base+2*n_sides] = er[:, 0]; vy[base+n_sides:base+2*n_sides] = er[:, 1]; vz[base+n_sides:base+2*n_sides] = er[:, 2]
                    for k in range(n_sides):
                        kn = (k + 1) % n_sides
                        fi[fptr] = base + k; fj[fptr] = base + n_sides + k; fk[fptr] = base + n_sides + kn
                        fi[fptr+1] = base + k; fj[fptr+1] = base + n_sides + kn; fk[fptr+1] = base + kn
                        fptr += 2
                    vptr += 2 * n_sides
                vx = vx[:vptr]; vy = vy[:vptr]; vz = vz[:vptr]
                fi = fi[:fptr]; fj = fj[:fptr]; fk = fk[:fptr]
                f.write(f"const tree_{NAMES[tid]}={{\n")
                f.write("  type:'mesh3d',\n")
                f.write(f"  name:'{NAMES[tid]} (≥{min_d:.0f} μm)',\n")
                f.write(f"  x:{jfloats(vx, 2)},\n  y:{jfloats(vy, 2)},\n  z:{jfloats(vz, 2)},\n")
                f.write(f"  i:{jints(fi)},\n  j:{jints(fj)},\n  k:{jints(fk)},\n")
                f.write(f"  color:'{COLORS[tid]}',opacity:1.0,flatshading:false,\n")
                # Tree cylinders: a bit less ambient so they keep some shading
                # vs the (intentionally flat) myocardium surface — readability.
                f.write("  lighting:{ambient:0.8,diffuse:0.45,specular:0.1,roughness:0.7,fresnel:0.1},\n")
                f.write("  lightposition:{x:0,y:0,z:1000000},\n")
                f.write("  showscale:false,hoverinfo:'skip'};\n")
                tree_trace_indices[tid] = len(traces)
                traces.append(f"tree_{NAMES[tid]}")

        # Layout + toggle buttons
        f.write("const layout={paper_bgcolor:'#101015',\n")
        f.write("  scene:{aspectmode:'data',bgcolor:'#161620',\n")
        f.write("    xaxis:{title:{text:'x (mm)'},color:'#ccc',gridcolor:'#333',backgroundcolor:'#101015'},\n")
        f.write("    yaxis:{title:{text:'y (mm)'},color:'#ccc',gridcolor:'#333',backgroundcolor:'#101015'},\n")
        # zaxis reversed: recon-frame z increases caudally (toward the apex) but
        # anatomically the apex should display DOWN. Reversing the z-axis flips
        # the rendered heart so the apex points inferior, as a clinician expects.
        f.write("    zaxis:{title:{text:'z (mm)'},color:'#ccc',gridcolor:'#333',backgroundcolor:'#101015',autorange:'reversed'}},\n")
        f.write("  margin:{l:80,r:0,t:30,b:30},showlegend:false")
        if tree_trace_indices:
            COLORS = {0: "#1f77ff", 1: "#e3342f", 2: "#22aa44"}
            NAMES = {0: "LAD", 1: "LCX", 2: "RCA"}
            bspecs = []
            for tid in (0, 1, 2):
                if tid not in tree_trace_indices: continue
                idx = tree_trace_indices[tid]
                bspecs.append(
                    f"{{label:'Toggle {NAMES[tid]}',method:'restyle',"
                    f"args:[{{'visible':'legendonly'}},[{idx}]],"
                    f"args2:[{{'visible':true}},[{idx}]],"
                    f"bgcolor:'{COLORS[tid]}',font:{{color:'#ffffff'}}}}"
                )
            f.write(",\n  updatemenus:[{")
            f.write("type:'buttons',direction:'left',showactive:false,")
            f.write("x:0,xanchor:'left',y:1.05,yanchor:'bottom',")
            f.write("pad:{l:5,r:5,t:5,b:5},bgcolor:'#222',")
            f.write("buttons:[" + ",".join(bspecs) + "]}]")
        f.write("};\n")
        f.write(f"Plotly.newPlot('plot',[{','.join(traces)}],layout,{{responsive:true}});\n")
        # ───────────── Camera control wiring ─────────────
        # Three-way sync: sliders ↔ number inputs ↔ Plotly camera. When the
        # user drags the plot, the listener pushes the new camera into both
        # widget sets + the JSON textarea so it's always copy-paste-ready.
        f.write(r"""
const gd = document.getElementById('plot');
const PAIRS = [
  ['EX','eye','x'], ['EY','eye','y'], ['EZ','eye','z'],
  ['UX','up' ,'x'], ['UY','up' ,'y'], ['UZ','up' ,'z'],
];
const DEFAULT_CAM = {
  eye:    {x:1.25, y:1.25, z:1.25},
  up:     {x:0,    y:0,    z:1},
  center: {x:0,    y:0,    z:0},
};

function readWidgets() {
  // Build a camera object from the current widget values.
  const cam = {eye:{}, up:{}, center:{x:0,y:0,z:0}};
  for (const [id, group, axis] of PAIRS) {
    cam[group][axis] = parseFloat(document.getElementById('n'+id).value);
  }
  // Preserve current center if Plotly has one
  const cur = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene.camera;
  if (cur && cur.center) cam.center = {x:cur.center.x, y:cur.center.y, z:cur.center.z};
  return cam;
}

function writeWidgets(cam) {
  // Push Plotly camera values into the widget set + JSON textarea.
  for (const [id, group, axis] of PAIRS) {
    const v = cam[group][axis];
    const s = (typeof v === 'number' ? v.toFixed(3) : '0.000');
    document.getElementById('r'+id).value = s;
    document.getElementById('n'+id).value = s;
  }
  document.getElementById('camJson').value = JSON.stringify(cam, null, 0);
}

function applyCam(cam) {
  Plotly.relayout(gd, {'scene.camera': cam});
}

// Slider/number ↔ each other and ↔ Plotly camera
for (const [id, group, axis] of PAIRS) {
  const r = document.getElementById('r'+id);
  const n = document.getElementById('n'+id);
  const onChange = (src) => {
    const v = parseFloat(src.value);
    if (Number.isNaN(v)) return;
    r.value = v; n.value = v;
    applyCam(readWidgets());
  };
  r.addEventListener('input', () => onChange(r));
  n.addEventListener('input', () => onChange(n));
}

// When the user drags the plot, push the new camera state into the widgets.
gd.on('plotly_relayout', (ev) => {
  // Only react when the camera actually moved (avoid loops from our own relayout)
  const cam = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene.camera;
  if (cam) writeWidgets(cam);
});

// 📋 Copy: just refreshes the JSON textarea from current camera. Then user
// can ctrl+A / ctrl+C inside the textarea to copy.
function camSnap() {
  const cam = gd._fullLayout.scene.camera;
  writeWidgets(cam);
  const ta = document.getElementById('camJson');
  ta.select();
  try { document.execCommand('copy'); } catch (e) {}
}

// 📥 Apply: parse textarea JSON and apply to scene + widgets.
function camApply() {
  const txt = document.getElementById('camJson').value.trim();
  if (!txt) return;
  let cam;
  try { cam = JSON.parse(txt); }
  catch (e) { alert('Invalid JSON: ' + e.message); return; }
  if (!cam.eye || !cam.up) { alert('JSON must have eye and up objects'); return; }
  if (!cam.center) cam.center = {x:0, y:0, z:0};
  applyCam(cam);
  writeWidgets(cam);
}

// ↺ Reset to Plotly defaults
function camReset() {
  applyCam(DEFAULT_CAM);
  writeWidgets(DEFAULT_CAM);
}

// Initialize widgets from the first rendered camera (Plotly normalizes
// aspect-ratio-adjusted defaults).
setTimeout(() => {
  const cam = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene.camera;
  if (cam) writeWidgets(cam);
}, 80);
""")
        f.write("</script></body></html>\n")

    sz = os.path.getsize(args.out_html) / 1024**2
    print(f"[surface] wrote {args.out_html}  ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
