#!/usr/bin/env python3
"""
Side-by-side HTML figure showing the LAD proximal anatomy WITHOUT and WITH the
smooth-tapered 80% stenosis. Each panel is a Plotly Scene with:

  * Synthetic AORTA stub (dark gray cylinder, ~30 mm diameter)
  * Synthetic LEFT MAIN (dark gray short cylinder bridging aorta → LAD entry)
  * LAD trunk + early branches (blue cylinders, true per-segment diameter)

The LAD CSV in this project does not include a Left Main segment (the tree
starts at the LM/LAD bifurcation), so we synthesize the LM + aorta visually
purely for the figure — they are NOT part of the simulation. The stenosis
narrowing comes from inject_lad_stenosis.py (cosine taper, default 10 mm
length, peak D × 0.2).

Usage:
  python3 build_stenosis_comparison_figure.py  TREE_DIR_NORMAL  TREE_DIR_STENOSIS  OUT_HTML
        [--n-trunk-segs 40]   # how many segments along LAD to show
        [--lad-min-diam 800]  # only render LAD segments with D ≥ this
"""
import argparse
import os
import sys
import math
import numpy as np

# Coord transform constants — match extract_tree_overlay.py
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
SCALE_BS_TO_R = BS_VOXEL_MM / RECON_VOXEL_MM


def nrb_to_recon_mm(pts_cm):
    """pts_cm: (N, 3) → (N, 3) mm in viewer coord (same convention as
    other viewers in this pipeline)."""
    p = pts_cm + NRB_OFFSET
    p = p / PH_VOXEL_CM
    p[:, 1] = (PH_DIMS[1] - 1) - p[:, 1]
    p[:, 2] = (PH_DIMS[2] - 1) - p[:, 2]
    p = (p - 1.0) / DOWNSAMPLE
    p = (p - BS_CENTER) * SCALE_BS_TO_R + R_CENTER
    return (p * RECON_VOXEL_MM).astype(np.float32)


# ── Load LAD CSV and pull early branches ───────────────────────────────────
def load_lad_early(tree_dir, n_trunk_segs, render_min_diam_um, load_min_diam_um=100.0):
    """Return list of (start_mm, end_mm, diam_um, depth) for LAD segments
    within `n_trunk_segs` BFS levels of the root.

    Two-stage filtering:
      * load_min_diam_um  — what we admit into the in-memory topology (must be
        low enough that the stenosis narrowing doesn't break parent→child links).
        Default 100 μm captures everything except true capillaries.
      * render_min_diam_um — what we actually return for rendering. Trunk
        segments are ALWAYS kept (even if narrower than render threshold)
        because the stenosis lives on the trunk and we explicitly want to see
        it narrow.
    """
    csv = os.path.join(tree_dir, "lad_segments.csv")
    if not os.path.isfile(csv):
        raise FileNotFoundError(csv)

    # Pass 1: load topology (keep low threshold so stenosis links stay intact)
    by_sid = {}     # sid → (pid, x1,y1,z1,x2,y2,z2, len_mm, d_um)
    children = {}   # pid → [sid, ...]
    with open(csv) as f:
        hdr = f.readline().strip().split(",")
        i_sid = hdr.index("segment_id"); i_pid = hdr.index("parent_segment_id")
        i_x1 = hdr.index("x1_cm"); i_y1 = hdr.index("y1_cm"); i_z1 = hdr.index("z1_cm")
        i_x2 = hdr.index("x2_cm"); i_y2 = hdr.index("y2_cm"); i_z2 = hdr.index("z2_cm")
        i_L  = hdr.index("length_mm"); i_D = hdr.index("diameter_um")
        for line in f:
            cols = line.split(",")
            if len(cols) <= i_D: continue
            try:
                d = float(cols[i_D])
            except ValueError: continue
            if d < load_min_diam_um: continue
            try:
                sid = int(cols[i_sid]); pid = int(cols[i_pid])
                x1,y1,z1 = float(cols[i_x1]), float(cols[i_y1]), float(cols[i_z1])
                x2,y2,z2 = float(cols[i_x2]), float(cols[i_y2]), float(cols[i_z2])
                L = float(cols[i_L])
            except (ValueError, IndexError): continue
            by_sid[sid] = (pid, x1,y1,z1, x2,y2,z2, L, d)
            children.setdefault(pid, []).append(sid)

    # Find LAD root (pid = 0)
    roots = [s for s, v in by_sid.items() if v[0] == 0]
    if not roots:
        raise RuntimeError("no LAD root found (pid==0)")
    root = roots[0]

    # Identify main trunk: largest-D child at each fork, walking until depth
    # exceeds n_trunk_segs OR no children. These segments are render-protected.
    trunk_set = set()
    cur = root
    for _ in range(n_trunk_segs):
        trunk_set.add(cur)
        kids = children.get(cur, [])
        if not kids: break
        cur = max(kids, key=lambda s: by_sid[s][8])

    # BFS from root, keep first n_trunk_segs depths
    queue = [(root, 0)]
    visited = set()
    out = []
    while queue:
        sid, depth = queue.pop(0)
        if sid in visited: continue
        if depth >= n_trunk_segs: continue
        visited.add(sid)
        v = by_sid[sid]
        d = v[8]
        # Filter for render: keep trunk always, side branches only above threshold
        if sid in trunk_set or d >= render_min_diam_um:
            pts_cm = np.array([[v[1], v[2], v[3]], [v[4], v[5], v[6]]])
            pts_mm = nrb_to_recon_mm(pts_cm)
            out.append({
                "sid": sid, "start_mm": pts_mm[0], "end_mm": pts_mm[1],
                "diam_um": d, "depth": depth, "on_trunk": sid in trunk_set,
            })
        for c in children.get(sid, []):
            queue.append((c, depth + 1))
    return out, root, trunk_set


# ── Cylinder mesh builder (re-used) ────────────────────────────────────────
def cylinder_mesh(start, end, radius, n_sides=12):
    """Returns (verts, faces) for a cylindrical tube from start → end with given
    radius. Higher n_sides than the smooth viewer for nicer close-up."""
    a = np.asarray(start, dtype=np.float64); b = np.asarray(end, dtype=np.float64)
    axis = b - a; L = np.linalg.norm(axis)
    if L < 1e-9 or radius < 1e-9:
        return None
    ah = axis / L
    ref = np.array([0., 0., 1.]) if abs(ah[2]) < 0.9 else np.array([1., 0., 0.])
    p1 = np.cross(ah, ref); p1 = p1 / np.linalg.norm(p1)
    p2 = np.cross(ah, p1)
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    cos_a = np.cos(angles); sin_a = np.sin(angles)
    offsets = (cos_a[:, None] * p1 + sin_a[:, None] * p2) * radius
    sr = a[None, :] + offsets
    er = b[None, :] + offsets
    verts = np.vstack([sr, er])
    faces = []
    for k in range(n_sides):
        kn = (k + 1) % n_sides
        faces.append([k, n_sides + k, n_sides + kn])
        faces.append([k, n_sides + kn, kn])
    return verts.astype(np.float32), np.array(faces, dtype=np.int32)


def cap_mesh(center, axis_hat, radius, n_sides=12):
    """Disk cap centered at `center` with normal `axis_hat`. Used to cap the
    aorta stub's far end so it doesn't look hollow."""
    ref = np.array([0., 0., 1.]) if abs(axis_hat[2]) < 0.9 else np.array([1., 0., 0.])
    p1 = np.cross(axis_hat, ref); p1 = p1 / np.linalg.norm(p1)
    p2 = np.cross(axis_hat, p1)
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    rim = np.array(center)[None, :] + (np.cos(angles)[:, None] * p1 + np.sin(angles)[:, None] * p2) * radius
    verts = np.vstack([rim, np.array(center)[None, :]])  # last vertex = center
    center_idx = n_sides
    faces = [[k, (k+1) % n_sides, center_idx] for k in range(n_sides)]
    return verts.astype(np.float32), np.array(faces, dtype=np.int32)


# ── Per-trace mesh writer ──────────────────────────────────────────────────
def merge_meshes(meshes):
    """Concatenate (verts, faces) tuples into one mesh with re-based indices."""
    all_v = []; all_f = []
    base = 0
    for v, fc in meshes:
        all_v.append(v)
        all_f.append(fc + base)
        base += len(v)
    return np.concatenate(all_v, axis=0), np.concatenate(all_f, axis=0)


def jfloats(arr, ndp=3):
    return "[" + ",".join(f"{x:.{ndp}f}" for x in arr) + "]"


def jints(arr):
    return "[" + ",".join(str(int(x)) for x in arr) + "]"


def write_mesh_trace(f, var_name, verts, faces, color, opacity=1.0,
                     name=None, scene_id=None):
    f.write(f"const {var_name}={{\n")
    f.write("  type:'mesh3d',\n")
    if name: f.write(f"  name:'{name}',\n")
    if scene_id: f.write(f"  scene:'{scene_id}',\n")
    f.write(f"  x:{jfloats(verts[:,0], 2)},\n  y:{jfloats(verts[:,1], 2)},\n  z:{jfloats(verts[:,2], 2)},\n")
    f.write(f"  i:{jints(faces[:,0])},\n  j:{jints(faces[:,1])},\n  k:{jints(faces[:,2])},\n")
    f.write(f"  color:'{color}',opacity:{opacity:.2f},flatshading:false,\n")
    f.write("  lighting:{ambient:0.55,diffuse:0.7,specular:0.15,roughness:0.7,fresnel:0.1},\n")
    f.write("  lightposition:{x:1000,y:1000,z:1000},\n")
    f.write("  showscale:false,hoverinfo:'skip'};\n")


# ── Synthesize aorta + Left Main near LAD root ─────────────────────────────
def build_aorta_and_lm(lad_root_start_mm, lad_root_end_mm,
                       aorta_d_mm=30.0, aorta_len_mm=50.0,
                       lm_d_mm=5.0, lm_len_mm=10.0):
    """
    Constructs a synthetic aorta cylinder + Left Main segment.

    Aorta orientation: anatomically, ascending aorta is roughly perpendicular
    to the LAD proximal direction. We point the aorta along a direction
    perpendicular to LAD's first segment. Left Main bridges from the aortic
    side wall to the LAD entry.

    All in viewer mm coords. Returns:
      aorta_mesh = (verts, faces) cylinder + cap
      lm_mesh    = (verts, faces) cylinder
    """
    lad_dir = lad_root_end_mm - lad_root_start_mm
    lad_dir = lad_dir / np.linalg.norm(lad_dir)
    # Aorta direction: perpendicular to LAD, in the +Z (sup→inf in recon frame
    # after reverse) and X-Y plane. Pick the largest perpendicular component.
    ref = np.array([0., 0., 1.])
    aorta_dir = np.cross(np.cross(lad_dir, ref), lad_dir)
    if np.linalg.norm(aorta_dir) < 1e-6:
        aorta_dir = np.cross(np.cross(lad_dir, np.array([1., 0., 0.])), lad_dir)
    aorta_dir = aorta_dir / np.linalg.norm(aorta_dir)

    # Place aorta so its side wall touches the LAD entry point at the LM stub.
    # LM stub: from LAD root, going aorta-ward, length lm_len_mm.
    lm_start = lad_root_start_mm.astype(np.float64)
    lm_end   = lm_start - aorta_dir * lm_len_mm  # going into the aorta
    # Aorta centerline: passes through lm_end, perpendicular to LM axis (=aorta_dir).
    # Pick an aorta axis perpendicular to LM direction. Use lad_dir cross-with
    # aorta_dir (which is perpendicular to aorta_dir already so cross gives a
    # vector orthogonal to both).
    aorta_axis = np.cross(aorta_dir, lad_dir)
    if np.linalg.norm(aorta_axis) < 1e-6:
        aorta_axis = np.array([0., 0., 1.])
    aorta_axis = aorta_axis / np.linalg.norm(aorta_axis)

    # Aorta centerline offset INWARD by aorta radius so its wall touches the
    # LM end point
    aorta_center_mid = lm_end + aorta_dir * (aorta_d_mm / 2.0)
    aorta_start = aorta_center_mid - aorta_axis * (aorta_len_mm / 2.0)
    aorta_end   = aorta_center_mid + aorta_axis * (aorta_len_mm / 2.0)

    aorta_v, aorta_f = cylinder_mesh(aorta_start, aorta_end, aorta_d_mm / 2.0, n_sides=24)
    # Caps on both ends
    cap1_v, cap1_f = cap_mesh(aorta_start, -aorta_axis, aorta_d_mm / 2.0, n_sides=24)
    cap2_v, cap2_f = cap_mesh(aorta_end,    aorta_axis, aorta_d_mm / 2.0, n_sides=24)
    aorta_mesh = merge_meshes([(aorta_v, aorta_f), (cap1_v, cap1_f), (cap2_v, cap2_f)])

    lm_v, lm_f = cylinder_mesh(lm_start, lm_end, lm_d_mm / 2.0, n_sides=18)
    lm_mesh = (lm_v, lm_f)
    return aorta_mesh, lm_mesh


def build_lad_mesh(segs):
    """Concatenate cylinder meshes for every LAD segment."""
    meshes = []
    for s in segs:
        m = cylinder_mesh(s["start_mm"], s["end_mm"], s["diam_um"] / 2.0 / 1000.0, n_sides=14)
        if m is not None:
            meshes.append(m)
    if not meshes:
        return None
    return merge_meshes(meshes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree_dir_normal")
    ap.add_argument("tree_dir_stenosis")
    ap.add_argument("out_html")
    ap.add_argument("--n-trunk-segs", type=int, default=40,
                    help="BFS depth from LAD root to render")
    ap.add_argument("--lad-min-diam", type=float, default=500.0,
                    help="render side branches with D ≥ this (μm); main trunk is "
                         "always rendered regardless of diameter (so the stenosis "
                         "narrowing remains visible)")
    args = ap.parse_args()

    print(f"[fig] loading normal LAD from {args.tree_dir_normal} …")
    segs_n, root_n, trunk_n = load_lad_early(args.tree_dir_normal, args.n_trunk_segs, args.lad_min_diam)
    print(f"[fig]   {len(segs_n)} segs ({sum(1 for s in segs_n if s['on_trunk'])} on trunk), depth ≤ {args.n_trunk_segs}")
    print(f"[fig]   diameter range [{min(s['diam_um'] for s in segs_n):.0f}, "
          f"{max(s['diam_um'] for s in segs_n):.0f}] μm")

    print(f"[fig] loading stenosed LAD from {args.tree_dir_stenosis} …")
    segs_s, root_s, trunk_s = load_lad_early(args.tree_dir_stenosis, args.n_trunk_segs, args.lad_min_diam)
    print(f"[fig]   {len(segs_s)} segs ({sum(1 for s in segs_s if s['on_trunk'])} on trunk), depth ≤ {args.n_trunk_segs}")
    print(f"[fig]   diameter range [{min(s['diam_um'] for s in segs_s):.0f}, "
          f"{max(s['diam_um'] for s in segs_s):.0f}] μm")
    trunk_diams = [s['diam_um'] for s in segs_s if s['on_trunk']]
    print(f"[fig]   stenosis trunk diameters (μm): " + ", ".join(f"{d:.0f}" for d in trunk_diams))

    # Build LAD meshes
    lad_n = build_lad_mesh(segs_n)
    lad_s = build_lad_mesh(segs_s)
    print(f"[fig] LAD mesh sizes: normal {len(lad_n[0])} verts, stenosis {len(lad_s[0])} verts")

    # Build Aorta + Left Main (use normal LAD root for placement; same for both panels)
    root_seg = segs_n[0]
    aorta_mesh, lm_mesh = build_aorta_and_lm(root_seg["start_mm"], root_seg["end_mm"])

    # Compute centered view box (use the LAD mesh bounds + a bit of padding)
    all_v = np.concatenate([lad_n[0], lad_s[0], aorta_mesh[0], lm_mesh[0]], axis=0)
    cx, cy, cz = all_v.mean(axis=0)
    span = np.max(all_v.max(axis=0) - all_v.min(axis=0))
    half = span * 0.7

    # Write HTML
    with open(args.out_html, "w") as f:
        f.write("<!DOCTYPE html><html><head><meta charset=\"utf-8\">\n")
        f.write("<title>LAD proximal stenosis — anatomical comparison</title>\n")
        f.write("<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>\n")
        f.write("<style>body{font-family:system-ui,sans-serif;margin:0;padding:8px;"
                "background:#101015;color:#eee}#plot{height:92vh}"
                "h2{margin:4px 0 6px}p{margin:4px 0;color:#aaa;font-size:12px}</style>\n")
        f.write("</head><body>\n")
        f.write("<h2>LAD proximal stenosis — anatomical comparison (cosine taper, peak D × 0.2)</h2>\n")
        f.write("<p>Left panel: normal anatomy. Right panel: 80 % peak diameter reduction over a "
                "10 mm cosine-tapered length on the LAD proximal trunk. <b>Aorta + Left Main are "
                "synthesized visually for the figure</b>; the simulation uses only the LAD/LCX/RCA "
                "trees (which start at the LM/LAD bifurcation). LAD in blue, Aorta + Left Main "
                "in dark gray. Drag in either scene to rotate (linked camera).</p>\n")
        f.write("<div id=\"plot\"></div>\n<script>\n")

        # Panel A (normal) traces
        write_mesh_trace(f, "aortaA",   *aorta_mesh, "#3a3a3a", name="Aorta",      scene_id="scene")
        write_mesh_trace(f, "lmA",      *lm_mesh,    "#5a5a5a", name="Left Main",  scene_id="scene")
        write_mesh_trace(f, "ladA",     *lad_n,      "#1f77ff", name="LAD normal", scene_id="scene")

        # Panel B (stenosis) traces
        write_mesh_trace(f, "aortaB",   *aorta_mesh, "#3a3a3a", name="Aorta",       scene_id="scene2")
        write_mesh_trace(f, "lmB",      *lm_mesh,    "#5a5a5a", name="Left Main",   scene_id="scene2")
        write_mesh_trace(f, "ladB",     *lad_s,      "#1f77ff", name="LAD stenosis",scene_id="scene2")

        # Layout — two scenes
        def scene_str(idx, title):
            x_dom = "[0,0.49]" if idx == 1 else "[0.51,1.0]"
            scene_name = "scene" if idx == 1 else "scene2"
            return (
                f"  {scene_name}:{{\n"
                f"    domain:{{x:{x_dom},y:[0,1]}},\n"
                "    aspectmode:'data',bgcolor:'#161620',\n"
                f"    xaxis:{{title:{{text:'x (mm)'}},color:'#ccc',gridcolor:'#333',backgroundcolor:'#101015',"
                f"range:[{cx-half:.2f},{cx+half:.2f}]}},\n"
                f"    yaxis:{{title:{{text:'y (mm)'}},color:'#ccc',gridcolor:'#333',backgroundcolor:'#101015',"
                f"range:[{cy-half:.2f},{cy+half:.2f}]}},\n"
                f"    zaxis:{{title:{{text:'z (mm)'}},color:'#ccc',gridcolor:'#333',backgroundcolor:'#101015',"
                f"range:[{cz-half:.2f},{cz+half:.2f}]}},\n"
                "    annotations:[{showarrow:false,x:0.5,y:1.05,xref:'paper',yref:'paper',"
                f"text:'{title}',font:{{color:'#fff',size:14}}}}]\n"
                "  }"
            )

        f.write("const layout={paper_bgcolor:'#101015',\n")
        f.write(scene_str(1, "Normal LAD origin") + ",\n")
        f.write(scene_str(2, "80 % peak stenosis (cosine taper, 10 mm)") + ",\n")
        f.write("  margin:{l:0,r:0,t:50,b:0},showlegend:false,\n")
        f.write("  annotations:[\n")
        f.write("    {showarrow:false,x:0.25,y:1.02,xref:'paper',yref:'paper',"
                "text:'Normal',font:{color:'#fff',size:18,family:'system-ui',weight:600}},\n")
        f.write("    {showarrow:false,x:0.75,y:1.02,xref:'paper',yref:'paper',"
                "text:'80% LAD proximal stenosis (smooth taper)',"
                "font:{color:'#ff6060',size:18,family:'system-ui',weight:600}}\n")
        f.write("  ]\n")
        f.write("};\n")
        f.write("const data=[aortaA,lmA,ladA, aortaB,lmB,ladB];\n")
        f.write("Plotly.newPlot('plot',data,layout,{responsive:true});\n")
        f.write("</script></body></html>\n")

    sz = os.path.getsize(args.out_html) / 1024**2
    print(f"[fig] wrote {args.out_html}  ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
