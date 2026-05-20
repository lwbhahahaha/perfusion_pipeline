#!/usr/bin/env python3
"""
Inject a focal stenosis at the LAD proximal trunk by narrowing the diameter
of the first ~STENOSIS_LEN_MM of the main chain to a target percentage of
the original.

Convention: clinical "X% stenosis" means X% diameter reduction (residual
diameter = (1 − X/100) × original). For Poiseuille R ∝ 1/D⁴, 80% reduction
amplifies that segment's resistance by 1/0.2⁴ = 625×, producing a
hemodynamically critical lesion.

Pipeline:
  1. Stream the LAD CSV, keeping only ≥ MIN_DIAM_UM segments in memory
     (~thousands of segments — enough to find the main trunk).
  2. Build parent → children map.
  3. Walk the main trunk from root (largest-diameter child at each
     bifurcation) until cumulative length ≥ STENOSIS_LEN_MM.
  4. Stream the LAD CSV again, rewriting the diameter column for any segment
     whose ID is in the trunk set. Everything else is copied through unchanged.
  5. LCX and RCA CSVs are copied verbatim.

Usage:
  python3 inject_lad_stenosis.py  IN_TREE_DIR  OUT_TREE_DIR
        [--stenosis-pct 80]   [--stenosis-len-mm 5]   [--min-diam-um 1000]
"""
import argparse
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_tree_dir")
    ap.add_argument("out_tree_dir")
    ap.add_argument("--stenosis-pct", type=float, default=80.0,
                    help="diameter reduction percent (80 means residual D = 20%% of original)")
    ap.add_argument("--stenosis-len-mm", type=float, default=5.0,
                    help="length of LAD trunk to narrow (mm), measured by cumulative segment length")
    ap.add_argument("--min-diam-um", type=float, default=1000.0,
                    help="only consider segments with D ≥ this when walking the main trunk")
    args = ap.parse_args()

    assert 0 < args.stenosis_pct < 100, "stenosis-pct must be in (0, 100)"
    residual_factor = (1.0 - args.stenosis_pct / 100.0)
    print(f"[stenosis] residual D factor = {residual_factor:.3f}  (R amplification ≈ {1/residual_factor**4:.0f}×)")

    os.makedirs(args.out_tree_dir, exist_ok=True)
    lad_in  = os.path.join(args.in_tree_dir, "lad_segments.csv")
    lad_out = os.path.join(args.out_tree_dir, "lad_segments.csv")
    if not os.path.isfile(lad_in):
        print(f"ERROR: {lad_in} not found", file=sys.stderr); sys.exit(1)

    # ── Pass 1: stream and keep ≥ min_diam_um segments only ──
    print(f"[stenosis] pass 1: load LAD segments with D ≥ {args.min_diam_um} μm …")
    keep = {}  # sid → (pid, length_mm, diam_um)
    with open(lad_in) as f:
        header = f.readline().strip().split(",")
        i_sid = header.index("segment_id")
        i_pid = header.index("parent_segment_id")
        i_L   = header.index("length_mm")
        i_D   = header.index("diameter_um")
        for line in f:
            cols = line.strip().split(",")
            if len(cols) <= i_D: continue
            try:
                d = float(cols[i_D])
            except ValueError:
                continue
            if d < args.min_diam_um: continue
            try:
                sid = int(cols[i_sid])
                pid = int(cols[i_pid])
                L   = float(cols[i_L])
            except (ValueError, IndexError):
                continue
            keep[sid] = (pid, L, d)
    print(f"[stenosis]   loaded {len(keep)} segments ≥ {args.min_diam_um} μm")

    # ── Find root: parent_segment_id == 0 ──
    roots = [sid for sid, (pid, _, _) in keep.items() if pid == 0]
    if not roots:
        # Fallback: smallest pid not in keep dict (orphan) is root candidate
        print("[stenosis]   no segment has pid=0; falling back to orphan detection")
        roots = [sid for sid, (pid, _, _) in keep.items() if pid not in keep]
    if not roots:
        print("ERROR: cannot find LAD root", file=sys.stderr); sys.exit(1)
    root = roots[0]
    print(f"[stenosis]   LAD root = segment {root}, D = {keep[root][2]:.1f} μm")

    # ── Build children map and walk main trunk ──
    children = {}
    for sid, (pid, _, _) in keep.items():
        children.setdefault(pid, []).append(sid)

    trunk = [root]
    cur = root
    cum_len_mm = keep[root][1]
    print(f"[stenosis] walking main trunk (largest-D child at each fork) until ≥ {args.stenosis_len_mm} mm cumulative length …")
    while cum_len_mm < args.stenosis_len_mm:
        kids = children.get(cur, [])
        if not kids:
            print(f"[stenosis]   reached leaf at segment {cur}; stopping")
            break
        # Pick the largest-D child
        next_sid = max(kids, key=lambda s: keep[s][2])
        cum_len_mm += keep[next_sid][1]
        trunk.append(next_sid)
        cur = next_sid
    print(f"[stenosis]   trunk: {len(trunk)} segments, cumulative length = {cum_len_mm:.3f} mm")
    print(f"[stenosis]   diameters along trunk (μm): " + ", ".join(f"{keep[s][2]:.0f}" for s in trunk))

    trunk_set = set(trunk)

    # ── Pass 2: stream LAD CSV and rewrite trunk segments' diameter ──
    print(f"[stenosis] pass 2: stream LAD CSV → {lad_out} (narrow {len(trunk_set)} trunk segs)")
    n_written = 0
    n_narrowed = 0
    with open(lad_in) as fin, open(lad_out, "w") as fout:
        header_line = fin.readline()
        fout.write(header_line)
        cols_hdr = header_line.strip().split(",")
        i_sid = cols_hdr.index("segment_id")
        i_D   = cols_hdr.index("diameter_um")
        for line in fin:
            cols = line.rstrip("\n").split(",")
            try:
                sid = int(cols[i_sid])
            except (ValueError, IndexError):
                fout.write(line); n_written += 1; continue
            if sid in trunk_set:
                try:
                    d = float(cols[i_D])
                    cols[i_D] = f"{d * residual_factor:.4f}"
                    n_narrowed += 1
                except (ValueError, IndexError):
                    pass
            fout.write(",".join(cols) + "\n")
            n_written += 1
    print(f"[stenosis]   wrote {n_written} rows, narrowed {n_narrowed} trunk segs")

    # ── Copy LCX, RCA unchanged ──
    for name in ("lcx", "rca"):
        src = os.path.join(args.in_tree_dir, f"{name}_segments.csv")
        dst = os.path.join(args.out_tree_dir, f"{name}_segments.csv")
        if os.path.isfile(src):
            if os.path.exists(dst):
                print(f"[stenosis] {dst} already exists, skipping copy")
            else:
                print(f"[stenosis] copying {src} → {dst}")
                shutil.copy(src, dst)

    print("[stenosis] DONE")


if __name__ == "__main__":
    main()
