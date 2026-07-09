#!/usr/bin/env bash
# run_dynamic_perfusion.sh — honest dynamic-contrast CT perfusion pipeline.
#
# This is the ONLY supported contrast pipeline. It simulates iodine transport
# with a real advection–dispersion PDE through the grown coronary tree, then
# voxelizes the ACTUAL vessel geometry (emergent occupied-volume, no imposed
# MBV, no flow-proportional / Kety / ECV formula painting of the myocardium),
# renders each timepoint through BasisSimulator, and reconstructs it.
#
# Myocardial enhancement is whatever the capillary bed physically carries —
# it is measured from the recon downstream (see recover_perfusion.py), never
# painted from a target-perfusion formula.
#
# Stages, per frame time t:
#   1. FlowContrastSim  extract_peak_iodine_bae.jl (FRAMES mode)
#        → per-segment real PDE iodine  $OUT/iodine/frame_${t}s/
#   2. VascularTreeSim  apply_contrast_at_peak.jl
#        → emergent occupied-volume voxelization into UInt16 phantom  $OUT/ph_${t}s
#   3. VascularTreeSim  add_chambers_to_phantom.jl
#        → patch LV/LA/aorta blood pools with the real arterial input  $OUT/phc_${t}s
#   4. run_basis_sim    run_cta_sim_param.jl
#        → BasisSim scan + reconstruction  $OUT/recon_${t}s/recon_fbp_hu_f32.raw
#
# Usage:
#   ROOT=... XCAT=... TREE=... OUT=... FRAMES="00 12 15 18 21 24" \
#     bash scripts/run_dynamic_perfusion.sh
set -euo pipefail

ROOT="${ROOT:-/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation}"
XCAT="${XCAT:?set XCAT to the vmale50 UInt8 phantom raw}"
TREE="${TREE:?set TREE to the grown coronary tree segment-CSV dir}"
OUT="${OUT:?set OUT to the output directory}"
FRAMES="${FRAMES:-00 12 15 18 21 24}"
NSUB="${NSUB:-5}"          # sub-voxel MC samples per axis for resolvable vessels
KVP="${KVP:-100}"; MA="${MA:-250}"
GPUS="${GPUS:-0 2 3 4}"    # CUDA devices to round-robin the scans over

VTS="$ROOT/VascularTreeSim.jl"
FCS="$ROOT/FlowContrastSim.jl"
RBS="$ROOT/phantom_ct_input/run_basis_sim"
mkdir -p "$OUT"

echo "[1] real-PDE per-frame iodine (FlowContrastSim)"
FRAMES_CSV="$(echo "$FRAMES" | sed 's/ /,/g')"
FRAMES="$FRAMES_CSV" julia --project="$FCS" --threads=auto \
  "$FCS/scripts/extract_peak_iodine_bae.jl" "$TREE" "$OUT/iodine"

for f in $FRAMES; do
  echo "[2] voxelize real capillary contrast — frame_${f}s"
  julia --project="$VTS" --threads=auto "$VTS/scripts/apply_contrast_at_peak.jl" \
    "$TREE" "$XCAT" "$OUT/iodine/frame_${f}s" "$OUT/ph_${f}s" "$NSUB" > "$OUT/vox_${f}.log" 2>&1
  echo "[3] patch blood pools with real AIF — frame_${f}s"
  julia --project="$VTS" --threads=auto "$VTS/scripts/add_chambers_to_phantom.jl" \
    "$OUT/ph_${f}s" "$OUT/iodine/frame_${f}s" "$OUT/phc_${f}s" >> "$OUT/vox_${f}.log" 2>&1
  rm -f "$OUT/ph_${f}s"/*.raw    # free the intermediate myo-only phantom
done

echo "[4] BasisSim scan + recon (round-robin GPUs: $GPUS)"
gi=0; pids=""; ng=$(echo "$GPUS" | wc -w)
for f in $FRAMES; do
  g=$(echo "$GPUS" | cut -d' ' -f$((gi%ng+1)))
  CUDA_VISIBLE_DEVICES=$g julia --project="$RBS" "$RBS/run_cta_sim_param.jl" \
    "$OUT/phc_${f}s" "$OUT/recon_${f}s" --kvp "$KVP" --mA "$MA" --no-dicom > "$OUT/scan_${f}.log" 2>&1 &
  pids="$pids $!"; gi=$((gi+1))
  [ $((gi%ng)) -eq 0 ] && { wait $pids; pids=""; }
done
[ -n "$pids" ] && wait $pids
echo "[done] recons in $OUT/recon_*s/  — run recover_perfusion.py next"
