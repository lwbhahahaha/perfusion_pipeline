#!/usr/bin/env julia
# Step 2 (KDTree variant): build per-myocardium-voxel territory map.
#
# For every myocardium voxel finds the nearest terminal-segment midpoint
# across all 3 trees via NearestNeighbors.KDTree (much faster than the
# ring-expansion grid: O(log N) per query vs O(K^3) for far voxels).
#
# Per-terminal midpoint approximation: each terminal capillary segment is
# only ~10-100 μm long (recursive subdivision down to 6 μm × ld_ratio),
# so midpoint vs endpoint differs by ≤ ~half-segment ≈ 50 μm — well below
# the 200 μm phantom voxel size.
#
# Output:
#   intermediate/myo_arrival.raw  (Float32, 1600x1400x500, sec; NaN = non-myo)
#   intermediate/myo_tree_id.raw  (UInt8,   0=non-myo / 1=LAD / 2=LCX / 3=RCA)
#   intermediate/myo_dist.raw     (Float32, distance to nearest terminal, cm)

using LinearAlgebra
using StaticArrays
using Statistics
using NearestNeighbors

const FCS_PATH = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/FlowContrastSim.jl"
using Pkg
Pkg.activate(FCS_PATH)
using FlowContrastSim

const PHANTOM_PATH = "/home/molloi-lab/smb_mount/shared_drive/Shu Nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"
# Tree dir can be overridden via env var STEP2_TREE_DIR (e.g. .../output_at_rest).
const TREE_DIR = get(ENV, "STEP2_TREE_DIR",
                     "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/VascularTreeSim.jl/output")
const OUT_DIR  = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline/intermediate"
const PHANTOM_DIMS = (1600, 1400, 500)
const VOXEL_SIZE_CM = 0.02
const NRB_TO_PHANTOM_OFFSET = SVector(2.1443, -9.5553, -20.0068)
const ROOT_P_PA = 100.0 * 133.322
const TERM_P_PA = 15.0  * 133.322
const HCT       = 0.45

# Pries-Secomb in-vivo capillary-bed + venous-return R (mmHg·min/mL/100g).
# Set to 0 because adding R uniformly across all leaves amplifies
# pathological zero-flow leaves into infinite-arrival outliers (median
# arrival jumped 1.08 → 75 s with R=0.15). Physiologically the missing
# distal R should be co-applied with autoregulation, which our tree
# lacks; we instead simulate autoregulation by overriding per-voxel
# arrival to a uniform physiological value in step 3 (see ARRIVAL_OVERRIDE).
const CAP_BED_R_PER_100G = 0.0

# Per-tree myocardium territory mass (g), measured from previous step 2 run.
# Used for the parallel→per-leaf R conversion. Re-running step 2 → these
# numbers will be refined and can be fed back as a second-pass.
const TERRITORY_MASS_G = Dict("LAD" => 58.9, "LCX" => 60.9, "RCA" => 63.8)

# (name, tree_id_uint8) — tree_id 0 reserved for non-myo
const TREES = [("LAD", UInt8(1)), ("LCX", UInt8(2)), ("RCA", UInt8(3))]

# ── Load phantom + extract myo voxels (in tree NRB cm) ──
println("[step2] Loading vmale50_act_1.raw …"); flush(stdout)
phantom = Array{UInt8}(undef, PHANTOM_DIMS)
read!(PHANTOM_PATH, phantom)

myo_lin = Int[]
sizehint!(myo_lin, 25_000_000)
@inbounds for k in 1:PHANTOM_DIMS[3], j in 1:PHANTOM_DIMS[2], i in 1:PHANTOM_DIMS[1]
    L = phantom[i, j, k]
    if 15 <= L <= 18
        push!(myo_lin, i + (j-1)*PHANTOM_DIMS[1] + (k-1)*PHANTOM_DIMS[1]*PHANTOM_DIMS[2])
    end
end
n_myo = length(myo_lin)
println("[step2]   myocardium voxels: $(n_myo) ($(round(100*n_myo/length(phantom); digits=2))%)"); flush(stdout)
phantom = nothing
GC.gc()

# Pre-compute myo voxel positions in NRB cm = phantom cm − offset
myo_pts = Matrix{Float64}(undef, 3, n_myo)  # column-major (3, N) for KDTree query
nx, ny, _ = PHANTOM_DIMS
@inbounds Threads.@threads for q in 1:n_myo
    lin = myo_lin[q] - 1
    i = lin % nx + 1
    rest = lin ÷ nx
    j = rest % ny + 1
    k = rest ÷ ny + 1
    myo_pts[1, q] = (i - 0.5) * VOXEL_SIZE_CM - NRB_TO_PHANTOM_OFFSET[1]
    myo_pts[2, q] = (j - 0.5) * VOXEL_SIZE_CM - NRB_TO_PHANTOM_OFFSET[2]
    myo_pts[3, q] = (k - 0.5) * VOXEL_SIZE_CM - NRB_TO_PHANTOM_OFFSET[3]
end

best_dist    = fill(Float32(Inf), n_myo)
best_arrival = fill(Float32(NaN), n_myo)
best_tree_id = zeros(UInt8, n_myo)

# ── Process each tree ──
for (name, tid) in TREES
    csv = joinpath(TREE_DIR, "$(lowercase(name))_segments.csv")
    isfile(csv) || error("CSV missing: $csv")

    println("\n" * "="^70)
    println("[step2] [$name] Loading tree from $(basename(csv)) …"); flush(stdout)
    t0 = time()
    tree = load_tree(name, csv)
    println("[step2] [$name]   loaded in $(round(time()-t0; digits=1))s: " *
            "$(length(tree.segment_start)) segs, $(length(tree.vertices)) verts"); flush(stdout)

    println("[step2] [$name] Computing hemodynamics (Pries+Poiseuille + Pries-Secomb cap R) …"); flush(stdout)
    t_mass = TERRITORY_MASS_G[name]
    println("[step2] [$name]   capillary_bed_R = $(CAP_BED_R_PER_100G) mmHg·min/mL/100g, territory_mass = $(t_mass) g")
    t1 = time()
    hemo = compute_hemodynamics(tree;
        root_pressure=ROOT_P_PA, terminal_pressure=TERM_P_PA, hematocrit=HCT,
        capillary_bed_R_per_100g_mmHgmin_ml=CAP_BED_R_PER_100G,
        territory_mass_g=t_mass)
    # Diagnostic: report root flow with cap R applied
    root_flow_m3s = 0.0
    @inbounds for c in tree.children[tree.root_vertex]
        s = tree.incoming_segment[c]
        s != 0 && (root_flow_m3s += hemo.segment_flow[s])
    end
    root_flow_mlmin = root_flow_m3s * 60e6
    println("[step2] [$name]   root flow = $(round(root_flow_mlmin; digits=1)) mL/min  (FCS no-cap-R was ~524/272/378; clinical hyperemic target ~242/116/214)")
    println("[step2] [$name]   hemo in $(round(time()-t1; digits=1))s"); flush(stdout)

    println("[step2] [$name] Computing arrival times …"); flush(stdout)
    t2 = time()
    arrivals_raw = FlowContrastSim._compute_arrival_times(tree, hemo)
    finite_arr = filter(isfinite, arrivals_raw)
    if !isempty(finite_arr)
        println("[step2] [$name]   arrivals_raw: median=$(round(median(finite_arr); digits=2))s p95=$(round(quantile(finite_arr, 0.95); digits=2))s max=$(round(maximum(finite_arr); digits=2))s")
    end
    println("[step2] [$name]   $(round(time()-t2; digits=1))s"); flush(stdout)

    # ── Extract terminal midpoints + arrivals (free tree afterwards) ──
    println("[step2] [$name] Extracting terminal midpoints …"); flush(stdout)
    t3 = time()
    nseg = length(tree.segment_start)
    terminal_idx = Int[]
    sizehint!(terminal_idx, nseg ÷ 2 + 1)
    @inbounds for s in 1:nseg
        ev = tree.segment_end[s]
        isempty(tree.children[ev]) && push!(terminal_idx, s)
    end
    n_term = length(terminal_idx)
    midpoints = Matrix{Float64}(undef, 3, n_term)   # KDTree expects (D, N)
    arrivals  = Vector{Float64}(undef, n_term)
    @inbounds for (k, s) in enumerate(terminal_idx)
        a = tree.vertices[tree.segment_start[s]]
        b = tree.vertices[tree.segment_end[s]]
        midpoints[1, k] = 0.5 * (a[1] + b[1])
        midpoints[2, k] = 0.5 * (a[2] + b[2])
        midpoints[3, k] = 0.5 * (a[3] + b[3])
        arrivals[k] = isfinite(arrivals_raw[s]) ? arrivals_raw[s] : Inf
    end
    println("[step2] [$name]   $(n_term) terminals extracted in $(round(time()-t3; digits=1))s")
    flush(stdout)

    # Free the heavy tree & arrivals_raw before the KDTree build
    tree = nothing
    arrivals_raw = nothing
    hemo = nothing
    GC.gc()

    # ── Build KDTree on midpoints, query myo voxels ──
    println("[step2] [$name] Building KDTree …"); flush(stdout)
    t4 = time()
    kd = KDTree(midpoints; leafsize=20)
    println("[step2] [$name]   KDTree built in $(round(time()-t4; digits=1))s")
    flush(stdout)

    println("[step2] [$name] Querying $(n_myo) voxels …"); flush(stdout)
    t5 = time()
    @inbounds Threads.@threads for q in 1:n_myo
        p = SVector(myo_pts[1, q], myo_pts[2, q], myo_pts[3, q])
        idx, dist = nn(kd, p)
        df = Float32(dist)
        if df < best_dist[q]
            best_dist[q] = df
            best_arrival[q] = Float32(arrivals[idx])
            best_tree_id[q] = tid
        end
    end
    println("[step2] [$name]   query done in $(round(time()-t5; digits=1))s")
    flush(stdout)

    # Free this tree's data before next
    kd = nothing
    midpoints = nothing
    arrivals = nothing
    GC.gc()
end

# ── Write results back into full-phantom-shaped arrays ──
println("\n[step2] Writing per-voxel arrays …"); flush(stdout)
arrival_full = fill(Float32(NaN), PHANTOM_DIMS)
tree_id_full = zeros(UInt8, PHANTOM_DIMS)
dist_full    = fill(Float32(NaN), PHANTOM_DIMS)
@inbounds for q in 1:n_myo
    lin = myo_lin[q]
    arrival_full[lin] = best_arrival[q]
    tree_id_full[lin] = best_tree_id[q]
    dist_full[lin]    = best_dist[q]
end

open(joinpath(OUT_DIR, "myo_arrival.raw"), "w") do io
    write(io, arrival_full)
end
open(joinpath(OUT_DIR, "myo_tree_id.raw"), "w") do io
    write(io, tree_id_full)
end
open(joinpath(OUT_DIR, "myo_dist.raw"), "w") do io
    write(io, dist_full)
end

# Stats
finite_arr  = filter(isfinite, vec(arrival_full))
finite_dist = filter(isfinite, vec(dist_full))
println("[step2] arrival times (s): median=$(round(median(finite_arr); digits=2))  p95=$(round(quantile(finite_arr, 0.95); digits=2))  max=$(round(maximum(finite_arr); digits=2))")
println("[step2] terminal distance (cm): median=$(round(median(finite_dist); digits=4))  p95=$(round(quantile(finite_dist, 0.95); digits=4))  max=$(round(maximum(finite_dist); digits=4))")
for (name, tid) in TREES
    n_owned = count(==(tid), tree_id_full)
    println("[step2]   $(name) territory: $(n_owned) voxels ($(round(100*n_owned/n_myo; digits=1))%)")
end
flush(stdout)
println("\n[step2] Done.")
