#!/usr/bin/env julia
# lad_stenosis_flowsim.jl — in-memory LAD proximal-stenosis sweep of the hemo
# LAD root flow. Loads the LAD tree once; for each stenosis %, applies the same
# cosine-taper narrowing as inject_lad_stenosis.py to the proximal 5 mm trunk,
# re-runs compute_hemodynamics, records LAD root flow. LCX/RCA unchanged.
import Pkg; Pkg.activate("/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/FlowContrastSim.jl")
using FlowContrastSim, Printf, LinearAlgebra

const TREE_DIR = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/VascularTreeSim.jl/output"
const STEN_LEN_MM = 5.0
const ROOT_P = 100.0*133.322; const TERM_P = 15.0*133.322; const HCT = 0.45
const MASS_LAD = 58.9
const PCTS = [0,10,20,30,40,50,60,70,80,90]

println("[flowsim] loading LAD tree …"); flush(stdout)
tree = load_tree("LAD", joinpath(TREE_DIR, "lad_segments.csv"))
@printf("[flowsim] LAD: %d segs\n", length(tree.segment_start)); flush(stdout)

seg_len(s) = norm(tree.vertices[tree.segment_end[s]] - tree.vertices[tree.segment_start[s]]) * 10.0  # cm→mm
# walk proximal trunk: largest-D child segment at each vertex, until cum length ≥ STEN_LEN_MM
function find_trunk(tree)
    trunk = Int[]; cur = tree.root_vertex; cum = 0.0
    while cum < STEN_LEN_MM
        kids = tree.children[cur]
        isempty(kids) && break
        segs = [tree.incoming_segment[ch] for ch in kids if tree.incoming_segment[ch] != 0]
        isempty(segs) && break
        s = segs[argmax([tree.segment_diameter_cm[x] for x in segs])]
        push!(trunk, s); cum += seg_len(s); cur = tree.segment_end[s]
    end
    spos = Float64[]; acc = 0.0
    for s in trunk
        L = seg_len(s); push!(spos, (acc + L/2)/max(cum,1e-9)); acc += L
    end
    return trunk, spos, cum
end
trunk, spos, cum = find_trunk(tree)
@printf("[flowsim] trunk: %d segs, %.2f mm; diam μm: %s\n", length(trunk), cum,
        join([@sprintf("%.0f", tree.segment_diameter_cm[s]*1e4) for s in trunk], ","))
orig_d = [tree.segment_diameter_cm[s] for s in trunk]

root_flow() = begin
    hemo = compute_hemodynamics(tree; root_pressure=ROOT_P, terminal_pressure=TERM_P,
        hematocrit=HCT, capillary_bed_R_per_100g_mmHgmin_ml=0.0, territory_mass_g=MASS_LAD)
    q = 0.0
    for cc in tree.children[tree.root_vertex]; sg=tree.incoming_segment[cc]; sg!=0 && (q+=hemo.segment_flow[sg]); end
    q*60e6  # mL/min
end

out = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline/output/lad_flowsim_sweep.csv"
open(out,"w") do io
    println(io, "stenosis_pct,Q_LAD_mlmin,Q_LAD_ratio")
    q0 = 0.0
    for pct in PCTS
        residual = 1.0 - pct/100.0
        for (k,s) in enumerate(trunk)
            bump = (1.0 - cos(2π*spos[k]))/2.0
            tree.segment_diameter_cm[s] = orig_d[k] * (1.0 - (1.0-residual)*bump)
        end
        q = root_flow()
        pct==0 && (q0 = q)
        @printf("[flowsim] %2d%% stenosis → Q_LAD = %.1f mL/min (ratio %.3f)\n", pct, q, q/q0); flush(stdout)
        @printf(io, "%d,%.3f,%.4f\n", pct, q, q/q0)
        for (k,s) in enumerate(trunk); tree.segment_diameter_cm[s] = orig_d[k]; end  # restore
    end
end
println("[flowsim] wrote $out")
