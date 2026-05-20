#!/usr/bin/env julia
# Step 3 (AIF variant): voxelize contrast-enhanced frames using a REAL patient
# AIF curve (from step0_prepare_aif.py) instead of the synthetic clinical gamma.
#
# For each frame time t = 0, 1, …, 20 s, copies the baseline HU and adds
# ΔHU = HU_PER_MG_ML · C_seg(t) to:
#
#   • LV blood pool (label 19), LA blood pool (label 21),
#     aorta lumen (mask from step 1), main coronary trunk (label 26)
#       → C_AIF(t) (undispersed; this is the aorta input function the bolus
#                   tracker sees)
#   • myocardium (labels 15–18)
#       → C_AIF(t, arrival[i,j,k]) × MYO_ECV   (dispersed, ECV-scaled)
#
# The AIF curve is loaded from a CSV produced by step0_prepare_aif.py:
#
#   # ... metadata comment lines ...
#   time_s,C_mg_per_mL
#   0.000,0.0000
#   0.275,0.0743
#   ...
#
# Dispersion model (same time-stretch as the synthetic-gamma version): the
# bolus is sampled at  t_eff = (t - arrival) / disp_factor  and the amplitude
# is divided by disp_factor (area-preserving). For a real AIF without an
# analytical t0, we use t_eff measured from t=0 of the curve.
#
# Output:
#   frames/frame_{tt:02d}.raw   (Int16, 1600x1400x500, 21 files, 47 GB total)
#
# Usage:
#   julia --threads=auto scripts/step3_voxelize_frames_aif.jl  AIF_CSV

using TOML

const PHANTOM_PATH = "/home/molloi-lab/smb_mount/shared_drive/Shu Nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"
const PIPELINE_DIR = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline"
const INTER_DIR = joinpath(PIPELINE_DIR, "intermediate")
const FRAMES_DIR = joinpath(PIPELINE_DIR, "frames")

const PHANTOM_DIMS = (1600, 1400, 500)
const N_VOX = prod(PHANTOM_DIMS)

# ── Read metadata from step 1 ──
const META = TOML.parsefile(joinpath(INTER_DIR, "metadata.toml"))
const HU_PER_MG_ML = Float64(META["hu_per_mg_ml_iodine"])
const DT_S      = Float64(META["frames"]["dt_s"])
const T_END_S   = Float64(META["frames"]["t_end_s"])
const N_FRAMES  = Int(META["frames"]["n_frames"])
const T_DISP_S  = 3.0  # capillary dispersion broadening time scale (s)

# Myocardium extracellular volume fraction (iodine distribution space).
# Pries-Secomb / clinical CT-MRI ECV: 0.25 ± 0.04 normal myocardium.
const MYO_ECV = 0.25

# Uniform myocardial arrival time (s). Healthy myocardium has tight arteriolar
# autoregulation → near-uniform capillary arrival ≈ 1–2 s. Override per-voxel
# arrival to a uniform physiological value to model autoregulation
# (set < 0 to disable and use per-voxel arrival from step2).
const ARRIVAL_OVERRIDE_S = 1.5

# ── AIF CSV loader ──
struct AIFCurve
    t::Vector{Float64}   # seconds
    C::Vector{Float64}   # mg/mL
end

function load_aif_csv(path::String)
    ts = Float64[]
    cs = Float64[]
    open(path, "r") do io
        for line in eachline(io)
            line = strip(line)
            (isempty(line) || startswith(line, "#") || startswith(line, "time_s")) && continue
            parts = split(line, ',')
            length(parts) < 2 && continue
            push!(ts, parse(Float64, strip(parts[1])))
            push!(cs, parse(Float64, strip(parts[2])))
        end
    end
    # Ensure sorted by time
    perm = sortperm(ts)
    AIFCurve(ts[perm], cs[perm])
end

# Linear interp on the AIF, clamped to [0, end]. Returns mg/mL.
@inline function aif_at(aif::AIFCurve, t::Float64)
    n = length(aif.t)
    n == 0 && return 0.0
    t <= aif.t[1] && return aif.C[1]
    t >= aif.t[end] && return aif.C[end]
    # Binary-search for the bracket
    lo, hi = 1, n
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        if aif.t[mid] <= t
            lo = mid
        else
            hi = mid
        end
    end
    t0, t1 = aif.t[lo], aif.t[hi]
    c0, c1 = aif.C[lo], aif.C[hi]
    α = (t - t0) / (t1 - t0)
    return c0 * (1.0 - α) + c1 * α
end

# Per-segment iodine concentration at time t, given arrival (s).
# Time-stretch dispersion (same form as the synthetic-gamma version):
#   disp_factor = sqrt(1 + arrival / t_disp)
#   t_eff       = (t - arrival) / disp_factor      ← measured from AIF t=0
#   C(t, a)     = AIF(t_eff) / disp_factor
@inline function contrast_at(aif::AIFCurve, t::Float64, arrival::Float64;
                             t_disp::Float64 = T_DISP_S)
    (!isfinite(arrival) || t <= arrival) && return 0.0
    df = sqrt(1.0 + arrival / t_disp)
    t_eff = (t - arrival) / df
    t_eff <= 0.0 && return 0.0
    return aif_at(aif, t_eff) / df
end

function main()
    if length(ARGS) < 1
        println("Usage: julia --threads=auto scripts/step3_voxelize_frames_aif.jl  AIF_CSV")
        exit(1)
    end
    aif_path = ARGS[1]
    isfile(aif_path) || error("AIF CSV not found: $aif_path")

    println("[step3-aif] aif_csv = $aif_path")
    aif = load_aif_csv(aif_path)
    @assert length(aif.t) > 1 "AIF curve must have ≥2 samples"
    @assert issorted(aif.t) "AIF time vector must be sorted"
    println("[step3-aif] AIF: $(length(aif.t)) samples, t ∈ [$(round(aif.t[1]; digits=3)), $(round(aif.t[end]; digits=3))] s, C ∈ [$(round(minimum(aif.C); digits=4)), $(round(maximum(aif.C); digits=4))] mg/mL")
    println("[step3-aif] Frames: 0..$(T_END_S)s @ $(DT_S)s → $(N_FRAMES) files")
    println("[step3-aif] HU_per_mg_ml=$(HU_PER_MG_ML), MYO_ECV=$(MYO_ECV), T_DISP_S=$(T_DISP_S)")
    println("[step3-aif] ARRIVAL_OVERRIDE_S=$(ARRIVAL_OVERRIDE_S) (≥0 ⇒ uniform myo arrival)")
    flush(stdout)

    # ── Load inputs ──
    println("[step3-aif] Loading phantom labels …"); flush(stdout)
    labels = Array{UInt8}(undef, PHANTOM_DIMS)
    read!(PHANTOM_PATH, labels)

    println("[step3-aif] Loading baseline HU …"); flush(stdout)
    baseline_HU = Array{Int16}(undef, PHANTOM_DIMS)
    read!(joinpath(INTER_DIR, "phantom_baseline_HU.raw"), baseline_HU)

    println("[step3-aif] Loading aorta lumen mask …"); flush(stdout)
    aorta_mask = Array{UInt8}(undef, PHANTOM_DIMS)
    read!(joinpath(INTER_DIR, "aorta_lumen_mask.raw"), aorta_mask)

    println("[step3-aif] Loading myo arrival times …"); flush(stdout)
    myo_arrival = Array{Float32}(undef, PHANTOM_DIMS)
    arrival_path = joinpath(INTER_DIR, "myo_arrival_patched.raw")
    if !isfile(arrival_path)
        arrival_path = joinpath(INTER_DIR, "myo_arrival.raw")
    end
    println("[step3-aif]   reading $(basename(arrival_path))")
    read!(arrival_path, myo_arrival)

    # ── Pre-flatten masks ──
    println("[step3-aif] Indexing AIF voxels and myo voxels …"); flush(stdout)

    aif_lin = Int[]
    myo_lin = Int[]
    sizehint!(aif_lin, 30_000_000)
    sizehint!(myo_lin, 25_000_000)

    @inbounds for q in 1:N_VOX
        L = labels[q]
        if L == 0x13 || L == 0x15 || L == 0x1A || aorta_mask[q] == 0x01
            push!(aif_lin, q)
        elseif 15 <= L <= 18
            push!(myo_lin, q)
        end
    end
    println("[step3-aif]   AIF voxels: $(length(aif_lin))")
    println("[step3-aif]   myo voxels: $(length(myo_lin))")
    flush(stdout)

    labels = nothing
    aorta_mask = nothing
    GC.gc()

    # ── Per-frame voxelize ──
    mkpath(FRAMES_DIR)
    times = collect(0.0:DT_S:T_END_S)
    @assert length(times) == N_FRAMES

    println("\n[step3-aif] C_AIF(t) preview (undispersed, arrival=0):")
    for t in times
        c = contrast_at(aif, t, 0.0)
        println("  t=$(round(t; digits=1))s  C_AIF=$(round(c; digits=3)) mg/mL  ΔHU=$(round(HU_PER_MG_ML*c; digits=1))")
    end
    flush(stdout)

    frame_HU = Array{Int16}(undef, PHANTOM_DIMS)

    for (fi, t) in enumerate(times)
        t0 = time()
        copyto!(frame_HU, baseline_HU)

        # Inject AIF (constant ΔHU across all AIF voxels)
        c_aif = contrast_at(aif, t, 0.0)
        dh_aif_int = round(Int, HU_PER_MG_ML * c_aif)
        if dh_aif_int != 0
            @inbounds Threads.@threads for k in 1:length(aif_lin)
                q = aif_lin[k]
                frame_HU[q] = clamp(Int(frame_HU[q]) + dh_aif_int, -32768, 32767) |> Int16
            end
        end

        # Inject myocardium (uniform arrival if override, else per-voxel; ECV-scaled).
        if ARRIVAL_OVERRIDE_S >= 0.0
            c_myo = contrast_at(aif, t, ARRIVAL_OVERRIDE_S)
            dh = round(Int, HU_PER_MG_ML * c_myo * MYO_ECV)
            if dh != 0
                @inbounds Threads.@threads for k in 1:length(myo_lin)
                    q = myo_lin[k]
                    frame_HU[q] = clamp(Int(frame_HU[q]) + dh, -32768, 32767) |> Int16
                end
            end
        else
            @inbounds Threads.@threads for k in 1:length(myo_lin)
                q = myo_lin[k]
                a = Float64(myo_arrival[q])
                c_myo = contrast_at(aif, t, a)
                dh = round(Int, HU_PER_MG_ML * c_myo * MYO_ECV)
                if dh != 0
                    frame_HU[q] = clamp(Int(frame_HU[q]) + dh, -32768, 32767) |> Int16
                end
            end
        end

        fname = "frame_$(lpad(fi-1, 2, '0')).raw"
        fpath = joinpath(FRAMES_DIR, fname)
        open(fpath, "w") do io
            write(io, frame_HU)
        end
        println("[step3-aif] frame $(fi-1)/$(N_FRAMES-1) (t=$(round(t; digits=1))s):" *
                " AIF ΔHU=$(dh_aif_int)  $(round(filesize(fpath)/1e9; digits=2)) GB" *
                "  ($(round(time()-t0; digits=1))s)")
        flush(stdout)
    end

    println("\n[step3-aif] Done. $(N_FRAMES) frames written to $(FRAMES_DIR)")
end

main()
