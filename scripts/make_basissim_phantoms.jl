#!/usr/bin/env julia
# make_basissim_phantoms.jl — produce two UInt16 cross-product phantom volumes
# (V1 baseline + V2 peak-contrast) plus their manifests for BasisSimulator.
#
# Both phantoms share the same encoding so V1 and V2 only differ by the iodine
# bin number — the underlying material composition is identical for matching
# voxels, isolating the contrast effect cleanly. This is what allows the
# downstream perfusion calc (V2 − V1) to reflect pure iodine enhancement
# without spurious baseline shifts from re-encoding.
#
# Cross-product encoding (same as apply_contrast_at_peak.jl):
#   label = label_base + (bin_b - 1) × (N_iodine_bins + 1) + bin_i
#   bin_b ∈ 1..100  (blood volume fraction in voxel = bin_b / 100)
#   bin_i ∈ 0..100  (iodine in blood phase = bin_i / 100 × iodine_max_mg_per_mL)
#
# Per voxel:
#   • Aorta lumen (mask) + LV bp (label 19) + LA bp (label 21) + coronary trunk
#     (label 26): bin_b = 100, bin_i = round(C_AIF(t) / iodine_max × N_iodine_bins)
#   • Myocardium (labels 15-18): bin_b = 25 (ECV), bin_i = round(C_myo / iodine_max × N_iodine_bins)
#     where C_myo = AIF(t - ARRIVAL_OVERRIDE_S) with dispersion broadening.
#   • Everything else: keep original XCAT label (0..32).
#
# Output (in OUT_DIR/{v1_baseline,v2_peak}/):
#   phantom.raw                (UInt16, 1600×1400×500, 4.48 GB)
#   phantom_manifest.toml      (mixture_materials + materials map)
#
# Usage:
#   julia --threads=auto make_basissim_phantoms.jl  AIF_CSV  T_V1  T_V2  OUT_DIR

using Base.Threads
using Printf

const PHANTOM_PATH = "/home/molloi-lab/smb_mount/shared_drive/Shu Nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"
const PIPELINE_DIR = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline"
const INTER_DIR = joinpath(PIPELINE_DIR, "intermediate")
const AORTA_MASK = joinpath(INTER_DIR, "aorta_lumen_mask.raw")
const PHANTOM_DIMS = (1600, 1400, 500)
const N_VOX = prod(PHANTOM_DIMS)
const VOXEL_CM = 0.02
const XCAT_ORIGIN_CM = (2.846980, -9.773884, -20.600891)
const NRB_OFFSET = (2.1443, -9.5553, -20.0068)
const NRB_TO_PHANTOM = NRB_OFFSET

const N_BLOOD_BINS = 100
const N_IODINE_BINS = 100
const LABEL_BASE = UInt16(256)
const MAX_LABEL = LABEL_BASE + UInt16((N_BLOOD_BINS - 1) * (N_IODINE_BINS + 1) + N_IODINE_BINS)

const MYO_ECV_BIN = 25   # bin_b for myocardium = ECV (0.25)
const AORTA_BIN_B = 100  # bin_b for AIF voxels = pure blood
const T_DISP_S = 3.0
const ARRIVAL_OVERRIDE_S = 1.5  # uniform myo arrival (healthy autoregulation model)

# XCAT label → BasisSim material symbol (matches apply_contrast_at_peak.jl)
const LABEL_TO_MATERIAL = Dict{Int, String}(
    0  => "air",            1  => "softtissue",     2  => "softtissue",
    3  => "cortical_bone",  4  => "muscle",         5  => "cortical_bone",
    6  => "lung",           7  => "softtissue",     8  => "cortical_bone",
    9  => "cortical_bone",  10 => "softtissue",     11 => "softtissue",
    12 => "softtissue",     13 => "softtissue",     14 => "softtissue",
    15 => "muscle",         16 => "muscle",         17 => "muscle",
    18 => "muscle",         19 => "blood",          20 => "blood",
    21 => "blood",          22 => "blood",          23 => "softtissue",
    24 => "blood",          25 => "blood",          26 => "blood",
    27 => "blood",          28 => "blood",          29 => "softtissue",
    30 => "muscle",         31 => "adipose",        32 => "cortical_bone",
    70 => "air",
)

# ── AIF loader (same as step3_voxelize_frames_aif.jl) ──
struct AIFCurve
    t::Vector{Float64}
    C::Vector{Float64}
end
function load_aif_csv(path::String)
    ts, cs = Float64[], Float64[]
    open(path) do io
        for line in eachline(io)
            s = strip(line)
            (isempty(s) || startswith(s, "#") || startswith(s, "time_s")) && continue
            parts = split(s, ',')
            length(parts) < 2 && continue
            push!(ts, parse(Float64, strip(parts[1])))
            push!(cs, parse(Float64, strip(parts[2])))
        end
    end
    perm = sortperm(ts)
    AIFCurve(ts[perm], cs[perm])
end
@inline function aif_at(aif::AIFCurve, t::Float64)
    n = length(aif.t)
    n == 0 && return 0.0
    t <= aif.t[1] && return aif.C[1]
    t >= aif.t[end] && return aif.C[end]
    lo, hi = 1, n
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        aif.t[mid] <= t ? (lo = mid) : (hi = mid)
    end
    α = (t - aif.t[lo]) / (aif.t[hi] - aif.t[lo])
    return aif.C[lo] * (1.0 - α) + aif.C[hi] * α
end
@inline contrast_aif(aif, t) = t <= 0 ? 0.0 : aif_at(aif, t)
@inline function contrast_myo(aif::AIFCurve, t::Float64, arrival::Float64)
    (!isfinite(arrival) || t <= arrival) && return 0.0
    df = sqrt(1.0 + arrival / T_DISP_S)
    t_eff = (t - arrival) / df
    t_eff <= 0.0 && return 0.0
    return aif_at(aif, t_eff) / df
end

# ── Flow-proportional first-pass myocardial uptake ──
# Mullani-Gould recovers  P = 60·C_myo/(AUC·ρ), so to make recovered ≈ input
# perfusion P [mL/min/g] we set  C_myo(t) = P · ∫₀ᵗAIF dτ · ρ / 60.
const RHO_MYO = 1.053   # g/mL
@inline function cum_auc(aif::AIFCurve, t::Float64)   # ∫₀ᵗ AIF dτ  [mg·s/mL]
    t <= 0.0 && return 0.0
    s = 0.0
    @inbounds for k in 2:length(aif.t)
        t0 = aif.t[k-1]; t1 = aif.t[k]
        if t1 <= t
            s += 0.5 * (aif.C[k-1] + aif.C[k]) * (t1 - t0)
        else
            t0 < t && (s += 0.5 * (aif.C[k-1] + aif_at(aif, t)) * (t - t0))
            break
        end
    end
    return s
end

# ── Cross-product label encoding ──
@inline function xprod_label(bin_b::Int, bin_i::Int)::UInt16
    LABEL_BASE + UInt16((bin_b - 1) * (N_IODINE_BINS + 1) + bin_i)
end

function voxelize_phantom_at_time!(out::Array{UInt16,3},
                                   labels::Array{UInt8,3},
                                   aorta_mask::Array{UInt8,3},
                                   aif::AIFCurve, t::Float64,
                                   iodine_max::Float64;
                                   per_voxel_arrival::Union{Nothing, Array{Float32,3}} = nothing,
                                   arrival_cap_s::Float64 = -1.0,
                                   per_voxel_perfusion::Union{Nothing, Array{Float32,3}} = nothing,
                                   cum_auc_t::Float64 = 0.0)
    nx, ny, nz = size(labels)
    use_perf = per_voxel_perfusion !== nothing

    # AIF voxels always share one label.
    c_aif_t = contrast_aif(aif, t)
    bin_i_aif = clamp(round(Int, c_aif_t / iodine_max * N_IODINE_BINS), 0, N_IODINE_BINS)
    label_aif = xprod_label(AORTA_BIN_B, bin_i_aif)

    # Myo: if per_voxel_arrival is given, each voxel gets its own bin_i; else
    # use ARRIVAL_OVERRIDE_S uniformly.
    use_pv = per_voxel_arrival !== nothing
    label_myo_shared = if !use_pv
        c_myo_t = contrast_myo(aif, t, ARRIVAL_OVERRIDE_S)
        bin = clamp(round(Int, c_myo_t / iodine_max * N_IODINE_BINS), 0, N_IODINE_BINS)
        xprod_label(MYO_ECV_BIN, bin)
    else
        UInt16(0)
    end

    if use_pv
        @printf("[voxelize] t=%.2fs  C_AIF=%.4f mg/mL (bin_i=%d, label=%d)  myo: per-voxel arrival (cap=%.1fs)\n",
                t, c_aif_t, bin_i_aif, Int(label_aif), arrival_cap_s)
    else
        c_myo_t = contrast_myo(aif, t, ARRIVAL_OVERRIDE_S)
        @printf("[voxelize] t=%.2fs  C_AIF=%.4f mg/mL (bin_i=%d, label=%d)  C_myo=%.4f mg/mL (uniform arrival=%.1fs)\n",
                t, c_aif_t, bin_i_aif, Int(label_aif), c_myo_t, ARRIVAL_OVERRIDE_S)
    end
    flush(stdout)

    n_aif = Threads.Atomic{Int}(0)
    n_myo = Threads.Atomic{Int}(0)
    n_myo_unreach = Threads.Atomic{Int}(0)

    @threads :static for k in 1:nz
        local_aif = 0; local_myo = 0; local_unreach = 0
        @inbounds for j in 1:ny, i in 1:nx
            L = labels[i, j, k]
            is_aif = (L == 0x13) | (L == 0x15) | (L == 0x1A) | (aorta_mask[i, j, k] == 0x01)
            if is_aif
                out[i, j, k] = label_aif
                local_aif += 1
            elseif 15 <= L <= 18
                if use_perf
                    P = per_voxel_perfusion[i, j, k]
                    cmyo = (isfinite(P) && P > 0.0) ? Float64(P) * cum_auc_t * RHO_MYO / 60.0 : 0.0
                    bin = clamp(round(Int, cmyo / iodine_max * N_IODINE_BINS), 0, N_IODINE_BINS)
                    out[i, j, k] = xprod_label(AORTA_BIN_B, bin)   # bin_b=100 → total voxel iodine = cmyo
                elseif use_pv
                    a_raw = per_voxel_arrival[i, j, k]
                    a = if isfinite(a_raw)
                        arrival_cap_s > 0.0 ? min(Float64(a_raw), arrival_cap_s) : Float64(a_raw)
                    else
                        local_unreach += 1
                        arrival_cap_s > 0.0 ? arrival_cap_s : Float64(ARRIVAL_OVERRIDE_S)
                    end
                    c = contrast_myo(aif, t, a)
                    bin = clamp(round(Int, c / iodine_max * N_IODINE_BINS), 0, N_IODINE_BINS)
                    out[i, j, k] = xprod_label(MYO_ECV_BIN, bin)
                else
                    out[i, j, k] = label_myo_shared
                end
                local_myo += 1
            else
                out[i, j, k] = UInt16(L)
            end
        end
        Threads.atomic_add!(n_aif, local_aif)
        Threads.atomic_add!(n_myo, local_myo)
        Threads.atomic_add!(n_myo_unreach, local_unreach)
    end
    if use_pv && n_myo_unreach[] > 0
        @printf("[voxelize]   %d/%d myo voxels had non-finite arrival (treated with cap or override)\n",
                n_myo_unreach[], n_myo[])
    end
    return n_aif[], n_myo[]
end

function write_manifest(path::String, raw_basename::String,
                        iodine_max::Float64, time_s::Float64,
                        n_aif_voxels::Int, n_myo_voxels::Int,
                        c_aif_t::Float64, c_myo_t::Float64,
                        label_aif::UInt16, label_myo::UInt16)
    open(path, "w") do io
        println(io, "# phantom_manifest.toml — V1/V2 cross-product phantom for BasisSim")
        println(io, "# Generated by make_basissim_phantoms.jl at frame time t=$(time_s)s")
        println(io)
        println(io, "[phantom]")
        println(io, "raw_path = \"$raw_basename\"")
        println(io, "dims = [$(PHANTOM_DIMS[1]), $(PHANTOM_DIMS[2]), $(PHANTOM_DIMS[3])]")
        println(io, "dtype = \"UInt16\"")
        println(io, "byte_order = \"little-endian\"")
        println(io, "voxel_ordering = \"x-fastest\"")
        println(io, "voxel_size_cm = [$VOXEL_CM, $VOXEL_CM, $VOXEL_CM]")
        println(io, "xcat_origin_cm = [$(XCAT_ORIGIN_CM[1]), $(XCAT_ORIGIN_CM[2]), $(XCAT_ORIGIN_CM[3])]")
        println(io)
        println(io, "[embed]")
        println(io, "stage = \"basissim_perfusion_frame\"")
        @printf(io, "time_s = %.4f\n", time_s)
        println(io, "nrb_to_phantom_offset_cm = [$(NRB_TO_PHANTOM[1]), $(NRB_TO_PHANTOM[2]), $(NRB_TO_PHANTOM[3])]")
        println(io, "writable_base_labels = [15, 16, 17, 18, 19, 21, 26]")
        println(io, "ecv_myo = 0.25")
        println(io, "t_dispersion_s = $T_DISP_S")
        println(io, "arrival_override_s = $ARRIVAL_OVERRIDE_S")
        println(io)
        println(io, "[contrast]")
        @printf(io, "c_aif_at_time_mg_per_mL  = %.6f\n", c_aif_t)
        @printf(io, "c_myo_at_time_mg_per_mL  = %.6f\n", c_myo_t)
        println(io, "n_aif_voxels = $n_aif_voxels")
        println(io, "n_myo_voxels = $n_myo_voxels")
        println(io, "aif_label = $(Int(label_aif))")
        println(io, "myo_label = $(Int(label_myo))")
        println(io)
        println(io, "[mixture_materials]")
        println(io, "components_base = [\"blood\", \"muscle\"]")
        println(io, "contrast_agent  = \"iodine\"")
        println(io, "n_blood_bins  = $N_BLOOD_BINS")
        println(io, "n_iodine_bins = $N_IODINE_BINS")
        println(io, "label_base    = $(Int(LABEL_BASE))")
        @printf(io, "iodine_max_mg_per_mL = %.6f\n", iodine_max)
        println(io, "encoding = \"label = label_base + (bin_b - 1) * (n_iodine_bins + 1) + bin_i\"")
        @printf(io, "max_label = %d\n", Int(MAX_LABEL))
        println(io)
        println(io, "[materials]")
        for k in sort(collect(keys(LABEL_TO_MATERIAL)))
            println(io, "\"$k\" = \"$(LABEL_TO_MATERIAL[k])\"")
        end
    end
    @printf("[write] %s\n", path); flush(stdout)
end

function main()
    # Parse positional + optional args
    pos = String[]
    use_pv = false
    pv_path = ""
    cap_s = -1.0
    use_perf = false
    perf_path = ""
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--use-myo-arrival"
            use_pv = true
            pv_path = ARGS[i+1]; i += 2
        elseif a == "--arrival-cap"
            cap_s = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--use-perfusion-map"
            use_perf = true
            perf_path = ARGS[i+1]; i += 2
        else
            push!(pos, a); i += 1
        end
    end
    if length(pos) < 4
        println("Usage: julia --threads=auto make_basissim_phantoms.jl  AIF_CSV  T_V1  T_V2  OUT_DIR  [--use-myo-arrival PATH]  [--arrival-cap SECONDS]")
        exit(1)
    end
    aif_csv = pos[1]
    t_v1 = parse(Float64, pos[2])
    t_v2 = parse(Float64, pos[3])
    out_dir = pos[4]
    isfile(aif_csv) || error("AIF CSV not found: $aif_csv")
    mkpath(out_dir)

    pv_arrival = nothing
    if use_pv
        isfile(pv_path) || error("myo_arrival.raw not found: $pv_path")
        @printf("[load] per-voxel myo_arrival = %s  (cap=%.1fs)\n", pv_path, cap_s)
        pv_arrival = Array{Float32}(undef, PHANTOM_DIMS)
        read!(pv_path, pv_arrival)
    end

    perf_map = nothing
    if use_perf
        isfile(perf_path) || error("perfusion map not found: $perf_path")
        @printf("[load] per-voxel perfusion map = %s  (flow-proportional mode)\n", perf_path)
        perf_map = Array{Float32}(undef, PHANTOM_DIMS)
        read!(perf_path, perf_map)
    end

    @printf("threads = %d\n", nthreads())
    @printf("aif_csv = %s\n", aif_csv)
    @printf("V1 time = %.2fs (baseline)\n", t_v1)
    @printf("V2 time = %.2fs (peak)\n", t_v2)
    @printf("out_dir = %s\n", out_dir)
    flush(stdout)

    aif = load_aif_csv(aif_csv)
    @printf("[aif] %d samples, t ∈ [%.3f, %.3f] s, C ∈ [%.4f, %.4f] mg/mL\n",
            length(aif.t), aif.t[1], aif.t[end], minimum(aif.C), maximum(aif.C))

    iodine_max = maximum(aif.C)
    if iodine_max <= 0
        error("AIF max is non-positive; cannot scale iodine bins")
    end
    if use_perf
        auc_v2 = cum_auc(aif, t_v2)
        finite_p = filter(p -> isfinite(p) && p > 0, vec(perf_map))
        maxP = isempty(finite_p) ? 0.0 : maximum(finite_p)
        max_cmyo = maxP * auc_v2 * RHO_MYO / 60.0
        iodine_max = max(iodine_max, max_cmyo) * 1.05
        @printf("[perfusion] AUC(0..%.1fs)=%.3f mg·s/mL, maxP=%.2f mL/min/g → max C_myo=%.2f mg/mL → iodine_max=%.2f\n",
                t_v2, auc_v2, maxP, max_cmyo, iodine_max)
    end
    @printf("[encoding] iodine_max = %.4f mg/mL (= max(AIF)); N_iodine_bins = %d\n",
            iodine_max, N_IODINE_BINS)
    flush(stdout)

    # Load shared inputs once
    println("[load] phantom labels …"); flush(stdout)
    labels = Array{UInt8}(undef, PHANTOM_DIMS)
    read!(PHANTOM_PATH, labels)

    println("[load] aorta lumen mask …"); flush(stdout)
    aorta_mask = Array{UInt8}(undef, PHANTOM_DIMS)
    read!(AORTA_MASK, aorta_mask)

    out_u16 = Array{UInt16}(undef, PHANTOM_DIMS)

    for (tag, t) in (("v1_baseline", t_v1), ("v2_peak", t_v2))
        sub = joinpath(out_dir, tag)
        mkpath(sub)
        println("\n=== $tag (t=$t s) ===")
        t0 = time()
        c_aif_t = contrast_aif(aif, t)
        c_myo_t = contrast_myo(aif, t, ARRIVAL_OVERRIDE_S)
        bin_i_aif = clamp(round(Int, c_aif_t / iodine_max * N_IODINE_BINS), 0, N_IODINE_BINS)
        bin_i_myo = clamp(round(Int, c_myo_t / iodine_max * N_IODINE_BINS), 0, N_IODINE_BINS)
        label_aif = xprod_label(AORTA_BIN_B, bin_i_aif)
        label_myo = xprod_label(MYO_ECV_BIN, bin_i_myo)

        n_aif, n_myo = voxelize_phantom_at_time!(out_u16, labels, aorta_mask, aif, t, iodine_max;
                                                  per_voxel_arrival = pv_arrival,
                                                  arrival_cap_s = cap_s,
                                                  per_voxel_perfusion = perf_map,
                                                  cum_auc_t = cum_auc(aif, t))
        @printf("[voxelize] aif_voxels=%d  myo_voxels=%d  (%.1fs)\n", n_aif, n_myo, time()-t0); flush(stdout)

        raw_path = joinpath(sub, "phantom.raw")
        open(raw_path, "w") do io;  write(io, out_u16);  end
        @printf("[write] %s (%.1f GB)\n", raw_path, filesize(raw_path)/1e9)
        flush(stdout)

        write_manifest(joinpath(sub, "phantom_manifest.toml"), "phantom.raw",
                       iodine_max, t, n_aif, n_myo, c_aif_t, c_myo_t,
                       label_aif, label_myo)
    end

    println("\n[done] V1 and V2 phantoms ready under $out_dir/")
    println("  v1_baseline/phantom.raw + phantom_manifest.toml")
    println("  v2_peak/phantom.raw     + phantom_manifest.toml")
end

main()
