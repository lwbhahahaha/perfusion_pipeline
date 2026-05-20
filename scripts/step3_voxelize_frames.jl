#!/usr/bin/env julia
# Step 3: write 21 contrast-enhanced raw frames at 1s resolution.
#
# For each t = 0, 1, …, 20 s, copies the baseline HU and adds ΔHU = 25·C(t)
# (clinical iodine, 25 HU per mg/mL) to:
#
#   • LV blood pool (label 19), LA blood pool (label 21),
#     aorta lumen (mask from step 1), main coronary trunk (label 26)
#       → C_AIF(t) (clinical gamma, undispersed; this is what SureStart sees)
#   • myocardium (labels 15–18)
#       → C_AIF(t, arrival[i,j,k]) (dispersed, arrival from step 2)
#
# Output:
#   frames/frame_{tt:02d}.raw   (Int16, 1600x1400x500, 21 files, 47 GB total)

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
const AIF_AMP   = Float64(META["aif_clinical"]["amplitude_mg_ml"])
const AIF_T0    = Float64(META["aif_clinical"]["t0_s"])
const AIF_TMAX  = Float64(META["aif_clinical"]["tmax_s"])
const AIF_ALPHA = Float64(META["aif_clinical"]["alpha"])
const DT_S      = Float64(META["frames"]["dt_s"])
const T_END_S   = Float64(META["frames"]["t_end_s"])
const N_FRAMES  = Int(META["frames"]["n_frames"])
const T_DISP_S  = 3.0  # capillary dispersion broadening time scale

# Myocardium extracellular volume fraction (iodine distribution space).
# Iodine is plasma-only and equilibrates rapidly with myocardial ECF over the
# bolus pass; tissue HU enhancement ≈ ECV × blood HU enhancement.
# Pries-Secomb / clinical CT-MRI ECV maps: 0.25 ± 0.04 normal myocardium.
# LV/LA blood-pool + aorta lumen voxels are pure blood and use ECV = 1.0.
const MYO_ECV = 0.25

# Uniform myocardial arrival time (s). Real myocardium has tight arteriolar
# autoregulation that produces near-uniform capillary arrival ≈ 1-2 s in
# health. Our 6 µm Pries+Poiseuille tree, lacking autoregulation, produces
# arrival times 1-100+ s with a long pathological tail. Override per-voxel
# arrival to a uniform physiological value to model the missing autoregulation
# — this gives a uniform perfusion map for healthy myocardium (no disease)
# as expected clinically. Set to <0 to disable (use per-voxel arrival).
const ARRIVAL_OVERRIDE_S = 1.5

println("[step3] AIF: amp=$(AIF_AMP) mg/mL, t0=$(AIF_T0)s, tmax=$(AIF_TMAX)s, α=$(AIF_ALPHA)")
println("[step3] Frames: 0..$(T_END_S)s @ $(DT_S)s → $(N_FRAMES) files")
flush(stdout)

# ── Contrast formula (clinical gamma + dispersion) ──
# c_aif(t)              = c(t, arrival=0)
# c_capillary(t, a)     = c(t, arrival=a) with sqrt(1+a/t_disp) broadening
@inline function contrast_at(t::Float64, arrival::Float64;
                             amp::Float64=AIF_AMP, t0::Float64=AIF_T0,
                             tmax::Float64=AIF_TMAX, alpha::Float64=AIF_ALPHA,
                             t_disp::Float64=T_DISP_S)
    (!isfinite(arrival) || t <= arrival) && return 0.0
    t_shifted = t - arrival
    df = sqrt(1.0 + arrival / t_disp)
    t_input = t0 + (t_shifted - t0) / df
    t_input <= t0 && return 0.0
    tp = (t_input - t0) / (tmax - t0)
    tp <= 0 && return 0.0
    return amp * tp^alpha * exp(alpha * (1.0 - tp)) / df
end

# ── Load inputs ──
println("[step3] Loading phantom labels …"); flush(stdout)
labels = Array{UInt8}(undef, PHANTOM_DIMS)
read!(PHANTOM_PATH, labels)

println("[step3] Loading baseline HU …"); flush(stdout)
baseline_HU = Array{Int16}(undef, PHANTOM_DIMS)
read!(joinpath(INTER_DIR, "phantom_baseline_HU.raw"), baseline_HU)

println("[step3] Loading aorta lumen mask …"); flush(stdout)
aorta_mask = Array{UInt8}(undef, PHANTOM_DIMS)
read!(joinpath(INTER_DIR, "aorta_lumen_mask.raw"), aorta_mask)

println("[step3] Loading myo arrival times …"); flush(stdout)
myo_arrival = Array{Float32}(undef, PHANTOM_DIMS)
arrival_path = joinpath(INTER_DIR, "myo_arrival_patched.raw")
if !isfile(arrival_path)
    arrival_path = joinpath(INTER_DIR, "myo_arrival.raw")
end
println("[step3]   reading $(basename(arrival_path))")
read!(arrival_path, myo_arrival)

# ── Pre-flatten masks ──
println("[step3] Indexing AIF voxels (LV/LA/aorta/coronary trunk) and myo voxels …")
flush(stdout)

aif_lin = Int[]    # voxels that get C_AIF(t) (no dispersion)
myo_lin = Int[]    # voxels that get C_capillary(t, arrival) with their own arrival
sizehint!(aif_lin, 30_000_000)
sizehint!(myo_lin, 25_000_000)

@inbounds for q in 1:N_VOX
    L = labels[q]
    if L == 0x13 || L == 0x15 || L == 0x1A || aorta_mask[q] == 0x01
        # 19 (LV bp), 21 (LA bp), 26 (coronary trunk), aorta lumen
        push!(aif_lin, q)
    elseif 15 <= L <= 18
        push!(myo_lin, q)
    end
end
println("[step3]   AIF voxels: $(length(aif_lin))")
println("[step3]   myo voxels: $(length(myo_lin))")
flush(stdout)

# Free phantom labels and aorta mask now
labels = nothing
aorta_mask = nothing
GC.gc()

# ── Per-frame voxelize ──
mkpath(FRAMES_DIR)
times = collect(0.0:DT_S:T_END_S)
@assert length(times) == N_FRAMES

# Print AIF curve for sanity
println("\n[step3] C_AIF(t) preview:")
for t in times
    c = contrast_at(t, 0.0)
    println("  t=$(round(t; digits=1))s  C_AIF=$(round(c; digits=3)) mg/mL  ΔHU=$(round(HU_PER_MG_ML*c; digits=1))")
end
flush(stdout)

# Working frame buffer (reused across iterations)
frame_HU = Array{Int16}(undef, PHANTOM_DIMS)

for (fi, t) in enumerate(times)
    t0 = time()
    # Copy baseline (full volume)
    copyto!(frame_HU, baseline_HU)

    # Inject AIF (constant ΔHU across all AIF voxels)
    c_aif = contrast_at(t, 0.0)
    dh_aif_int = round(Int, HU_PER_MG_ML * c_aif)
    if dh_aif_int != 0
        @inbounds Threads.@threads for k in 1:length(aif_lin)
            q = aif_lin[k]
            frame_HU[q] = clamp(Int(frame_HU[q]) + dh_aif_int, -32768, 32767) |> Int16
        end
    end

    # Inject myocardium (uniform arrival if ARRIVAL_OVERRIDE_S >= 0,
    # else per-voxel from territory map; ECV-scaled to tissue HU).
    if ARRIVAL_OVERRIDE_S >= 0.0
        c_myo = contrast_at(t, ARRIVAL_OVERRIDE_S)
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
            c_myo = contrast_at(t, a)
            dh = round(Int, HU_PER_MG_ML * c_myo * MYO_ECV)
            if dh != 0
                frame_HU[q] = clamp(Int(frame_HU[q]) + dh, -32768, 32767) |> Int16
            end
        end
    end

    # Write frame
    fname = "frame_$(lpad(fi-1, 2, '0')).raw"
    fpath = joinpath(FRAMES_DIR, fname)
    open(fpath, "w") do io
        write(io, frame_HU)
    end
    println("[step3] frame $(fi-1)/$(N_FRAMES-1) (t=$(round(t; digits=1))s):" *
            " AIF ΔHU=$(dh_aif_int)  $(round(filesize(fpath)/1e9; digits=2)) GB" *
            "  ($(round(time()-t0; digits=1))s)")
    flush(stdout)
end

println("\n[step3] Done. $(N_FRAMES) frames written to $(FRAMES_DIR)")
