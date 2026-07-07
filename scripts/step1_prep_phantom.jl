#!/usr/bin/env julia
# Step 1: build XCAT phantom baseline HU (Int16) + aorta lumen mask.
#
# Outputs:
#   intermediate/phantom_baseline_HU.raw  (Int16, 1600x1400x500, no contrast)
#   intermediate/aorta_lumen_mask.raw     (UInt8, 1600x1400x500, 1=aorta lumen)
#   intermediate/metadata.toml            (dims, voxel size, AIF parameters)
#
# The phantom is XCAT vmale50 act_1.raw (UInt8 activity labels). We map labels
# to typical CT HU via a lookup table. Aorta lumen is voxelized from the
# dias_aorta NRB surface (XCAT activity phantom does not have a dedicated
# aorta-lumen label — vmale50 label 23 is body, not aorta).

using LinearAlgebra
using StaticArrays
using TOML

const VTS_PATH = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/VascularTreeSim.jl"
push!(LOAD_PATH, VTS_PATH)
using Pkg
Pkg.activate(VTS_PATH)
using VascularTreeSim

const PHANTOM_PATH = "/home/molloi-lab/smb_mount/shared_drive/Shu Nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"
const NRB_PATH = "/home/molloi-lab/smb_mount/shared_drive/XCAT Phantom/xcat_adult_nrb_files/vmale50_heart.nrb"
const PHANTOM_DIMS = (1600, 1400, 500)
const VOXEL_SIZE_CM = 0.02
const COORDINATE_SCALE = 0.1  # NRB mm → cm
# Empirically determined by VascularTreeSim/voxelizer.jl for vmale50.
const NRB_TO_PHANTOM_OFFSET = SVector(2.1443, -9.5553, -20.0068)

const OUT_DIR = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline/intermediate"

# ── XCAT activity label → CT HU lookup ──
# vmale50_act_1.raw uses XCAT activity labels (UInt8 0..30+). Most are 30 HU
# (soft tissue) by default; specific labels are overridden below. Reference:
# typical adult CT @ 120 kVp.
function build_label_hu_lookup()
    lut = fill(Int16(30), 256)              # default soft tissue 30 HU
    lut[1]  = Int16(-1000)                  # 0  = air outside body
    lut[5]  = Int16(-700)                   # 4  = lung
    lut[7]  = Int16(-100)                   # 6  = fat
    lut[16] = Int16(50)                     # 15 = LV myocardium
    lut[17] = Int16(50)                     # 16 = RV myocardium
    lut[18] = Int16(50)                     # 17 = LA myocardium
    lut[19] = Int16(50)                     # 18 = RA myocardium
    lut[20] = Int16(30)                     # 19 = LV blood pool (no contrast)
    lut[21] = Int16(30)                     # 20 = RV blood pool
    lut[22] = Int16(30)                     # 21 = LA blood pool
    lut[23] = Int16(30)                     # 22 = RA blood pool
    lut[24] = Int16(30)                     # 23 = body soft tissue
    lut[25] = Int16(30)                     # 24 = pulmonary artery
    lut[26] = Int16(30)                     # 25 = pulmonary veins
    lut[27] = Int16(30)                     # 26 = coronary arteries
    lut[28] = Int16(30)                     # 27 = coronary veins
    lut[29] = Int16(30)                     # 28 = vena cava
    lut[30] = Int16(30)                     # 29 = pericardium
    for L in 30:255                         # 30+ = bone/cartilage etc.
        lut[L+1] = Int16(300)
    end
    return lut
end

println("[step1] Loading vmale50_act_1.raw …")
flush(stdout)
phantom = Array{UInt8}(undef, PHANTOM_DIMS)
read!(PHANTOM_PATH, phantom)
println("[step1]   loaded $(round(sizeof(phantom)/1e9; digits=2)) GB phantom")
flush(stdout)

println("[step1] Building Int16 baseline HU volume …")
flush(stdout)
lut = build_label_hu_lookup()
baseline_HU = Array{Int16}(undef, PHANTOM_DIMS)
@inbounds Threads.@threads for k in 1:PHANTOM_DIMS[3]
    for j in 1:PHANTOM_DIMS[2], i in 1:PHANTOM_DIMS[1]
        baseline_HU[i, j, k] = lut[Int(phantom[i, j, k]) + 1]
    end
end
println("[step1]   HU range $(minimum(baseline_HU)) .. $(maximum(baseline_HU))")
flush(stdout)

baseline_path = joinpath(OUT_DIR, "phantom_baseline_HU.raw")
println("[step1] Writing $(baseline_path) …")
flush(stdout)
open(baseline_path, "w") do io
    write(io, baseline_HU)
end
println("[step1]   $(round(filesize(baseline_path)/1e9; digits=2)) GB written")
flush(stdout)

# ── Aorta lumen mask = XCAT aorta label (28), phantom-aligned ──
# Previously this rasterized a cylinder along the NRB dias_aorta centerline, but
# the diastolic NRB surface does not co-register with the act_1 LABEL volume:
# the mask landed ~16-27 voxels off and spilled ~9% into the spine (descending
# aorta runs against the vertebra). The XCAT label IS the phantom anatomy, so it
# aligns exactly — no NRB parse, no transform, no cylinder approximation.
println("\n[step1] Building aorta lumen mask from XCAT label 28 …")
flush(stdout)
aorta_label = UInt8(28)
aorta_mask = zeros(UInt8, PHANTOM_DIMS)
@inbounds Threads.@threads for k in 1:PHANTOM_DIMS[3]
    for j in 1:PHANTOM_DIMS[2], i in 1:PHANTOM_DIMS[1]
        phantom[i, j, k] == aorta_label && (aorta_mask[i, j, k] = UInt8(1))
    end
end
n_aorta = count(==(UInt8(1)), aorta_mask)
println("[step1]   aorta lumen (XCAT label 28): $(n_aorta) voxels = $(round(n_aorta * VOXEL_SIZE_CM^3; digits=2)) cm³")
flush(stdout)

aorta_path = joinpath(OUT_DIR, "aorta_lumen_mask.raw")
println("[step1] Writing $(aorta_path) …")
flush(stdout)
open(aorta_path, "w") do io
    write(io, aorta_mask)
end
println("[step1]   $(round(filesize(aorta_path)/1e9; digits=2)) GB written")
flush(stdout)

# ── Save metadata ──
metadata = Dict(
    "phantom_dims"           => collect(PHANTOM_DIMS),
    "voxel_size_cm"          => VOXEL_SIZE_CM,
    "coordinate_scale"       => COORDINATE_SCALE,
    "nrb_to_phantom_offset_cm" => collect(NRB_TO_PHANTOM_OFFSET),
    "n_aorta_voxels"         => n_aorta,
    "hu_per_mg_ml_iodine"    => 25.0,
    "aif_clinical"           => Dict(
        "amplitude_mg_ml" => 8.0,
        "t0_s"            => 2.0,
        "tmax_s"          => 8.0,
        "alpha"           => 4.0,
    ),
    "frames"                 => Dict(
        "dt_s"     => 1.0,
        "t_end_s"  => 20.0,
        "n_frames" => 21,
    ),
)
metadata_path = joinpath(OUT_DIR, "metadata.toml")
open(metadata_path, "w") do io
    TOML.print(io, metadata)
end
println("[step1] Wrote metadata: $(metadata_path)")
flush(stdout)
println("\n[step1] Done.")
