#!/usr/bin/env julia
# recon_to_dicom.jl — convert saved BasisSim Float32 HU recons (.raw) to DICOM
# series, without re-scanning. Used for the perfusion closed-loop V1/V2 volumes
# that were produced with --no-dicom.
import Pkg; Pkg.activate("/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/phantom_ct_input/run_basis_sim")
include("/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/phantom_ct_input/run_basis_sim/write_dicom.jl")
using TOML, Dates

PP = "/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline"

function convert(recon_dir, out_dir, desc; kind="fbp", kvp=100.0)
    meta = TOML.parsefile(joinpath(PP, recon_dir, "recon_meta.toml"))["recon"]
    nx, ny, nz = Int.(meta["shape"])
    vmm = Float64.(meta["voxel_size_mm"])        # [dy, dx, dz]
    hu = Array{Float32}(undef, nx, ny, nz)
    read!(joinpath(PP, recon_dir, "recon_$(kind)_hu_f32.raw"), hu)
    mkpath(joinpath(PP, out_dir))
    write_hu_to_dicom_series(joinpath(PP, out_dir), hu, (vmm[1], vmm[2], vmm[3]);
        series_description=desc, series_number=1, kvp=kvp, mA=250.0,
        window_center=200.0, window_width=600.0, acquisition_dt=now())
    println("[dicom] $desc → $(joinpath(PP, out_dir))  ($nz slices, voxel $(vmm) mm)")
end

convert("basissim_out_bae/v1_baseline", "dicom_perfusion/baseline_v1",  "Perfusion baseline V1 (t=0)")
convert("basissim_out_fp_rest/v2_peak",  "dicom_perfusion/rest_v2",      "Perfusion REST V2 peak (flow-prop)")
convert("basissim_out_fp_stress/v2_peak","dicom_perfusion/stress_v2",    "Perfusion STRESS V2 peak (flow-prop)")
println("DONE")
