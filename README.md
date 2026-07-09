# perfusion_pipeline

End-to-end CT myocardial perfusion **simulation** (not fabrication): iodine is
transported through the grown coronary tree with a real advection–dispersion
PDE, the actual vessel geometry is voxelized into an XCAT phantom by true
occupied volume, each timepoint is forward-projected + reconstructed through
BasisSimulator, and perfusion is *measured* from the resulting recons.

Built on [VascularTreeSim.jl](../VascularTreeSim.jl) (tree growth + voxelizer)
and [FlowContrastSim.jl](../FlowContrastSim.jl) (hemodynamics + contrast PDE).

**Physics-only, no calibration knob, no painted signal.** The myocardial
enhancement is whatever the capillary bed physically carries into each voxel —
it is never set from a target-perfusion formula. Because the signal is
emergent, the downstream perfusion estimate is an honest measurement and
reproduces the real CT-MPI flow roll-off (high stress flow is under-recovered),
rather than round-tripping an imposed number.

---

## The pipeline

```
grown tree CSVs ─┬─▶ FlowContrastSim: simulate_contrast (Taylor–Aris PDE)
                 │      → per-segment iodine C(seg, t)      [extract_peak_iodine_bae.jl]
                 │
                 ▼
   apply_contrast_at_peak.jl   emergent occupied-volume voxelization
       f_blood = Σ(vessel∩voxel volume)/voxel_vol   (πr²·len for capillaries,
       sub-voxel Monte-Carlo for resolvable vessels ≥200 μm) — NO imposed MBV
                 │
                 ▼
   add_chambers_to_phantom.jl   patch LV/LA/aorta blood pools with the real AIF
                 │
                 ▼
   run_cta_sim_param.jl (BasisSim)   scan + FBP recon  →  recon_fbp_hu_f32.raw
                 │
                 ▼
   recover_perfusion.py   map myo/territory masks into recon space, measure
                          max-slope CT-MPI perfusion vs. known input flow
```

Driver: [`scripts/run_dynamic_perfusion.sh`](scripts/run_dynamic_perfusion.sh)
(stages 1–4). Measurement: [`scripts/recover_perfusion.py`](scripts/recover_perfusion.py).

The lab flow-map method (Hubbard–Molloi single-frame ΔHU / AIF-integral, from
`MolloiLab/PerfusionImagingNotebooks`) is the alternative estimator run on the
same recons; it recovers more of the true flow than max-slope but still
under-estimates high stress flow — the expected, honest CT-MPI behavior.

---

## Result (closed loop, no per-organ tuning)

Ground-truth perfusion = tree flow ÷ territory mass; both estimators run on the
BasisSim recon of the emergent contrast:

| Territory | Input (truth) | Lab ΔHU/AIF map | Max-slope |
|---|:---:|:---:|:---:|
| LAD | 4.20 | 2.37 | 0.61 |
| LCX | 4.01 | 3.05 | 0.74 |
| RCA | 3.94 | 2.19 | 0.66 |

Peak myocardial HU is ~70–85 (real-patient range) — a consequence of the
~12 % emergent myocardial blood volume, not an imposed ECV.

---

## Scripts

### Pipeline (honest, canonical)
| script | role |
|---|---|
| `run_dynamic_perfusion.sh` | end-to-end driver: real-PDE iodine → emergent voxelize → chambers → BasisSim recon |
| `recover_perfusion.py` | measure max-slope CT-MPI perfusion from the recon series vs. input flow; writes `dynamic_perfusion_figure.png` |
| `per_territory_recovery.py` | split a recon-space perfusion map by coronary territory (LAD/LCX/RCA) |
| `lab_method_equiv.py` | run the lab `compute_perfusion` (Hubbard–Molloi) as an alternative estimator on the recons |
| `recon_to_dicom.jl` | convert BasisSim Float32 HU recons to DICOM series |

The voxelizers themselves live upstream:
`../VascularTreeSim.jl/scripts/apply_contrast_at_peak.jl` (emergent contrast
voxelization) and `add_chambers_to_phantom.jl`;
`../FlowContrastSim.jl/scripts/extract_peak_iodine_bae.jl` (real-PDE per-frame
iodine).

### Inputs / one-time setup
| script | role |
|---|---|
| `step0_prepare_aif.py` | patient bolus-tracking `.mat` → `aif_curve_*.csv` (HU→mg/mL, baseline-subtracted) |
| `step0b_extrapolate_aif.py` | optional gamma-variate wash-out extrapolation for truncated AIFs |
| `step1_prep_phantom.jl` | XCAT label remap → baseline HU + aorta lumen mask (aorta from XCAT label 28) |
| `step2_build_territory.jl` | per-myocardium-voxel territory map `myo_tree_id.raw` (UInt8 0/1/2/3 = none/LAD/LCX/RCA) |
| `scan_all_aifs.py` | rank UCLA subjects by REST/STRESS AUC ratio (CFR-candidate selection) |

### Viewers / QA
`build_perfusion_viewer.py`, `build_basissim_perfusion_viewer.py`,
`surface_perfusion_viewer.py`, `smooth_downsample_viewer.py`,
`build_chamber_mesh.py`, `extract_tree_overlay.py`, `qa_heart_crop.py`,
`qa_viewer.py`, `rsna_figure.py` — 3D perfusion/tree viewers and figure
helpers.

---

## Intermediate files (per phantom — rebuild as needed, gitignored)

```
intermediate/
  phantom_baseline_HU.raw   Int16 1600×1400×500, no contrast (step1)
  aorta_lumen_mask.raw      UInt8 1600×1400×500, 1 = aorta lumen (step1)
  myo_tree_id.raw           UInt8, 0/1/2/3 = none/LAD/LCX/RCA territory (step2)
```

---

## BasisSim integration

The BasisSim runner lives in `../phantom_ct_input/run_basis_sim/`.
`run_cta_sim_param.jl PHANTOM_DIR OUTPUT_DIR [--kvp 100] [--mA 250] [--no-dicom]`
writes the FBP recon + `recon_meta.toml` immediately after FBP completes (HIR
recon is best-effort; FBP is the canonical recon used downstream).

The recon is half-scale (BasisSim renders the phantom at 0.4 mm); the recon-
space myo/territory masks are obtained by reverse(y,z) + downsample-2 +
resample (see `recover_perfusion.py:to_recon`). NumPy masks written for the
Julia estimators must use `ravel(order="F")` (Julia reads column-major).

---

## References

- Hubbard L, Molloi S, et al. Dynamic CT myocardial perfusion — single-frame
  ΔHU / AIF-integral method (`MolloiLab/PerfusionImagingNotebooks`).
- Mullani NA, Gould KL. "First-pass measurements of regional blood flow with
  external detectors." J Nucl Med 1983;24:577-81 (max-slope estimator).
- `../FlowContrastSim.jl/README.md` — hemodynamics + contrast PDE.
- `../VascularTreeSim.jl/README.md` — tree grower + emergent voxelizer.
