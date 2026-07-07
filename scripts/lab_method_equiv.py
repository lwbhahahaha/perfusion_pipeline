#!/usr/bin/env python3
# lab_method_equiv.py — run the lab's PerfusionImaging.compute_organ_metrics
# (verbatim) on our simulated V1/V2 recons + myo mask, to confirm it matches
# our step5 Mullani-Gould output (proves the two implementations are equivalent).
import numpy as np, os

PP="/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline"
SPACING=(0.6836,0.6836,0.2)      # recon voxel mm (fov 35 / 512, z 5/250)
AUC_HUS=3329.9911                 # AIF AUC(0..20s) in HU·s (== step5)

def load_recon(d):
    p=os.path.join(PP,d,"recon_fbp_hu_f32.raw")
    return np.fromfile(p,dtype=np.float32).reshape((512,512,250),order="F").astype(float)

class Img(np.ndarray):                       # ndarray + .spacing (mimics ANTs image)
    def __new__(cls,a,spacing):
        o=np.asarray(a,dtype=float).view(cls); o.spacing=spacing; return o
    def __array_finalize__(self,obj):
        if obj is None: return
        self.spacing=getattr(obj,"spacing",None)

# ---- VERBATIM from PerfusionImaging/tool.py:compute_organ_metrics ----
def compute_organ_metrics(dcm_rest, dcm_mask_rest, v1, input_conc, tissue_rho=1.053):
    dcm_mask_rest = dcm_mask_rest[:].astype(bool)
    try:
        v1_arr = dcm_rest.copy(); v1_arr[dcm_mask_rest] = v1
    except:
        v1_arr = v1.copy()
    voxel_size = dcm_rest.spacing
    v1_arr[~dcm_mask_rest[:].astype(bool)] = np.nan
    dcm_rest[~dcm_mask_rest[:].astype(bool)] = np.nan
    organ_mass = (np.sum(dcm_mask_rest[:]) * tissue_rho * voxel_size[0]*voxel_size[1]*voxel_size[2]/1000)
    delta_hu = np.mean(dcm_rest[dcm_mask_rest]) - np.mean(v1_arr[dcm_mask_rest])
    organ_vol_inplane = voxel_size[0]*voxel_size[1]*voxel_size[2]/1000
    v1_mass = np.sum(v1_arr[dcm_mask_rest])*organ_vol_inplane
    v2_mass = np.sum(dcm_rest[dcm_mask_rest])*organ_vol_inplane
    flow = (60/input_conc)*(v2_mass - v1_mass)
    flow_map = (dcm_rest - v1_arr)/(np.mean(dcm_rest[dcm_mask_rest]) - np.mean(v1_arr[dcm_mask_rest]))*flow
    perf = flow/organ_mass
    return {"organ_mass":organ_mass,"delta_hu":delta_hu,"v1_mass":v1_mass,"v2_mass":v2_mass,"flow":flow,"perf":perf}
# ----------------------------------------------------------------------

V1=load_recon("basissim_out_bae/v1_baseline")
step5={"REST":dict(flow=24.5306,perf=1.070941,dhu=62.5872),
       "STRESS":dict(flow=29.5793,perf=1.291353,dhu=75.4684)}
print(f"{'state':>7} | {'src':>6} {'organ_mass':>10} {'delta_HU':>9} {'flow(mL/min)':>13} {'perf(mL/min/g)':>15}")
for state,v2dir,maskf in (("REST","basissim_out_bae_rest/v2_peak","myo_mask_bae_rest.npy"),
                          ("STRESS","basissim_out_bae_stress/v2_peak","myo_mask_bae_stress.npy")):
    V2=Img(load_recon(v2dir),SPACING)
    mask=np.load(os.path.join(PP,"output",maskf))
    m=compute_organ_metrics(V2, mask, V1.copy(), AUC_HUS)
    s=step5[state]
    print(f"{state:>7} | {'LAB':>6} {m['organ_mass']:>10.3f} {m['delta_hu']:>9.3f} {m['flow']:>13.3f} {m['perf']:>15.4f}")
    print(f"{'':>7} | {'step5':>6} {'22.906':>10} {s['dhu']:>9.3f} {s['flow']:>13.3f} {s['perf']:>15.4f}")
    print(f"{'':>7} | {'Δ%':>6} {'':>10} {100*(m['delta_hu']-s['dhu'])/s['dhu']:>8.2f}% {100*(m['flow']-s['flow'])/s['flow']:>12.2f}% {100*(m['perf']-s['perf'])/s['perf']:>14.2f}%")
