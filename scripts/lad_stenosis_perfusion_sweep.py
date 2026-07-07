#!/usr/bin/env python3
# lad_stenosis_perfusion_sweep.py — perfusion-measured LAD flow vs stenosis.
# For each stenosis %, scale the LAD-territory myocardial iodine in the baseline
# flow-proportional stress V2 phantom to C_myo = (Q_LAD/mass)·∫AIF·ρ/60, re-scan
# (BasisSim), Mullani-Gould, recover measured LAD perfusion. LCX/RCA untouched.
import numpy as np, os, subprocess, re, shutil, sys, time

PP="/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline"
ROOT="/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation"
RBS=f"{ROOT}/phantom_ct_input/run_basis_sim/run_cta_sim_param.jl"
RBS_PROJ=f"{ROOT}/phantom_ct_input/run_basis_sim"
BASE_V2=f"{PP}/basissim_phantoms_fp_stress/v2_peak"
V1_DIR=f"{PP}/basissim_out_bae/v1_baseline"
AIF=f"{PP}/intermediate/aif_curve_bae.csv"
TREE_ID=f"{PP}/intermediate/myo_tree_id.raw"
CUMAUC=95.143; RHO=1.053; MASS={1:58.9,2:60.9,3:63.8}; BASE100=10255
GPUS=["GPU-7bc57a3a-7255-7da2-b5df-48575f12f936","GPU-cae9ac13-6ef2-703b-0e19-6d8125f4d952","GPU-d5879c1b-ba43-2cf0-b650-b823c3f17e1f"]
NX,NY,NZ=1600,1400,500; DOWN=2; BSX,BSY,BSZ=800,700,250; BSVOX=0.2; VX,VZ=0.6836,0.2

qlad={}
for line in open(f"{PP}/output/lad_flowsim_sweep.csv"):
    if line.startswith("stenosis"): continue
    p,q,r=line.split(","); qlad[int(p)]=float(q)
imax=float(re.search(r"iodine_max_mg_per_mL = ([\d.]+)",open(f"{BASE_V2}/phantom_manifest.toml").read()).group(1))
print(f"iodine_max={imax:.2f}  Q_LAD(0%)={qlad[0]:.0f}  Q_LAD(90%)={qlad[90]:.0f}")

print("loading baseline phantom + tree_id …"); sys.stdout.flush()
base_ph=np.fromfile(f"{BASE_V2}/phantom.raw",dtype=np.uint16)
tid_flat=np.fromfile(TREE_ID,dtype=np.uint8)
lad_mask=tid_flat==1
def lad_label(pct):
    P=qlad[pct]/MASS[1]; cmyo=P*CUMAUC*RHO/60.0
    return BASE100+int(round(np.clip(cmyo/imax*100,0,100)))

# tree_id → recon grid (for per-territory recovery), built once
def treeid_recon():
    tid=tid_flat.reshape((NX,NY,NZ),order="F")[:,::-1,::-1][1::DOWN,1::DOWN,1::DOWN]
    sx=sz=None
    nxr,nyr,nzr=512,512,250
    sxv,szv=VX/BSVOX,VZ/BSVOX
    phc=[BSX/2-.5,BSY/2-.5,BSZ/2-.5]; rc=[nxr/2-.5,nyr/2-.5,nzr/2-.5]
    ii=np.rint((np.arange(nxr)-rc[0])*sxv+phc[0]).astype(int)
    jj=np.rint((np.arange(nyr)-rc[1])*sxv+phc[1]).astype(int)
    kk=np.rint((np.arange(nzr)-rc[2])*szv+phc[2]).astype(int)
    ok=lambda a,n:(a>=0)&(a<n)
    tr=tid[np.clip(ii,0,BSX-1)[:,None,None],np.clip(jj,0,BSY-1)[None,:,None],np.clip(kk,0,BSZ-1)[None,None,:]]
    valid=ok(ii,BSX)[:,None,None]&ok(jj,BSY)[None,:,None]&ok(kk,BSZ)[None,None,:]
    return np.where(valid,tr,0).astype(np.uint8)
TID_R=treeid_recon()

PCTS=[0,10,20,30,40,50,60,70,80,90]
results={}
os.makedirs(f"{PP}/sweep_tmp",exist_ok=True); os.makedirs(f"{PP}/sweep_out",exist_ok=True)
for i in range(0,len(PCTS),3):
    batch=PCTS[i:i+3]; procs=[]
    for j,pct in enumerate(batch):
        tmp=f"{PP}/sweep_tmp/st{pct}"; os.makedirs(tmp,exist_ok=True)
        ph=base_ph.copy(); ph[lad_mask]=lad_label(pct); ph.tofile(f"{tmp}/phantom.raw")
        shutil.copy(f"{BASE_V2}/phantom_manifest.toml",f"{tmp}/phantom_manifest.toml")
        out=f"{PP}/sweep_out/st{pct}"
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[j])
        p=subprocess.Popen(["julia",f"--project={RBS_PROJ}",RBS,tmp,out,"--kvp","100","--mA","250","--no-dicom"],
                           env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        procs.append((pct,p,tmp,out)); print(f"[st{pct}] BasisSim launched on GPU{j}"); sys.stdout.flush()
    for pct,p,tmp,out in procs:
        p.wait()
        # step5
        subprocess.run(["python3",f"{PP}/scripts/step5_perfusion_from_recon.py",AIF,V1_DIR,out,
                        "--v2_t","20","--recon","fbp","--out_suffix",f"_st{pct}"],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        perf=np.load(f"{PP}/output/perfusion_map_st{pct}.npy")
        m=(TID_R==1)&(perf>0); mLAD=float(perf[m].mean())
        results[pct]=mLAD
        print(f"[st{pct}] Q_LAD_sim={qlad[pct]:.0f} mL/min | measured LAD perf={mLAD:.3f} mL/min/g = {mLAD*MASS[1]:.0f} mL/min"); sys.stdout.flush()
        shutil.rmtree(tmp,ignore_errors=True)
        for fn in ("recon_fbp_hu_f32.raw","recon_hir_hu_f32.raw"):
            try: os.remove(f"{out}/{fn}")
            except: pass

with open(f"{PP}/output/lad_stenosis_sweep_result.csv","w") as f:
    f.write("stenosis_pct,Q_LAD_sim_mlmin,measured_LAD_perf_mlming,measured_LAD_flow_mlmin\n")
    for pct in PCTS:
        f.write(f"{pct},{qlad[pct]:.2f},{results[pct]:.4f},{results[pct]*MASS[1]:.2f}\n")
print("DONE → output/lad_stenosis_sweep_result.csv")
