#!/usr/bin/env python3
# per_territory_recovery.py — split a recon-space perfusion map by coronary
# territory (LAD/LCX/RCA) and report recovered per-artery perfusion (mL/min/g)
# and absolute flow (mL/min), for the closed-loop vs the hemo input root_flow.
#
# Usage: per_territory_recovery.py PERFUSION_MAP.npy [--vx 0.6836 --vz 0.2]
import numpy as np, sys, os, argparse

NX,NY,NZ=1600,1400,500; DOWN=2; BSX,BSY,BSZ=800,700,250; BSVOX=0.2
TREE_ID_RAW="/media/molloi-lab/2TB3/wenbo playground/flow simulation tree generation/perfusion_pipeline/intermediate/myo_tree_id.raw"
TERRITORY_MASS_G={1:58.9,2:60.9,3:63.8}   # LAD/LCX/RCA true-scale myo mass
NAME={1:"LAD",2:"LCX",3:"RCA"}

def treeid_to_recon(recon_shape, vx_mm, vz_mm):
    """Map myo_tree_id (1600³) through reverse(y,z)+downsample-2 onto recon grid (NN)."""
    tid=np.fromfile(TREE_ID_RAW,dtype=np.uint8).reshape((NX,NY,NZ),order="F")
    tid=tid[:,::-1,::-1][1::DOWN,1::DOWN,1::DOWN]
    assert tid.shape==(BSX,BSY,BSZ), tid.shape
    nxr,nyr,nzr=recon_shape
    sx,sy,sz=vx_mm/BSVOX,vx_mm/BSVOX,vz_mm/BSVOX
    phcx,phcy,phcz=BSX/2-.5,BSY/2-.5,BSZ/2-.5
    rcx,rcy,rcz=nxr/2-.5,nyr/2-.5,nzr/2-.5
    ii=np.rint((np.arange(nxr)-rcx)*sx+phcx).astype(int)
    jj=np.rint((np.arange(nyr)-rcy)*sy+phcy).astype(int)
    kk=np.rint((np.arange(nzr)-rcz)*sz+phcz).astype(int)
    iok=(ii>=0)&(ii<BSX); jok=(jj>=0)&(jj<BSY); kok=(kk>=0)&(kk<BSZ)
    tr=tid[np.clip(ii,0,BSX-1)[:,None,None],np.clip(jj,0,BSY-1)[None,:,None],np.clip(kk,0,BSZ-1)[None,None,:]]
    valid=iok[:,None,None]&jok[None,:,None]&kok[None,None,:]
    tr=np.where(valid,tr,0)
    return tr.astype(np.uint8)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("perfusion_map"); ap.add_argument("--vx",type=float,default=0.6836); ap.add_argument("--vz",type=float,default=0.2)
    a=ap.parse_args()
    perf=np.load(a.perfusion_map)              # mL/min/g, recon space
    print(f"perfusion map {perf.shape}, nonzero voxels {int((perf>0).sum())}")
    tid=treeid_to_recon(perf.shape,a.vx,a.vz)
    print(f"recon voxel = {a.vx}×{a.vx}×{a.vz} mm³")
    print(f"\n{'terr':>5} {'voxels':>8} {'perf_mean':>10} {'perf_med':>9}  {'flow=perf×mass(mL/min)':>22}")
    tot_flow=0.0
    for t in (1,2,3):
        m=(tid==t)&(perf>0)
        n=int(m.sum())
        if n==0: print(f"{NAME[t]:>5} {n:>8} (no voxels)"); continue
        pm=float(perf[m].mean()); pmd=float(np.median(perf[m]))
        flow=pm*TERRITORY_MASS_G[t]            # mL/min/g × g = mL/min (true-scale mass)
        tot_flow+=flow
        print(f"{NAME[t]:>5} {n:>8} {pm:>10.3f} {pmd:>9.3f}  {flow:>22.1f}")
    print(f"{'TOTAL':>5} {'':>8} {'':>10} {'':>9}  {tot_flow:>22.1f}")
    print("\n(recovered flow = territory mean perfusion × true-scale territory mass)")
    print("compare to hemo input root_flow per artery)")

if __name__=="__main__": main()
