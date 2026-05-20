#!/usr/bin/env python3
"""
Step 0b: extend a baseline-subtracted AIF curve (output of step0_prepare_aif.py)
by fitting a gamma-variate to the measured wash-in samples and continuing the
fitted curve through the wash-out tail that was not captured by bolus
tracking (which typically stops at SureStart trigger ≈ AIF peak).

Why: when the bolus tracker truncates at peak, the measured AUC under-represents
the true bolus pass. Mullani-Gould perfusion ∝ 60/AUC, so a truncated AUC over-
estimates flow. Extrapolating the wash-out gives a more representative AUC and
brings the CFR estimate closer to clinical values.

Gamma model:
    C(t) = amp · ((t-t0)/(tmax-t0))^α · exp(α · (1 - (t-t0)/(tmax-t0)))   t > t0
         = 0                                                              t ≤ t0

The 4 parameters (amp, t0, tmax, α) are fitted to the measured C(t) samples
via non-linear least squares (scipy.optimize.curve_fit). The fitted curve is
then sampled on a dense uniform grid out to T_END_S (default 30 s) and written
to a CSV in the same format as step0_prepare_aif.py.

Usage:
    python3 step0b_extrapolate_aif.py  INPUT_CSV  OUTPUT_CSV  [--t_end 30] [--dt 0.1]
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import curve_fit


def gamma_variate(t, amp, t0, tmax, alpha):
    out = np.zeros_like(t, dtype=np.float64)
    mask = t > t0
    tt = t[mask]
    tp = (tt - t0) / max(tmax - t0, 1e-6)
    valid = tp > 0
    out[mask] = np.where(
        valid,
        amp * np.power(tp, alpha) * np.exp(alpha * (1.0 - tp)),
        0.0,
    )
    return out


def load_csv(path):
    ts, cs = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("time_s"):
                continue
            parts = s.split(",")
            if len(parts) < 2:
                continue
            ts.append(float(parts[0]))
            cs.append(float(parts[1]))
    return np.asarray(ts), np.asarray(cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="output of step0_prepare_aif.py")
    ap.add_argument("output_csv", help="extrapolated AIF CSV")
    ap.add_argument("--t_end", type=float, default=30.0,
                    help="extrapolate up to this time (s); default 30")
    ap.add_argument("--dt", type=float, default=0.1, help="output sample step (s)")
    args = ap.parse_args()

    t, c = load_csv(args.input_csv)
    t_end_meas = float(t[-1])
    peak_idx = int(np.argmax(c))
    print(f"[step0b] loaded {len(t)} samples  t ∈ [{t[0]:.3f}, {t_end_meas:.3f}] s  C_max = {c.max():.4f} mg/mL @ t={t[peak_idx]:.3f}")

    # Initial guesses + bounds (physiologically constrained):
    #   amp must stay close to observed peak (within 1.3× — clinical bolus
    #   doesn't sneak past tracker by >30%, since the tracker fires at peak).
    #   tmax not too far past tracker stop (≤ 5 s — typical post-trigger delay).
    #   alpha 1–5 (gamma shape characteristic of IV contrast bolus).
    c_max_obs = float(c.max())
    amp0   = c_max_obs * 1.05
    tmax0  = float(t[peak_idx])
    t0_0   = max(0.0, tmax0 - 5.0)   # bolus rise ≈ 5 s
    alpha0 = 3.0
    p0 = [amp0, t0_0, tmax0, alpha0]

    lo = [0.5 * c_max_obs, 0.0,            max(tmax0 - 1.0, 0.5),  1.0]
    hi = [1.3 * c_max_obs, max(t0_0, 0.5), tmax0 + 5.0,            5.0]

    try:
        popt, pcov = curve_fit(gamma_variate, t, c, p0=p0,
                                bounds=(lo, hi), maxfev=20000)
        amp, t0, tmax, alpha = popt
        print(f"[step0b] fitted: amp={amp:.4f}  t0={t0:.3f}  tmax={tmax:.3f}  α={alpha:.3f}")
    except Exception as e:
        print(f"[step0b] WARN: gamma fit failed ({e}); falling back to last sample tail",
              file=sys.stderr)
        amp = float(c.max()); t0 = 0.0; tmax = tmax0; alpha = 3.0

    # Build extrapolated curve on dense grid
    t_grid = np.arange(0.0, args.t_end + args.dt / 2, args.dt)
    c_grid = gamma_variate(t_grid, amp, t0, tmax, alpha)

    # Where measured data exists, prefer measured (smooth the join near end)
    # Strategy: use measured for t ≤ t_end_meas, fitted for t > t_end_meas.
    in_meas = t_grid <= t_end_meas
    c_meas = np.interp(t_grid[in_meas], t, c, left=0.0, right=c[-1])
    c_blend = c_grid.copy()
    c_blend[in_meas] = c_meas
    # Light smoothing at the join
    join_idx = int(np.searchsorted(t_grid, t_end_meas))
    if 0 < join_idx < len(t_grid) - 5:
        # linear blend over 1 s window
        n_blend = max(1, int(1.0 / args.dt))
        for k in range(n_blend):
            i = join_idx + k
            if i >= len(t_grid):
                break
            w = (k + 1) / (n_blend + 1)
            c_blend[i] = (1 - w) * c_meas[-1] + w * c_grid[i] if i < len(c_meas) else c_grid[i]

    trapz_fn = getattr(np, "trapezoid", np.trapz)
    auc_meas = float(trapz_fn(c, t))
    auc_full = float(trapz_fn(c_blend, t_grid))
    print(f"[step0b] AUC measured (raw, 0..{t_end_meas:.1f}s) = {auc_meas:.4f} mg·s/mL")
    print(f"[step0b] AUC extrapolated (gamma, 0..{args.t_end:.1f}s) = {auc_full:.4f} mg·s/mL  "
          f"({auc_full/max(auc_meas,1e-9):.2f}×)")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w") as f:
        f.write(f"# Extrapolated AIF (gamma-variate fit + wash-out continuation)\n")
        f.write(f"# source = {args.input_csv}\n")
        f.write(f"# gamma fit: amp={amp:.4f}, t0={t0:.4f}, tmax={tmax:.4f}, alpha={alpha:.4f}\n")
        f.write(f"# t_end_meas = {t_end_meas:.4f} s\n")
        f.write(f"# t_end_extrap = {args.t_end:.4f} s\n")
        f.write(f"# AUC (measured)     = {auc_meas:.6f} mg·s/mL\n")
        f.write(f"# AUC (extrapolated) = {auc_full:.6f} mg·s/mL\n")
        f.write("time_s,C_mg_per_mL\n")
        for ti, ci in zip(t_grid, c_blend):
            f.write(f"{ti:.4f},{max(ci, 0.0):.6f}\n")
    print(f"[step0b] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
