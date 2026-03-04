# -*- coding: utf-8 -*-
"""
Recover instance characteristics from (rho, tau, eps).

- rho_delta: max|rho_i - rho_perfect_i| where rho_perfect_i = b*sort_rank_i
- eps_delta: eps_delta_total - rho_delta, rounded to 2dp, non-negative       
- sigma:     mean(eps_i - rho_perfect_i)

"""

import ast
import math
import numpy as np
import pandas as pd

INPUT_XLSX  = "Raw_file.xlsx"
OUTPUT_XLSX = "Instances_characteristics.xlsx"

EPS_B_TOL = 1e-12

def parse_1d(cell) -> np.ndarray:
    return np.array(ast.literal_eval(str(cell)), dtype=float).flatten()


def fit_step_through_origin(r: np.ndarray, rho: np.ndarray) -> float:
    """Fit rho ≈ b*r with intercept = 0: b = (r·rho)/(r·r)."""
    r = r.astype(float)
    denom = float(np.dot(r, r))
    if denom <= 0:
        return 0.0
    return float(np.dot(r, rho) / denom)


def main() -> None:
    df = pd.read_excel(INPUT_XLSX)

    theta_deg_list                = []
    sigma_mult_I_tauavg_list      = []
    sigma_mult_tauavg_list        = []
    rho_delta_mult_I_tauavg_list  = []
    rho_delta_mult_tauavg_list    = []
    eps_delta_mult_I_tauavg_list  = []
    eps_delta_mult_tauavg_list    = []

    print("Testing (first 5 instances):")
    print("-" * 130)
    print(
        "Format: theta_tilt_deg | sigma=(x)*(I*tau_avg)=(y)*tau_avg | "
        "rho_delta=(x)*(I*tau_avg)=(y)*tau_avg | eps_delta=(x)*(I*tau_avg)=(y)*tau_avg"
    )
    print("-" * 130)

    for idx, row in df.iterrows():
        I   = int(row["I"])
        rho = parse_1d(row["rho"]).astype(float)
        tau = parse_1d(row["tau"]).astype(float)
        eps = parse_1d(row["eps"]).astype(float)

        tau_avg   = float(np.mean(tau))
        denom_I   = max(I * tau_avg, 1e-12)
        denom_tau = max(tau_avg, 1e-12)

        # --- Step 1: fit stagger step b using sorted rho ---
        sort_order  = np.argsort(rho, kind="mergesort")
        lattice_idx = np.arange(I, dtype=float)
        rho_sorted  = rho[sort_order]
        b = fit_step_through_origin(lattice_idx, rho_sorted)

        # --- Step 2: perfect lattice in sorted space, mapped back to original order ---
        rho_perfect_sorted = b * lattice_idx
        inv_sort = np.empty(I, dtype=int)
        inv_sort[sort_order] = np.arange(I, dtype=int)
        rho_perfect = rho_perfect_sorted[inv_sort]

        # --- Step 3: rho_delta ---
        rho_delta = float(np.max(np.abs(rho_sorted - rho_perfect_sorted)))

        # --- Step 4: sigma ---
        sigma = float(np.mean(eps - rho_perfect))

        # --- Step 5: eps_delta = total deviation - rho noise, rounded, non-negative ---
        eps_delta_total = float(np.max(np.abs(eps - (rho_perfect + sigma))))
        eps_delta = max(0.0, round(eps_delta_total - rho_delta, 2))

        # --- theta tilt ---
        theta      = math.pi / 2.0 if abs(b) <= EPS_B_TOL else float(math.atan(1.0 / b))
        theta_tilt = (math.pi / 2.0) - theta
        theta_deg  = round(float(theta_tilt * 180.0 / math.pi), 2)

        # --- normalise, rounded to 2dp ---
        sigma_mult_I   = round(sigma     / denom_I,   2)
        sigma_mult_tau = round(sigma     / denom_tau, 2)
        rho_d_mult_I   = round(rho_delta / denom_I,   2)
        rho_d_mult_tau = round(rho_delta / denom_tau, 2)
        eps_d_mult_I   = round(eps_delta / denom_I,   2)
        eps_d_mult_tau = round(eps_delta / denom_tau, 2)

        # store
        theta_deg_list.append(theta_deg)
        sigma_mult_I_tauavg_list.append(sigma_mult_I)
        sigma_mult_tauavg_list.append(sigma_mult_tau)
        rho_delta_mult_I_tauavg_list.append(rho_d_mult_I)
        rho_delta_mult_tauavg_list.append(rho_d_mult_tau)
        eps_delta_mult_I_tauavg_list.append(eps_d_mult_I)
        eps_delta_mult_tauavg_list.append(eps_d_mult_tau)

        if idx < 5:
            print(
                f"Instance {idx+1}: "
                f"{theta_deg:8.2f}° | "
                f"sigma={sigma_mult_I:6.2f}*(I*tau_avg)={sigma_mult_tau:6.2f}*tau_avg | "
                f"rho_delta={rho_d_mult_I:6.2f}*(I*tau_avg)={rho_d_mult_tau:6.2f}*tau_avg | "
                f"eps_delta={eps_d_mult_I:6.2f}*(I*tau_avg)={eps_d_mult_tau:6.2f}*tau_avg"
            )

    out = pd.DataFrame({
        "theta_deg":                   theta_deg_list,
        "sigma_mult_I_tauavg":         sigma_mult_I_tauavg_list,
        "sigma_mult_tauavg":           sigma_mult_tauavg_list,
        "rho_delta_mult_I_tauavg":     rho_delta_mult_I_tauavg_list,
        "rho_delta_mult_tauavg":       rho_delta_mult_tauavg_list,
        "eps_delta_mult_I_tauavg":     eps_delta_mult_I_tauavg_list,
        "eps_delta_mult_tauavg":       eps_delta_mult_tauavg_list,
    })

    out.to_excel(OUTPUT_XLSX, index=False)
    print(f"\nSaved to {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()