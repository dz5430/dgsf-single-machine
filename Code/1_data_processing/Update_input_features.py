# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import ast
import argparse

def _rank01(x):
    I = len(x)
    if I <= 1:
        return np.full(I, 0.5)
    return np.argsort(np.argsort(x)) / (I - 1)

def _static_atc(rho, tau, eps, k=2.0, t=0.0, eps_num=1e-8):
    """ATC(t) = (1/τ) * exp(- max(0, ε - τ - t) / (k * mean(τ)))"""
    tau_bar = np.mean(tau)
    buffer = np.maximum(eps - tau - t, 0.0)
    denom = max(k * tau_bar, eps_num)
    atc = (1.0 / np.maximum(tau, eps_num)) * np.exp(-buffer / denom)
    return atc

def calculate_lp_free_features(rho, tau, eps, k_atc=2.0):
    """
    Compact, LP-free, non-leaky features (9 total).
    Returns a dict of numpy arrays with descriptive names.
    """
    rho = np.asarray(rho, dtype=float).flatten()
    tau = np.asarray(tau, dtype=float).flatten()
    eps = np.asarray(eps, dtype=float).flatten()
    I = len(rho)
    tau_bar = np.mean(tau)

    # Core basics
    slack = eps - rho - tau
    rel_proc_time = tau / tau_bar
    window_tightness_jobs = (eps - rho) / np.maximum(tau, 1e-8)
    slack_time = slack

    # Percentiles (ranks 0..1)
    release_percentile = _rank01(rho)
    due_percentile = _rank01(eps)

    # ATC signals (ranked; robust, scale-free)
    atc_t0 = _static_atc(rho, tau, eps, k=k_atc, t=0.0)
    atc_trho = _static_atc(rho, tau, eps, k=k_atc, t=rho)
    atc_t0_rank = _rank01(atc_t0)
    atc_at_release_rank = _rank01(atc_trho)

    # Local backlog pressure before each job's due date
    # load_before_due = sum_j tau_j where rho_j < eps_i
    # O(N^2) but fine for N ≤ few hundreds
    load_before_due = np.array([tau[rho < eps[i]].sum() for i in range(I)], dtype=float)
    load_before_due_norm = load_before_due / (I * tau_bar + 1e-8)

    # Myopic lateness if you start at your own release
    myopic_lateness = np.maximum(0.0, rho + tau - eps) / (tau_bar + 1e-8)

    return {
        "rel_proc_time": rel_proc_time,
        "window_tightness_jobs": window_tightness_jobs,
        "slack_time": slack_time,
        "release_percentile": release_percentile,
        "due_percentile": due_percentile,
        "atc_t0_rank": atc_t0_rank,
        "atc_at_release_rank": atc_at_release_rank,
        "load_before_due_norm": load_before_due_norm,
        "myopic_lateness": myopic_lateness,
    }

def process_excel_file(input_file, output_file):
    print(f"Loading data from {input_file}...")
    df = pd.read_excel(input_file)
    out_rows = []

    for idx, row in df.iterrows():
        try:
            I = int(row["I"])
            rho = np.array(ast.literal_eval(str(row["rho"])) ).flatten()
            tau = np.array(ast.literal_eval(str(row["tau"])) ).flatten()
            eps = np.array(ast.literal_eval(str(row["eps"])) ).flatten()

            feats = calculate_lp_free_features(rho, tau, eps, k_atc=2.0)

            entry = {
                "I": I,
                "rho": rho.tolist(),           # keep raw primitives you already use elsewhere
                "tau": tau.tolist(),
                "eps": eps.tolist(),

                # --- compact feature set (9) ---
                "rel_proc_time": feats["rel_proc_time"].tolist(),
                "window_tightness_jobs": feats["window_tightness_jobs"].tolist(),
                "slack_time": feats["slack_time"].tolist(),
                "release_percentile": feats["release_percentile"].tolist(),
                "due_percentile": feats["due_percentile"].tolist(),
                "atc_t0_rank": feats["atc_t0_rank"].tolist(),
                "atc_at_release_rank": feats["atc_at_release_rank"].tolist(),
                "load_before_due_norm": feats["load_before_due_norm"].tolist(),
                "myopic_lateness": feats["myopic_lateness"].tolist(),
            }

            # preserve any other metadata columns the row already had (e.g., labels)
            for col in df.columns:
                if col not in entry:
                    entry[col] = row[col]

            out_rows.append(entry)

        except Exception as e:
            print(f"❌ Error at instance {idx+1}: {e}")
            out_rows.append(row.to_dict())

    out_df = pd.DataFrame(out_rows)
    print(f"Saving data to {output_file}...")
    out_df.to_excel(output_file, index=False)
    print("Features saved!")
    return out_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DGSF input features.")
    parser.add_argument("--input", default="Raw_file.xlsx", help="Input Excel file.")
    parser.add_argument("--output", default="Updated_file.xlsx", help="Output Excel file.")
    args = parser.parse_args()

    process_excel_file(args.input, args.output)
