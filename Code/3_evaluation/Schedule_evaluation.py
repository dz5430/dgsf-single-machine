# -*- coding: utf-8 -*-
"""
Schedule evaluations:
1) Optimality gap (%) of a method vs a reference objective
2) Count of optimal/tied solutions within tolerance
3) Spearman's rho between predicted and reference rank vectors

Required Excel columns:
- METHOD_OBJ_COL (e.g., tardiness from your method)
- REF_OBJ_COL    (e.g., tardiness from discrete-time reference)
- PRED_RANK_COL  = "rank_dgsf_mip"
- REF_RANK_COL   = "rank_vector_dtime"

Rank-vector cells can be stored as:
- Python-like lists: "[3, 1, 2, ...]"
- Or comma/space separated strings: "3,1,2" / "3 1 2"
"""

import ast
import argparse
import numpy as np
import pandas as pd

# ----------------------------
# Config
# ----------------------------
INPUT_XLSX = "schedule_results.xlsx"

METHOD_OBJ_COL = "tardiness_dgsf_mip"
REF_OBJ_COL = "tardiness_dtime"

PRED_RANK_COL = "rank_dgsf_mip"
REF_RANK_COL = "rank_vector_dtime"

TOL = 1e-12


# ----------------------------
# Helpers
# ----------------------------
def to_float_array(s: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    if np.any(np.isnan(arr)):
        bad = np.where(np.isnan(arr))[0][:10]
        raise ValueError(f"Found NaNs in numeric column '{s.name}'. First bad indices: {bad.tolist()}")
    return arr


def parse_rank_vector(x):
    """Parse a cell containing a list-like rank vector."""
    if isinstance(x, (list, tuple, np.ndarray)):
        v = np.array(x, dtype=float).flatten()
        return v

    if pd.isna(x):
        return None

    s = str(x).strip()
    if not s:
        return None

    # Try Python literal first
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, np.ndarray)):
            return np.array(v, dtype=float).flatten()
    except Exception:
        pass

    # Fallback: comma/space separated
    s = s.strip("[]()")
    parts = [p for p in s.replace(",", " ").split() if p]
    try:
        return np.array([float(p) for p in parts], dtype=float).flatten()
    except Exception:
        return None


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman's rho via Pearson correlation of average ranks (handles ties).
    """
    if a.size != b.size or a.size == 0:
        return np.nan

    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()

    if np.allclose(ra, ra[0]) or np.allclose(rb, rb[0]):
        return np.nan

    return float(np.corrcoef(ra, rb)[0, 1])


# ----------------------------
# Main
# ----------------------------
def main(
    input_xlsx: str = INPUT_XLSX,
    method_obj_col: str = METHOD_OBJ_COL,
    ref_obj_col: str = REF_OBJ_COL,
    pred_rank_col: str = PRED_RANK_COL,
    ref_rank_col: str = REF_RANK_COL,
) -> None:
    df = pd.read_excel(input_xlsx)

    # ---- Gap stats ----
    method_obj = to_float_array(df[method_obj_col])
    ref_obj = to_float_array(df[ref_obj_col])

    denom = np.where(np.abs(ref_obj) <= TOL, np.nan, ref_obj)
    gap_pct = 100.0 * (method_obj - ref_obj) / denom

    gap_mean = float(np.nanmean(gap_pct))
    gap_std = float(np.nanstd(gap_pct))
    gap_med = float(np.nanmedian(gap_pct))
    p90 = float(np.nanpercentile(gap_pct, 90))
    p95 = float(np.nanpercentile(gap_pct, 95))

    optimal_count = int(np.sum(method_obj <= ref_obj + TOL))
    total_n = int(len(method_obj))

    print("========================================")
    print("Objective / Gap Summary")
    print("========================================")
    print(f"Method objective col:    {method_obj_col}")
    print(f"Reference objective col: {ref_obj_col}")
    print("")
    print("Gap % = 100*(method - ref)/ref")
    print(f"Mean ± Std: {gap_mean:.4f}% ± {gap_std:.4f}%")
    print(f"Median:     {gap_med:.4f}%")
    print(f"P90 / P95:  {p90:.4f}% / {p95:.4f}%")
    print(f"Optimal (<= ref + tol): {optimal_count} / {total_n}")
    print("")

    # ---- Spearman rho ----
    if pred_rank_col not in df.columns:
        raise KeyError(f"Missing predicted rank column: {pred_rank_col}")
    if ref_rank_col not in df.columns:
        raise KeyError(f"Missing reference rank column: {ref_rank_col}")

    rhos = []
    for i in range(total_n):
        pred_vec = parse_rank_vector(df.loc[i, pred_rank_col])
        ref_vec = parse_rank_vector(df.loc[i, ref_rank_col])

        if pred_vec is None or ref_vec is None:
            rhos.append(np.nan)
            continue

        m = min(pred_vec.size, ref_vec.size)
        if m == 0:
            rhos.append(np.nan)
            continue

        rhos.append(spearman_rho(pred_vec[:m], ref_vec[:m]))

    rhos = np.array(rhos, dtype=float)
    valid_n = int(np.sum(~np.isnan(rhos)))
    rho_mean = float(np.nanmean(rhos))
    rho_std = float(np.nanstd(rhos))

    print("========================================")
    print("Spearman's rho (rank vectors)")
    print("========================================")
    print(f"Pred rank col: {pred_rank_col}")
    print(f"Ref  rank col: {ref_rank_col}")
    print(f"Valid instances: {valid_n} / {total_n}")
    print(f"Mean ± Std: {rho_mean:.4f} ± {rho_std:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate objective gaps and rank-vector similarity.")
    parser.add_argument("--input", default=INPUT_XLSX, help="Input Excel file.")
    parser.add_argument("--method-obj-col", default=METHOD_OBJ_COL, help="Method objective column.")
    parser.add_argument("--ref-obj-col", default=REF_OBJ_COL, help="Reference objective column.")
    parser.add_argument("--pred-rank-col", default=PRED_RANK_COL, help="Predicted rank-vector column.")
    parser.add_argument("--ref-rank-col", default=REF_RANK_COL, help="Reference rank-vector column.")
    args = parser.parse_args()
    main(args.input, args.method_obj_col, args.ref_obj_col, args.pred_rank_col, args.ref_rank_col)
