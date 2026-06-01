# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import ast
import time
import argparse
from MIP_ContTime_Post_Tight import single_machine_ct_post_tight # returns (model, rank_idx, ml_order)

INPUT_XLSX = "Post_ML_file.xlsx" # Update with own file
OUTPUT_XLSX = "Final_prediction.xlsx" # Update with own file
TIME_LIMIT_SEC = 60

PRINT_SOLVER_LOG = True
SAVE_RESULTS = True

rank_vectors = []
start_time_vectors = []
solve_times = []
tardiness_values = []
termination_conditions = []

parser = argparse.ArgumentParser(description="Run continuous-time MIP post-processing.")
parser.add_argument("--input", default=INPUT_XLSX, help="Input Excel file.")
parser.add_argument("--output", default=OUTPUT_XLSX, help="Output Excel file.")
parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_SEC, help="Solver time limit in seconds.")
parser.add_argument("--solver", default="gurobi", help="Pyomo solver name.")
parser.add_argument("--quiet", action="store_true", help="Suppress solver log output.")
args = parser.parse_args()

df = pd.read_excel(args.input)

for k, (idx, row) in enumerate(df.iterrows(), start=1):
    print(f"🔄 Solving instance {k}/{len(df)} (row {idx})...")
    try:
        I = int(row["I"])
        batch_list = [f"job{i+1}" for i in range(I)]

        # Parse input
        rho = np.array(ast.literal_eval(str(row["rho"]))).flatten()
        tau = np.array(ast.literal_eval(str(row["tau"]))).flatten()
        eps = np.array(ast.literal_eval(str(row["eps"]))).flatten()
        pred_rank = np.array(ast.literal_eval(str(row["rank_swap"]))).flatten()

        # Build input dicts
        param_data = {
            batch_list[i]: {"rho": float(rho[i]), "tau": float(tau[i]), "eps": float(eps[i])}
            for i in range(I)
        }
        predicted_ranks = {batch_list[i]: int(pred_rank[i]) for i in range(I)}

        # Build model (only need 'model' from the triple)
        model, _, _ = single_machine_ct_post_tight(param_data, predicted_ranks)

        # Solve with time limit
        solver = pyo.SolverFactory(args.solver)
        solver.options["TimeLimit"] = args.time_limit

        t0 = time.time()
        result = solver.solve(model, tee=(PRINT_SOLVER_LOG and not args.quiet), warmstart=True)
        elapsed_time = time.time() - t0
        solve_times.append(elapsed_time)

        # Termination condition
        try:
            term_cond = str(result.solver.termination_condition)
        except Exception:
            term_cond = "UNKNOWN"
        termination_conditions.append(term_cond)

        # Extract best incumbent (no Y/S; use X & Theta)
        period_of = {}
        for b in batch_list:
            best_t, best_val = None, -1.0
            for t in model.Periods:
                v = pyo.value(model.X[b, t])
                if v is None:
                    v = 0.0
                if v > best_val:
                    best_val, best_t = v, int(t)
            if best_t is None or best_val <= 1e-8:
                raise RuntimeError(f"No incumbent period chosen for {b} (all X[b,t] ~ 0?)")
            period_of[b] = best_t

        rank_vector = [period_of[b] for b in batch_list]
        start_times = [pyo.value(model.Theta[period_of[b]]) for b in batch_list]
        tardiness_value = pyo.value(model.obj)

        rank_vectors.append(rank_vector)
        start_time_vectors.append(start_times)
        tardiness_values.append(tardiness_value)

        print(f"Finished in {elapsed_time:.2f}s | {term_cond} | Tardiness = {tardiness_value:.4f}")

    except Exception as e:
        print(f"❌ Error at instance {k}: {e}")
        # Keep alignment so we can save a complete table
        rank_vectors.append(None)
        start_time_vectors.append(None)
        tardiness_values.append(None)
        termination_conditions.append(f"ERROR: {e}")
        solve_times.append(None)
        continue

# Save results for all rows
if SAVE_RESULTS:
    df_out = df.copy().reset_index(drop=True)
    df_out["rank_mip_post_tight_old"]  = [str(v) if v is not None else "" for v in rank_vectors]
    df_out["tardiness_mip_post_tight_old"]    = tardiness_values
    df_out.to_excel(args.output, index=False)
    print(f"Wrote results to: {args.output}")
