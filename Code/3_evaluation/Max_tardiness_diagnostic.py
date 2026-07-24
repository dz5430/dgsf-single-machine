"""Reproduce the capped MIP and DGSF post-processing schedules in Table 6.

The input workbook must contain one or more rows with ``I``, ``rho``, ``tau``,
``eps``, and ``new_rank_swap`` columns.  The latter is the DGSF sequence after
the ordinary local-swap step.  The script writes the capped time-indexed MIP
reference solution and the capped rank-window post-processing solution for
each requested threshold.  The paper uses delta=6 and a 120-second time limit.
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR.parents[1] / "Code").is_dir():
    # Repository layout: Code/3_evaluation/this_file.py
    ROOT = SCRIPT_DIR.parents[1]
    DATA_PROCESSING_DIR = ROOT / "Code" / "1_data_processing"
    DGSF_DIR = ROOT / "Code" / "2_dgsf_main"
elif (SCRIPT_DIR.parent / "04_scripts").is_dir():
    # Google Drive archive layout: 04_scripts/this_file.py
    ROOT = SCRIPT_DIR.parent
    DATA_PROCESSING_DIR = SCRIPT_DIR
    DGSF_DIR = SCRIPT_DIR
else:
    raise RuntimeError("Could not identify the repository or Google Drive archive layout.")

sys.path.insert(0, str(DATA_PROCESSING_DIR))
sys.path.insert(0, str(DGSF_DIR))

from Single_machine_discrete_time import time_indexed_single_machine_model
from MIP_ContTime_Post_Tight import single_machine_ct_post_tight


def parse_vector(value: object) -> np.ndarray:
    return np.asarray(ast.literal_eval(str(value))).reshape(-1)


def build_instance(row: pd.Series):
    n_jobs = int(row["I"])
    jobs = [f"job{i + 1}" for i in range(n_jobs)]
    rho = parse_vector(row["rho"]).astype(float)
    tau = parse_vector(row["tau"]).astype(float)
    eps = parse_vector(row["eps"]).astype(float)
    data = {
        jobs[i]: {"rho": rho[i], "tau": tau[i], "eps": eps[i]}
        for i in range(n_jobs)
    }
    return jobs, rho, tau, eps, data


def solve_reference(data, jobs, tau, threshold, solver_name, quiet):
    horizon = round(1.1 * sum(tau))
    model = time_indexed_single_machine_model(data, horizon, max_tard_mult=threshold)
    solver = pyo.SolverFactory(solver_name)
    started = time.time()
    result = solver.solve(model, tee=not quiet)
    elapsed = time.time() - started
    start_by_job = {job: float(pyo.value(model.S[job])) for job in jobs}
    order = sorted(jobs, key=lambda job: start_by_job[job])
    rank = {job: position + 1 for position, job in enumerate(order)}
    return {
        "rank": [rank[job] for job in jobs],
        "tardiness": float(pyo.value(model.obj)),
        "max_tardiness": max(float(pyo.value(model.Tardiness[job])) for job in jobs),
        "solve_time": elapsed,
        "termination": str(result.solver.termination_condition),
    }


def solve_dgsf_post(data, jobs, tau, rank_column, row, threshold, delta, time_limit, solver_name, quiet):
    seed = parse_vector(row[rank_column]).astype(int)
    predicted_ranks = {job: int(seed[i]) for i, job in enumerate(jobs)}
    model, _, _ = single_machine_ct_post_tight(data, predicted_ranks, delta_slots=delta)
    tau_avg = float(np.mean(tau))
    model.max_tardiness_cap = pyo.Constraint(
        model.Jobs, rule=lambda m, job: m.T[job] <= threshold * tau_avg
    )
    solver = pyo.SolverFactory(solver_name)
    solver.options["TimeLimit"] = time_limit
    started = time.time()
    result = solver.solve(model, tee=not quiet, warmstart=True)
    elapsed = time.time() - started
    period_by_job = {}
    for job in jobs:
        period_by_job[job] = max(
            model.Periods,
            key=lambda period: float(pyo.value(model.X[job, period]) or 0.0),
        )
    return {
        "rank": [int(period_by_job[job]) for job in jobs],
        "tardiness": float(pyo.value(model.obj)),
        "max_tardiness": max(float(pyo.value(model.T[job])) for job in jobs),
        "solve_time": elapsed,
        "termination": str(result.solver.termination_condition),
    }


def save_result(df, row_index, prefix, threshold, result):
    suffix = f"{prefix}_d6_t120_k{threshold}"
    df.at[row_index, f"rank_{suffix}"] = str(result["rank"])
    for key in ("tardiness", "max_tardiness", "solve_time", "termination"):
        df.at[row_index, f"{key}_{suffix}"] = result[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Workbook containing the diagnostic instance.")
    parser.add_argument("--output", required=True, help="Workbook to receive recreated results.")
    parser.add_argument("--rank-column", default="new_rank_swap")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[10, 12, 15])
    parser.add_argument("--delta", type=int, default=6)
    parser.add_argument("--time-limit", type=float, default=120)
    parser.add_argument("--solver", default="gurobi")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.delta != 6 or args.time_limit != 120:
        raise ValueError("Table 6 uses delta=6 and a 120-second post-processing limit.")

    df = pd.read_excel(args.input, sheet_name="Schedules")
    for row_index, row in df.iterrows():
        jobs, _, tau, _, data = build_instance(row)
        for threshold in args.thresholds:
            reference = solve_reference(data, jobs, tau, threshold, args.solver, args.quiet)
            post = solve_dgsf_post(
                data, jobs, tau, args.rank_column, row, threshold,
                args.delta, args.time_limit, args.solver, args.quiet,
            )
            save_result(df, row_index, "mip", threshold, reference)
            save_result(df, row_index, "dgsf", threshold, post)

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Schedules", index=False)
    print(f"Wrote recreated diagnostic results to {args.output}")


if __name__ == "__main__":
    main()
