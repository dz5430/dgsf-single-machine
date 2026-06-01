# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import ast
import time
import argparse
from Single_machine_discrete_time import time_indexed_single_machine_model

parser = argparse.ArgumentParser(description="Solve reference schedules with the time-indexed MIP.")
parser.add_argument("--input", default="Updated_file.xlsx", help="Input Excel file.")
parser.add_argument("--output", default="Updated_file.xlsx", help="Output Excel file.")
parser.add_argument("--solver", default="gurobi", help="Pyomo solver name.")
args = parser.parse_args()

df = pd.read_excel(args.input)

rank_vectors = []
start_time_vectors = []
solve_times = []
tardiness_values = []

for idx, row in df.iterrows():
    print(f"Solving instance {idx+1}/{len(df)}...")
    try:
        I = int(row["I"])
        batch_list = [chr(65 + i) for i in range(I)]

        rho = np.array(ast.literal_eval(str(row["rho"]))).flatten()
        tau = np.array(ast.literal_eval(str(row["tau"]))).flatten()
        eps = np.array(ast.literal_eval(str(row["eps"]))).flatten()

        param_data = {
            batch_list[i]: {
                "rho": rho[i],
                "tau": tau[i],
                "eps": eps[i]
            }
            for i in range(I)
        }
        
        T = round(1.1 * sum(param_data[b]["tau"] for b in param_data))

        model = time_indexed_single_machine_model(param_data, T)
        solver = pyo.SolverFactory(args.solver)

        start_time = time.time()
        result = solver.solve(model, tee=True)
        elapsed_time = time.time() - start_time

        if (result.solver.status != pyo.SolverStatus.ok or
            result.solver.termination_condition != pyo.TerminationCondition.optimal):
            print(f"Instance {idx+1} not optimal.")
            rank_vectors.append(["infeasible"] * I)
            start_time_vectors.append(["infeasible"] * I)
            tardiness_values.append("infeasible")
            solve_times.append(elapsed_time)
            continue

        sequence = []
        # Extract start times from the model
        start_times = {b: pyo.value(model.S[b]) for b in batch_list}
        
        # Sort batches by start time to get execution order
        sequence = sorted(batch_list, key=lambda b: start_times[b])
        
        # Create rank vector (1-based ranks)
        batch_rank_dict = {batch: rank + 1 for rank, batch in enumerate(sequence)}
        rank_vector = [batch_rank_dict[b] for b in batch_list]
        tardiness_value = pyo.value(model.obj)

        # Record
        rank_vectors.append(rank_vector)
        start_time_vectors.append([start_times[b] for b in batch_list])
        tardiness_values.append(tardiness_value)
        solve_times.append(elapsed_time)
        #print(f"Solved in {elapsed_time:.4f} seconds.")

    except Exception as e:
        print(f"Error at instance {idx+1}: {e}")
        rank_vectors.append(["error"] * int(row["I"]))
        start_time_vectors.append(["error"] * int(row["I"]))
        solve_times.append(0)
        tardiness_values.append("error")

df["rank_vector_dtime"] = [str(vec) for vec in rank_vectors]
df["start_times_dtime"] = [str(vec) for vec in start_time_vectors]
df["solve_time_dtime"] = solve_times
df["tardiness_dtime"] = tardiness_values
df.to_excel(args.output, index=False)
