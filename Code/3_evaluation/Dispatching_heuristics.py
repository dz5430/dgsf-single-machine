# -*- coding: utf-8 -*-
"""
Dispatching-rule baselines for single-machine total tardiness with release dates.

The PRTT rule follows Chu and Portmann's dynamic priority index for
1|r_i|sum T_i:

    PRTT(i, t) = max(t, r_i) + max(d_i, max(t, r_i) + p_i).

At each decision point, the job with the smallest PRTT value is selected.
Unlike availability-only dispatching rules, this definition can intentionally
idle until a not-yet-released job becomes available.
"""

import argparse
import ast
from typing import Callable

import numpy as np
import pandas as pd


DEFAULT_RULES = ("edd", "spt", "lsf", "atc", "prtt")


def parse_1d(cell) -> np.ndarray:
    return np.array(ast.literal_eval(str(cell)), dtype=float).flatten()


def schedule_from_rule(
    rho: np.ndarray,
    tau: np.ndarray,
    eps: np.ndarray,
    rule: str,
    atc_k: float = 2.0,
) -> tuple[list[int], list[float], float]:
    n_jobs = len(rho)
    unscheduled = set(range(n_jobs))
    sequence = []
    start_times = []
    t = 0.0
    tau_bar = float(np.mean(tau)) if n_jobs else 0.0

    while unscheduled:
        if rule == "prtt":
            candidates = list(unscheduled)
        else:
            available = [j for j in unscheduled if rho[j] <= t]
            if not available:
                t = min(rho[j] for j in unscheduled)
                available = [j for j in unscheduled if rho[j] <= t]
            candidates = available

        priority = _priority_function(rule, rho, tau, eps, t, tau_bar, atc_k)
        if rule == "atc":
            chosen = max(candidates, key=lambda j: (priority(j), -eps[j], -rho[j], -j))
        else:
            chosen = min(candidates, key=lambda j: (priority(j), rho[j], eps[j], tau[j], j))

        start = max(t, float(rho[chosen]))
        sequence.append(chosen)
        start_times.append(start)
        t = start + float(tau[chosen])
        unscheduled.remove(chosen)

    completion_times = np.array(start_times, dtype=float) + tau[sequence]
    tardiness = float(np.maximum(completion_times - eps[sequence], 0.0).sum())
    rank_vector = [0] * n_jobs
    for rank, job_idx in enumerate(sequence, start=1):
        rank_vector[job_idx] = rank
    return rank_vector, start_times, tardiness


def _priority_function(
    rule: str,
    rho: np.ndarray,
    tau: np.ndarray,
    eps: np.ndarray,
    t: float,
    tau_bar: float,
    atc_k: float,
) -> Callable[[int], float]:
    if rule == "edd":
        return lambda j: float(eps[j])
    if rule == "spt":
        return lambda j: float(tau[j])
    if rule == "lsf":
        return lambda j: float(eps[j] - t - tau[j])
    if rule == "atc":
        denom = max(atc_k * tau_bar, 1e-8)
        return lambda j: float(
            (1.0 / max(tau[j], 1e-8))
            * np.exp(-max(eps[j] - tau[j] - t, 0.0) / denom)
        )
    if rule == "prtt":
        return lambda j: float(max(t, rho[j]) + max(eps[j], max(t, rho[j]) + tau[j]))
    raise ValueError(f"Unknown dispatching rule: {rule}")


def run_dispatching_rules(
    input_xlsx: str,
    output_xlsx: str,
    rules: tuple[str, ...],
    atc_k: float,
) -> pd.DataFrame:
    df = pd.read_excel(input_xlsx)

    for rule in rules:
        rank_values = []
        start_values = []
        tardiness_values = []

        for idx, row in df.iterrows():
            try:
                rho = parse_1d(row["rho"])
                tau = parse_1d(row["tau"])
                eps = parse_1d(row["eps"])
                rank_vector, start_times, tardiness = schedule_from_rule(
                    rho, tau, eps, rule, atc_k=atc_k
                )
                rank_values.append(str(rank_vector))
                start_values.append(str(start_times))
                tardiness_values.append(tardiness)
            except Exception as exc:
                print(f"Error applying {rule.upper()} to row {idx}: {exc}")
                rank_values.append("")
                start_values.append("")
                tardiness_values.append(np.nan)

        df[f"rank_{rule}"] = rank_values
        df[f"start_times_{rule}"] = start_values
        df[f"tardiness_{rule}"] = tardiness_values

    df.to_excel(output_xlsx, index=False)
    print(f"Wrote dispatching-rule results to: {output_xlsx}")
    return df


def parse_rules(raw_rules: str) -> tuple[str, ...]:
    rules = tuple(rule.strip().lower() for rule in raw_rules.split(",") if rule.strip())
    unknown = sorted(set(rules) - set(DEFAULT_RULES))
    if unknown:
        raise ValueError(f"Unknown rules: {', '.join(unknown)}")
    return rules


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dispatching-rule baselines.")
    parser.add_argument("--input", default="Updated_file.xlsx", help="Input Excel file.")
    parser.add_argument("--output", default="Dispatching_heuristics.xlsx", help="Output Excel file.")
    parser.add_argument(
        "--rules",
        default=",".join(DEFAULT_RULES),
        help="Comma-separated rules from: edd,spt,lsf,atc,prtt.",
    )
    parser.add_argument("--atc-k", type=float, default=2.0, help="ATC lookahead parameter.")
    args = parser.parse_args()

    run_dispatching_rules(args.input, args.output, parse_rules(args.rules), args.atc_k)
