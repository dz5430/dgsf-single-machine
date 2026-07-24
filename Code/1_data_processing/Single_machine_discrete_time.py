# -*- coding: utf-8 -*-
import pyomo.environ as pyo

def time_indexed_single_machine_model(param_data, T, max_tard_mult=None):
    model = pyo.ConcreteModel()

    batch_list = list(param_data.keys())
    model.Batches = pyo.Set(initialize=batch_list)
    model.Time = pyo.Set(initialize=range(T))

    # Parameters
    model.rho = pyo.Param(model.Batches, initialize={b: param_data[b]["rho"] for b in batch_list})
    model.eps = pyo.Param(model.Batches, initialize={b: param_data[b]["eps"] for b in batch_list})
    model.tau = pyo.Param(model.Batches, initialize={b: param_data[b]["tau"] for b in batch_list})

    # Decision Variables
    model.X = pyo.Var(model.Batches, model.Time, within=pyo.Binary)  # X[i, t] = 1 if i starts at t
    model.S = pyo.Var(model.Batches, within=pyo.NonNegativeIntegers)  # Start time
    model.Tardiness = pyo.Var(model.Batches, within=pyo.NonNegativeIntegers)

    # Each batch assigned exactly once
    def assign_once(model, i):
        return sum(model.X[i, t] for t in model.Time) == 1
    model.assign_once = pyo.Constraint(model.Batches, rule=assign_once)

    # Respect release times
    def respect_release(model, i, t):
        if t < model.rho[i]:
            return model.X[i, t] == 0
        return pyo.Constraint.Skip
    model.release_constr = pyo.Constraint(model.Batches, model.Time, rule=respect_release)

    # No overlap on the single machine
    def no_overlap(model, t):
        return sum(
            model.X[i, tp]
            for i in model.Batches for tp in model.Time
            if tp <= t < tp + model.tau[i]
        ) <= 1
    model.no_overlap = pyo.Constraint(model.Time, rule=no_overlap)

    # Start time definition
    def start_time_def(model, i):
        return model.S[i] == sum(t * model.X[i, t] for t in model.Time)
    model.start_time_def = pyo.Constraint(model.Batches, rule=start_time_def)

    # Tardiness definition: Tardiness[i] >= Start + tau - due
    def tardiness_def(model, i):
        return model.Tardiness[i] >= model.S[i] + model.tau[i] - model.eps[i]
    model.tardiness_def = pyo.Constraint(model.Batches, rule=tardiness_def)

    # Optional diagnostic constraint used for the max-tardiness sensitivity
    # experiment.  Leaving ``max_tard_mult`` unset preserves the original
    # total-tardiness reference formulation.
    if max_tard_mult is not None:
        tau_avg = sum(float(param_data[b]["tau"]) for b in batch_list) / len(batch_list)

        def max_tardiness_cap(model, i):
            return model.Tardiness[i] <= max_tard_mult * tau_avg

        model.max_tardiness_cap = pyo.Constraint(model.Batches, rule=max_tardiness_cap)

    # Objective: Minimize total tardiness
    model.obj = pyo.Objective(expr=sum(model.Tardiness[i] for i in model.Batches), sense=pyo.minimize)

    return model
