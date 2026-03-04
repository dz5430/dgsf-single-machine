# -*- coding: utf-8 -*-
import pyomo.environ as pyo

def single_machine_ct_post_tight(
    param_data, predicted_ranks, *,
    delta_slots=4,          # window half-width: Wi = [r_i-Δ, r_i+Δ] ∩ {1..P}
    bigM=None,              # global M
    inject_mipstart=True,   # seed X/Theta/T from ML order
    tie_break_key=None      # optional: function i -> key for breaking rank ties
):
    """
    Event-based continuous-time SMTTP (no Y_ij) with rank-window tightening.

    - Periods t=1..P with time points Theta[t] (start) and Theta[P+1] (end).
    - X[i,t] = 1 iff job i is assigned to period t (permutation).
    - No-overlap via period length: Theta[t+1] - Theta[t] >= tau[i] * X[i,t].
    - Release and tardiness enforced via big-M on allowed (i,t) pairs only.
    - Tightening: constraints are built only for (i,t) where t ∈ Wi; outside Wi, X[i,t] is fixed to 0.
    - Warm start: ML order forward pass, writing both Theta[t] and Theta[t+1] each period.

    Returns: (model, rank_idx, order, Wi)
    """

    # ---------- Sets & data ----------
    model = pyo.ConcreteModel()
    jobs = list(param_data.keys())
    P = len(jobs)

    model.Jobs    = pyo.Set(initialize=jobs, ordered=True)
    model.Periods = pyo.RangeSet(1, P)
    model.Points  = pyo.RangeSet(1, P+1)

    rho = {i: float(param_data[i]["rho"]) for i in jobs}
    tau = {i: float(param_data[i]["tau"]) for i in jobs}
    eps = {i: float(param_data[i]["eps"]) for i in jobs}

    model.rho = pyo.Param(model.Jobs, initialize=rho)
    model.tau = pyo.Param(model.Jobs, initialize=tau)
    model.eps = pyo.Param(model.Jobs, initialize=eps)

    # ---------- Rank order & windows ----------
    if tie_break_key is None:
        order = sorted(jobs, key=lambda i: (predicted_ranks[i], i))
    else:
        order = sorted(jobs, key=lambda i: (predicted_ranks[i], tie_break_key(i)))
    rank_idx = {i: k for k, i in enumerate(order, start=1)}

    # Wi per job; ensure coverage of each period
    Wi = {
        i: list(range(max(1, rank_idx[i] - delta_slots),
                      min(P, rank_idx[i] + delta_slots) + 1))
        for i in jobs
    }
    # Coverage: each period must have at least one candidate job
    allowed_jobs_by_t = {t: [i for i in jobs if t in Wi[i]] for t in range(1, P+1)}
    for t, cand in allowed_jobs_by_t.items():
        if not cand:
            raise ValueError(f"No candidate jobs for period t={t}. Increase delta_slots.")

    # Allowed set for constraints
    Allowed = [(i, t) for i in jobs for t in Wi[i]]
    model.Allowed = pyo.Set(dimen=2, initialize=Allowed)

    # ---------- Variables ----------
    # Keep X on the full rectangle for compatibility with existing driver code
    model.X     = pyo.Var(model.Jobs, model.Periods, within=pyo.Binary)
    model.Theta = pyo.Var(model.Points, within=pyo.NonNegativeReals)
    model.T     = pyo.Var(model.Jobs, within=pyo.NonNegativeReals)

    # Fix X outside Wi to 0 (tightening by elimination)
    for i in jobs:
        for t in range(1, P+1):
            if t not in Wi[i]:
                model.X[i, t].fix(0)

    # ---------- Time grid monotonicity ----------
    def chain_rule(m, t):
        if t == P+1:
            return pyo.Constraint.Skip
        return m.Theta[t+1] >= m.Theta[t]
    model.time_chain = pyo.Constraint(model.Points, rule=chain_rule)

    # ---------- Permutation constraints ----------
    # 1) Each job once, over its window
    def one_period_per_job_rule(m, i):
        return sum(m.X[i, t] for t in Wi[i]) == 1
    model.one_period_per_job = pyo.Constraint(model.Jobs, rule=one_period_per_job_rule)

    # 2) Each period once, over jobs that include that period in their window
    def one_job_per_period_rule(m, t):
        return sum(m.X[i, t] for i in allowed_jobs_by_t[t]) == 1
    model.one_job_per_period = pyo.Constraint(model.Periods, rule=one_job_per_period_rule)

    # ---------- Big-M default ----------
    if bigM is None:
        bigM = (max(rho.values()) if rho else 0.0) + sum(tau.values()) + 1.0

    # ---------- Constraints only on Allowed pairs ----------
    def period_length_rule(m, i, t):
        return m.Theta[t+1] - m.Theta[t] >= m.tau[i] * m.X[i, t]
    model.period_length = pyo.Constraint(model.Allowed, rule=period_length_rule)

    def release_rule(m, i, t):
        return m.Theta[t] + bigM * (1 - m.X[i, t]) >= m.rho[i]
    model.release = pyo.Constraint(model.Allowed, rule=release_rule)

    def tardiness_rule(m, i, t):
        return m.T[i] >= (m.Theta[t] + m.tau[i] - m.eps[i]) - bigM * (1 - m.X[i, t])
    model.tardiness = pyo.Constraint(model.Allowed, rule=tardiness_rule)

    # ---------- Handy expressions ----------
    def S_expr(m, i):
        return sum(m.Theta[t] * m.X[i, t] for t in Wi[i])
    model.S = pyo.Expression(model.Jobs, rule=S_expr)

    def C_expr(m, i):
        return model.S[i] + m.tau[i]
    model.C = pyo.Expression(model.Jobs, rule=C_expr)

    # ---------- Objective ----------
    model.obj = pyo.Objective(expr=sum(model.T[i] for i in model.Jobs), sense=pyo.minimize)

    # ---------- Warm start ----------
    if inject_mipstart:
        # 1) X: put each job in its ML slot (must be inside Wi)
        for i in jobs:
            r = rank_idx[i]
            if r not in Wi[i]:
                raise ValueError(f"Warm-start infeasible for job {i}: r={r} not in Wi; increase delta_slots.")
            for t in Wi[i]:
                model.X[i, t].set_value(1 if t == r else 0)
            # (X outside Wi is fixed(0) above)

        # 2) Theta using forward pass in ML order: write both ends every period
        theta_vals = {}
        prev_completion = 0.0
        for t in range(1, P+1):
            i_t = order[t-1]
            s_t = max(prev_completion, rho[i_t])
            c_t = s_t + tau[i_t]
            theta_vals[t]   = s_t
            theta_vals[t+1] = c_t
            prev_completion = c_t

        for t, v in theta_vals.items():
            model.Theta[t].set_value(v)

        # 3) T seeds
        for i in jobs:
            r = rank_idx[i]
            s_i = theta_vals[r]
            c_i = s_i + tau[i]
            model.T[i].set_value(max(0.0, c_i - eps[i]))

    return model, rank_idx, order