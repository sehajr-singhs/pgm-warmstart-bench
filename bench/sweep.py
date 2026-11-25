"""Benchmark sweep: what would a user-supplied Newton-Raphson start save in PGM?

For each PGM validation network:
  1. Build synthetic time series (smooth daily profile + per-appliance AR(1) noise) and an
     independent random batch on top of the PGM input.
  2. PGM itself: per-scenario NR iteration counts (from PGM's own info logger, tag 3253) for
     linear / flat / average_source, per-iteration cost from the same logger, and batch wall time.
  3. External NR (pgm_nr.py, validated against PGM voltages and iteration counts) for the same
     scenarios: PGM's starts plus the starts PGM cannot take today (previous step, perturbed truth).
Writes one JSON with every number.

Usage: python sweep.py <pgm_repo_root> <out_json>
"""
from __future__ import annotations

import copy
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import power_grid_model as pgm_pkg
from power_grid_model import CalculationInitialization, CalculationMethod, PowerGridModel, initialize_array
from power_grid_model.utils import json_deserialize_from_file
from power_grid_model._core.logger import Logger, LoggerType

try:
    from pgm_nr import Network  # local layout
except ImportError:  # bundled kernel: Network is defined above this file
    pass

TOL = 1e-8
MAX_ITER = 30
N_DAYS = int(os.environ.get('SWEEP_DAYS', 5))
T_DAY = 96
N_INDEP = int(os.environ.get('SWEEP_INDEP', 200))
SIGMAS = [0.001, 0.005, 0.01, 0.02, 0.05]
INITS = ["linear", "flat", "average_source"]

NETWORKS = {
    # name: (relative dir, load scale range of daily profile, has rooftop-style generation)
    "ieee14": ("ieee14", (0.80, 1.00), False),
    "case300": ("matpower-case300/flat-start", (0.80, 1.00), False),
    "vision47": ("vision-validation-network", (0.40, 1.00), True),
    "ieee118": ("ieee118", (0.80, 1.00), False),
}


def daily_shape(t):
    h = (t % T_DAY) / T_DAY * 24
    morning = np.exp(-0.5 * ((h - 8) / 2.0) ** 2)
    evening = np.exp(-0.5 * ((h - 19) / 2.5) ** 2)
    s = 0.15 + 0.55 * evening + 0.35 * morning
    return s / 0.75  # roughly 0.2..1


def solar_shape(t):
    h = (t % T_DAY) / T_DAY * 24
    return np.clip(np.sin(np.pi * (h - 6) / 12), 0, None)


def make_profiles(data, scale_rng, solar, rng):
    """Returns dict comp -> (T, n) multipliers for time series and for an independent batch."""
    T = N_DAYS * T_DAY
    t = np.arange(T)
    lo, hi = scale_rng
    base = lo + (hi - lo) * np.clip(daily_shape(t), 0, 1)
    ts, ind = {}, {}
    for comp in ("sym_load", "sym_gen"):
        if comp not in data:
            continue
        n = len(data[comp])
        # AR(1) noise per appliance, 3% stationary std, correlation 0.9 per 15 min
        e = np.zeros((T, n))
        phi, sd = 0.9, 0.03
        e[0] = rng.normal(0, sd, n)
        for k in range(1, T):
            e[k] = phi * e[k - 1] + rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
        if comp == "sym_gen" and solar:
            ts[comp] = np.clip(solar_shape(t)[:, None] * (1 + e), 0, None)
            ind[comp] = rng.uniform(0, 1, (N_INDEP, n))
        else:
            ts[comp] = base[:, None] * (1 + e)
            ind[comp] = rng.uniform(lo, hi, (N_INDEP, n))
    return ts, ind


def scaled_data(data, mult, k):
    d = {key: arr for key, arr in data.items()}
    for comp, m in mult.items():
        a = data[comp].copy()
        a["p_specified"] = a["p_specified"] * m[k]
        a["q_specified"] = a["q_specified"] * m[k]
        d[comp] = a
    return d


def update_batch(data, mult):
    upd = {}
    for comp, m in mult.items():
        n_s, n = m.shape
        u = initialize_array("update", comp, (n_s, n))
        u["id"] = data[comp]["id"][None, :]
        u["p_specified"] = data[comp]["p_specified"][None, :] * m
        u["q_specified"] = data[comp]["q_specified"][None, :] * m
        upd[comp] = u
    return upd


def pgm_per_scenario(data, mult, init):
    """PGM iteration count and logged phase timings per scenario (single calculations)."""
    model = PowerGridModel(data)
    n_s = next(iter(mult.values())).shape[0]
    upd = update_batch(data, mult)
    iters, solver_t, iter_t, init_t, u_all = [], [], [], [], []
    for k in range(n_s):
        m = model.copy()
        m.update(update_data={c: v[k] for c, v in upd.items()})
        try:
            with Logger(LoggerType.info) as log:
                out = m.calculate_power_flow(calculation_method=CalculationMethod.newton_raphson,
                                             error_tolerance=TOL, max_iterations=MAX_ITER,
                                             calculation_initialization=CalculationInitialization[init])
            txt = log.output
            iters.append(int(re.findall(r"Tag:3253: (\d+)", txt)[0]))
            tag = lambda c: sum(float(x) for x in re.findall(rf"Tag:{c}: ([0-9.eE+-]+)", txt))
            solver_t.append(tag(3200))
            iter_t.append(tag(3233) + tag(3241) + tag(3251))
            init_t.append(tag(3211))
            u_all.append(out["node"]["u_pu"] * np.exp(1j * out["node"]["u_angle"]))
        except Exception as e:  # noqa: BLE001
            iters.append(None); solver_t.append(None); iter_t.append(None); init_t.append(None)
            u_all.append(None)
    return iters, solver_t, iter_t, init_t, u_all


def pgm_batch_time(data, mult, init, repeats=7):
    model = PowerGridModel(data)
    upd = update_batch(data, mult)
    n_s = next(iter(mult.values())).shape[0]
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.calculate_power_flow(update_data=upd, calculation_method=CalculationMethod.newton_raphson,
                                   error_tolerance=TOL, max_iterations=MAX_ITER, threading=-1,
                                   calculation_initialization=CalculationInitialization[init],
                                   continue_on_batch_error=True)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / n_s


def stats(xs):
    v = [x for x in xs if x is not None]
    if not v:
        return {"n": 0, "n_fail": len(xs)}
    a = np.array(v, dtype=float)
    return {"n": len(v), "n_fail": len(xs) - len(v), "median": float(np.median(a)), "mean": float(a.mean()),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max()),
            "hist": {str(int(k)): int(c) for k, c in zip(*np.unique(a, return_counts=True))} if a.max() < 100 else None}


def ext_run(net, sc, u0):
    t0 = time.perf_counter()
    u, it, ok = net.solve(sc, u0, tol=TOL, max_iter=MAX_ITER)
    return u, (it if ok else None), time.perf_counter() - t0


def run_network(name, rel, scale_rng, solar, root, rng):
    data = json_deserialize_from_file(root / rel / "input.json")
    net = Network(data)
    ts_mult, ind_mult = make_profiles(data, scale_rng, solar, rng)
    res = {"dir": f"tests/data/power_flow/{rel}", "n_bus_after_link_merge": net.n,
           "n_node": int(len(data["node"])), "n_pv": int(len(net.pv_bus)),
           "components": {str(k): int(len(v)) for k, v in data.items()}}
    for setname, mult in (("timeseries", ts_mult), ("independent", ind_mult)):
        n_s = next(iter(mult.values())).shape[0]
        out = {"n_scenarios": n_s}
        # PGM
        pgm = {}
        pgm_u = None
        for init in INITS:
            iters, solver_t, iter_t, init_t, u_all = pgm_per_scenario(data, mult, init)
            per_iter = [a / b for a, b in zip(iter_t, iters) if a is not None]
            pgm[init] = {"iterations": stats(iters), "iterations_raw": iters,
                         "logged_solver_time_s": stats(solver_t),
                         "logged_time_per_iteration_s": stats(per_iter),
                         "logged_init_time_s": stats(init_t),
                         "batch_wall_time_per_scenario_s": pgm_batch_time(data, mult, init)}
            if pgm_u is None and all(u is not None for u in u_all):
                pgm_u = u_all
        out["pgm"] = pgm
        # external NR
        ext = {k: {"iters": [], "time": []} for k in INITS + ["previous_step", "previous_step_lag4"] + [f"perturbed_{s}" for s in SIGMAS]}
        max_du_vs_pgm = 0.0
        agree = {init: 0 for init in INITS}
        compared = {init: 0 for init in INITS}
        prev_u = None
        sols = []
        step_change = []
        prev_s = None
        for k in range(n_s):
            sc = net.scenario(scaled_data(data, mult, k))
            s_tot = sc["s0"] + sc["s1"] + sc["s2"]
            step_change.append(None if prev_s is None else
                               float(np.max(np.abs(s_tot - prev_s)) / max(np.max(np.abs(prev_s)), 1e-12)))
            prev_s = s_tot
            u_true = None
            for init in INITS:
                u, it, dt = ext_run(net, sc, getattr(net, f"start_{init}")(sc))
                ext[init]["iters"].append(it); ext[init]["time"].append(dt)
                if it is not None and u_true is None:
                    u_true = u
                pi = pgm[init]["iterations_raw"][k]
                if pi is not None or it is not None:
                    compared[init] += 1
                    agree[init] += int(pi == it)
            if pgm_u is not None and u_true is not None:
                u_p = np.zeros(net.n, complex)
                for nid, val in zip(data["node"]["id"], pgm_u[k]):
                    u_p[net.id2bus[int(nid)]] = val
                max_du_vs_pgm = max(max_du_vs_pgm, float(np.max(np.abs(u_true - u_p))))
            # previous step warm start (first scenario: PGM default linear, or flat where linear fails)
            u0 = prev_u if prev_u is not None else (net.start_flat(sc) if name == "case300" else net.start_linear(sc))
            u, it, dt = ext_run(net, sc, u0)
            ext["previous_step"]["iters"].append(it); ext["previous_step"]["time"].append(dt)
            prev_u = u if it is not None else None
            sols.append(u_true)
            # warm start from the solution four steps back (one hour at 15 min resolution)
            if k >= 4 and sols[k - 4] is not None:
                u, it, dt = ext_run(net, sc, sols[k - 4])
            else:
                it, dt = None, None
            ext["previous_step_lag4"]["iters"].append(it); ext["previous_step_lag4"]["time"].append(dt)
            for s in SIGMAS:
                if u_true is None:
                    ext[f"perturbed_{s}"]["iters"].append(None); ext[f"perturbed_{s}"]["time"].append(None)
                    continue
                n1, n2 = rng.normal(0, s, net.n), rng.normal(0, s, net.n)
                u0 = np.abs(u_true) * (1 + n1) * np.exp(1j * (np.angle(u_true) + n2))
                u, it, dt = ext_run(net, sc, u0)
                ext[f"perturbed_{s}"]["iters"].append(it); ext[f"perturbed_{s}"]["time"].append(dt)
        out["external"] = {k: {"iterations": stats(v["iters"]), "python_wall_time_s": stats(v["time"]),
                               "iterations_raw": v["iters"]} for k, v in ext.items()}
        out["external_validation"] = {
            "max_abs_u_diff_vs_pgm_pu": max_du_vs_pgm,
            "iteration_count_agreement": {i: {"agree": agree[i], "compared": compared[i]} for i in INITS}}
        out["relative_step_change"] = stats(step_change[1:])
        out["relative_step_change_raw"] = step_change
        res[setname] = out
        print(name, setname, "done", flush=True)
    return res


def main():
    root = Path(sys.argv[1]) / "tests/data/power_flow"
    out_path = Path(sys.argv[2])
    results = {"meta": {"power_grid_model_version": getattr(pgm_pkg, "__version__", None),
                        "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                        "tol": TOL, "max_iter": MAX_ITER, "n_days": N_DAYS, "steps_per_day": T_DAY,
                        "n_independent": N_INDEP, "sigmas": SIGMAS, "seed": 20261028,
                        "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
               "networks": {}}
    try:
        import importlib.metadata as md
        results["meta"]["power_grid_model_version"] = md.version("power-grid-model")
    except Exception:  # noqa: BLE001
        pass
    only = os.environ.get("SWEEP_ONLY")
    for i, (name, (rel, scale_rng, solar)) in enumerate(NETWORKS.items()):
        if only and name not in only.split(","):
            continue
        rng = np.random.default_rng(20261028 + i)
        results["networks"][name] = run_network(name, rel, scale_rng, solar, root, rng)
    results["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1, default=lambda o: None))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
