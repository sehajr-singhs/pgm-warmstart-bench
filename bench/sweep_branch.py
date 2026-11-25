"""Same sweep as sweep.py, but on the real PGM build of branch
pgm/experiment/custom-starting-nodal-states-for-power-flow, using its node input u_pu / u_angle as the start.

For every network and scenario set (same synthetic profiles and seeds as sweep.py):
  cold      : PGM's own start (linear; flat on case300, where linear diverges)
  previous  : node u_pu / u_angle = PGM's converged solution of the previous scenario
  noisy_1pct, noisy_5pct : converged solution with Gaussian noise on magnitude and angle
Iterations come from PGM's info logger (tag 3253) when available, and are cross-checked by probing max_iterations.
Every warm-started result is compared with the cold result (max |dU| in p.u.).
Wall time: median of repeated single calculations (model construction excluded), for the first N_TIME scenarios.
Usage: python sweep_branch.py <dir with network folders> <out_json> [<v2 results.json for comparison>]
"""
from __future__ import annotations

import json, os, re, sys, time
from pathlib import Path

import numpy as np
from power_grid_model import CalculationInitialization, CalculationMethod, PowerGridModel
from power_grid_model.utils import json_deserialize_from_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import NETWORKS, make_profiles, scaled_data, stats, TOL, MAX_ITER  # noqa: E402

try:
    from power_grid_model._core.logger import Logger, LoggerType
except Exception:  # noqa: BLE001
    Logger = None

N_TIME = int(os.environ.get("N_TIME", 40))
REPEATS = int(os.environ.get("REPEATS", 15))


def with_start(data, u):
    d = dict(data)
    node = data["node"].copy()
    if u is not None:
        node["u_pu"] = np.abs(u)
        node["u_angle"] = np.angle(u)
    d["node"] = node
    return d


def solve(data, init, max_iter=MAX_ITER):
    m = PowerGridModel(data)
    kw = dict(calculation_method=CalculationMethod.newton_raphson, error_tolerance=TOL, max_iterations=max_iter,
              calculation_initialization=CalculationInitialization[init])
    it = None
    try:
        if Logger is not None:
            with Logger(LoggerType.info) as log:
                out = m.calculate_power_flow(**kw)
            f = re.findall(r"Tag:3253: (\d+)", log.output)
            it = int(f[0]) if f else None
        else:
            out = m.calculate_power_flow(**kw)
    except Exception:  # noqa: BLE001
        return None, None, m, kw
    return out["node"]["u_pu"] * np.exp(1j * out["node"]["u_angle"]), it, m, kw


def probe_iters(data, init):
    """Smallest max_iterations that converges (independent of the logger)."""
    lo, hi = 1, MAX_ITER
    if solve(data, init, hi)[0] is None:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if solve(data, init, mid)[0] is not None:
            hi = mid
        else:
            lo = mid + 1
    return lo


def wall(m, kw):
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter(); m.calculate_power_flow(**kw); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def run(name, rel, scale_rng, solar, root, seed):
    data = json_deserialize_from_file(root / rel / "input.json")
    rng = np.random.default_rng(seed)
    ts_mult, ind_mult = make_profiles(data, scale_rng, solar, rng)
    noise = np.random.default_rng(seed + 999)
    cold_init = "flat" if name == "case300" else "linear"
    res = {"cold_init": cold_init}
    for setname, mult in (("timeseries", ts_mult), ("independent", ind_mult)):
        n_s = next(iter(mult.values())).shape[0]
        keys = ["cold", "previous", "noisy_1pct", "noisy_5pct"]
        it = {k: [] for k in keys}; probe = {k: [] for k in keys}; du = {k: [] for k in keys}; tm = {k: [] for k in keys}
        prev = None
        for k in range(n_s):
            dk = scaled_data(data, mult, k)
            u_true, i0, m0, kw0 = solve(with_start(dk, None), cold_init)
            it["cold"].append(i0)
            starts = {"previous": prev}
            if u_true is not None:
                for key, s in (("noisy_1pct", 0.01), ("noisy_5pct", 0.05)):
                    n = len(u_true)
                    starts[key] = np.abs(u_true) * (1 + noise.normal(0, s, n)) * np.exp(1j * (np.angle(u_true) + noise.normal(0, s, n)))
            for key in ["previous", "noisy_1pct", "noisy_5pct"]:
                u0 = starts.get(key)
                if u0 is None or u_true is None:
                    it[key].append(None); continue
                dd = with_start(dk, u0)
                u, i1, m1, kw1 = solve(dd, cold_init)
                it[key].append(i1)
                if u is not None:
                    du[key].append(float(np.max(np.abs(u - u_true))))
                if k < N_TIME and u is not None:
                    tm[key].append(wall(m1, kw1))
                if k < 25:
                    probe[key].append([i1, probe_iters(dd, cold_init)])
            if k < N_TIME and u_true is not None:
                tm["cold"].append(wall(m0, kw0))
            if k < 25:
                probe["cold"].append([i0, probe_iters(with_start(dk, None), cold_init)])
            if setname == "timeseries":
                prev = u_true
            else:  # independent set: the start is an unrelated earlier scenario's solution
                prev = u_true if u_true is not None else prev
        res[setname] = {k: {"iterations": stats(it[k]), "iterations_raw": it[k],
                            "max_abs_du_vs_cold_pu": max(du[k]) if du[k] else None,
                            "wall_s_per_calc": stats(tm[k]),
                            "logger_vs_probe_agree": f"{sum(a == b for a, b in probe[k])}/{len(probe[k])}"} for k in keys}
        print(name, setname, {k: (res[setname][k]["iterations"].get("median"), res[setname][k]["iterations"].get("n_fail")) for k in keys}, flush=True)
    return res


def main():
    root = Path(sys.argv[1]); out = Path(sys.argv[2])
    import importlib.metadata as md
    results = {"meta": {"pgm_version": md.version("power-grid-model"),
                        "branch": "pgm/experiment/custom-starting-nodal-states-for-power-flow",
                        "logger": Logger is not None, "tol": TOL, "max_iter": MAX_ITER,
                        "n_time": N_TIME, "repeats": REPEATS}, "networks": {}}
    for i, (name, (rel, scale_rng, solar)) in enumerate(NETWORKS.items()):
        results["networks"][name] = run(name, rel, scale_rng, solar, root, 20261028 + i)
    if len(sys.argv) > 3:  # agreement with the NumPy reimplementation (previous-step start) from the v2 run
        v2 = json.load(open(sys.argv[3]))
        agree = {}
        for name in results["networks"]:
            for s in ("timeseries", "independent"):
                a = results["networks"][name][s]["previous"]["iterations_raw"]
                b = v2["networks"][name][s]["external"]["previous_step"]["iterations_raw"]
                pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
                agree[f"{name}/{s}"] = dict(same=sum(x == y for x, y in pairs), compared=len(pairs),
                                            mean_branch=float(np.mean([x for x, _ in pairs])) if pairs else None,
                                            mean_reimpl=float(np.mean([y for _, y in pairs])) if pairs else None)
        results["agreement_with_reimplementation_previous_step"] = agree
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=lambda o: None))
    print("wrote", out)


if __name__ == "__main__":
    main()
