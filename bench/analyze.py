"""Summarize results.json from sweep.py into tables (summary.json + summary.md). Every number comes from results.json."""
import json
import sys
from pathlib import Path

import numpy as np

res = json.loads(Path(sys.argv[1]).read_text())
out_dir = Path(sys.argv[2])
SIG = res["meta"]["sigmas"]
summary = {"meta": res["meta"], "networks": {}}
md = []


def arr(xs, start=0):
    v = xs[start:]
    ok = np.array([x for x in v if x is not None], dtype=float)
    return ok, len(v) - len(ok), len(v)


def med(xs, start=0):
    ok, nf, n = arr(xs, start)
    return (float(np.median(ok)) if len(ok) else None, float(ok.mean()) if len(ok) else None, nf, n)


for name, net in res["networks"].items():
    s = {"n_bus": net["n_bus_after_link_merge"], "n_pv": net["n_pv"]}
    for setname in ("timeseries", "independent"):
        o = net[setname]
        pgm, ext = o["pgm"], o["external"]
        row = {"validation": o["external_validation"]}
        for init in ("linear", "flat", "average_source"):
            r = pgm[init]
            row[f"pgm_{init}"] = {"median_iter": r["iterations"].get("median"), "mean_iter": r["iterations"].get("mean"),
                                  "n_fail": r["iterations"]["n_fail"],
                                  "median_time_per_iter_us": None if not r["logged_time_per_iteration_s"].get("median") else r["logged_time_per_iteration_s"]["median"] * 1e6,
                                  "median_solver_time_us": None if not r["logged_solver_time_s"].get("median") else r["logged_solver_time_s"]["median"] * 1e6,
                                  "batch_wall_us_per_scenario": r["batch_wall_time_per_scenario_s"] * 1e6}
        # best existing PGM option = lowest mean iterations among options with no failures
        ok_inits = [i for i in ("linear", "flat", "average_source") if row[f"pgm_{i}"]["n_fail"] == 0]
        best = min(ok_inits, key=lambda i: row[f"pgm_{i}"]["mean_iter"])
        row["pgm_default"] = "linear"
        row["pgm_best_existing"] = best
        warm_start = 1  # first scenario has no previous step
        for key, start in (("previous_step", warm_start), ("previous_step_lag4", 4)):
            m, mean, nf, n = med(ext[key]["iterations_raw"], start)
            row[key] = {"median_iter": m, "mean_iter": mean, "n_fail": nf, "n": n}
        for sg in SIG:
            m, mean, nf, n = med(ext[f"perturbed_{sg}"]["iterations_raw"])
            row[f"perturbed_{sg}"] = {"median_iter": m, "mean_iter": mean, "n_fail": nf, "n": n}
        # projected saving inside PGM: (best existing mean iters - warm mean iters) x PGM time per iteration
        t_it = row[f"pgm_{best}"]["median_time_per_iter_us"]
        base_mean = row[f"pgm_{best}"]["mean_iter"]
        for key in ["previous_step"] + [f"perturbed_{sg}" for sg in SIG]:
            w = row[key]["mean_iter"]
            row[key]["mean_iter_saved_vs_best_existing"] = None if w is None else base_mean - w
            row[key]["projected_pgm_us_saved_per_scenario"] = None if w is None else (base_mean - w) * t_it
            row[key]["projected_fraction_of_pgm_batch_time"] = (None if w is None else
                                                                (base_mean - w) * t_it / row[f"pgm_{best}"]["batch_wall_us_per_scenario"])
        # where warm start does not help: per-step comparison with the best existing PGM option
        warm = ext["previous_step"]["iterations_raw"]
        base = pgm[best]["iterations_raw"]
        pairs = [(w, b) for w, b in zip(warm[1:], base[1:]) if w is not None and b is not None]
        row["previous_step_vs_best_existing_per_step"] = {
            "fewer": int(sum(w < b for w, b in pairs)), "equal": int(sum(w == b for w, b in pairs)),
            "more": int(sum(w > b for w, b in pairs)), "n": len(pairs)}
        # step size dependence (time series only)
        if setname == "timeseries":
            sc = np.array([x if x is not None else np.nan for x in o["relative_step_change_raw"]], dtype=float)
            wv = np.array([x if x is not None else np.nan for x in warm], dtype=float)
            idx = np.arange(1, len(sc))
            qs = np.nanpercentile(sc[idx], [0, 50, 90, 100])
            bins = []
            for lo, hi in zip(qs[:-1], qs[1:]):
                m = (sc[idx] >= lo) & (sc[idx] <= hi)
                bins.append({"rel_step_change_range": [float(lo), float(hi)], "n": int(m.sum()),
                             "median_warm_iter": float(np.nanmedian(wv[idx][m]))})
            row["previous_step_by_step_change"] = bins
        s[setname] = row
    summary["networks"][name] = s

lines = ["| network | buses (PV) | set | PGM default (linear) | best existing option | previous step | 1 h old solution | "
         + " | ".join(f"truth + {sg*100:g}% noise" for sg in SIG) + " | PGM time per NR iteration | projected PGM saving per step (previous step) |",
         "|" + "---|" * (8 + len(SIG) + 1)]


def cell(r):
    if r is None or r["median_iter"] is None:
        return "diverged" if r else "n/a"
    f = f" ({r['n_fail']} fail)" if r.get("n_fail") else ""
    return f"{r['median_iter']:g} / {r['mean_iter']:.2f}{f}"


for name, s in summary["networks"].items():
    for setname in ("timeseries", "independent"):
        r = s[setname]
        lin = r["pgm_linear"]
        lin_txt = "diverges (all)" if lin["n_fail"] == lin.get("n_fail") and lin["median_iter"] is None else f"{lin['median_iter']:g} / {lin['mean_iter']:.2f}"
        b = r["pgm_best_existing"]
        bt = f"{b}: {r[f'pgm_{b}']['median_iter']:g} / {r[f'pgm_{b}']['mean_iter']:.2f}"
        ps = r["previous_step"]
        lines.append(f"| {name} | {s['n_bus']} ({s['n_pv']}) | {setname} | {lin_txt} | {bt} | {cell(ps)} | "
                     f"{cell(r['previous_step_lag4']) if setname == 'timeseries' else 'n/a'} | "
                     + " | ".join(cell(r[f'perturbed_{sg}']) for sg in SIG)
                     + f" | {r[f'pgm_{b}']['median_time_per_iter_us']:.1f} us | "
                     f"{ps['projected_pgm_us_saved_per_scenario']:.1f} us ({100*ps['projected_fraction_of_pgm_batch_time']:.0f}% of batch time) |")
md.append("Cells: median / mean Newton-Raphson iterations (PGM counting, tol 1e-8).\n")
md.extend(lines)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
(out_dir / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
