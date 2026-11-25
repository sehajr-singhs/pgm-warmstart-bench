# Warm-start benchmarks for Power Grid Model's Newton-Raphson

Scripts and raw results behind my comment on
[PowerGridModel/power-grid-model#1609](https://github.com/PowerGridModel/power-grid-model/issues/1609)
(user-supplied starting voltages for iterative power flow).

There are two runs:

1. **Reimplementation** (`bench/pgm_nr.py`, `bench/sweep.py`): Power Grid Model's symmetric Newton-Raphson,
   rewritten in NumPy/SciPy so that starts PGM did not have yet could be tried. It is checked against
   PGM 1.13.214: same iteration counts in every scenario where both converge, voltages within 2.4e-13 p.u.
   Results: `results/reimplementation_v2.json`.
2. **The real branch** (`bench/sweep_branch.py`, `kaggle/branch/run.py`): a build of
   `pgm/experiment/custom-starting-nodal-states-for-power-flow` at f01dd8d (GCC 14), with the start passed
   through the new `node.u_pu` / `node.u_angle` inputs. Results: `results/branch.json`.

## Setup

Four grids, the input files are in `data/` (from PGM's `tests/data/power_flow`, MPL-2.0, plus the IEEE 118-bus
case from PR #1720): `ieee14`, `matpower-case300/flat-start`, `vision-validation-network`, `ieee118`.
For each: a 480-step time series (5 days at 15 minutes, a smooth daily profile plus AR(1) noise per load,
3% standard deviation) and 200 independent random load scenarios, seeded (20261028 + grid index).
Newton-Raphson with tolerance 1e-8 and at most 30 iterations.

Starts compared: PGM's default (linear; flat on case300, where linear diverges), the previous scenario's converged
solution, and the converged solution with 1% or 5% Gaussian noise on magnitude and angle (a stand-in for a
learned or estimated start of a given quality).

## Main results (real branch)

| grid | set | iterations, default start | iterations, previous step | measured time saved |
|---|---|---|---|---|
| ieee14 | time series | 4.22 | 3.00 | 2.7% |
| ieee118 | time series | 5.78 | 3.00 | 17.4% |
| ieee118 | independent | 6.00 | 3.99 | 15.0% |
| matpower-case300 | time series | 6.00 | 3.97 | 15.5% |
| matpower-case300 | independent | 6.02 | 4.90 | 7.5% |
| vision-validation-network | time series | 3.00 | 3.01 | 1.8% |
| vision-validation-network | independent | 3.00 | 4.00 | -0.4% |

- The branch's iteration counts (logger tag 3253, cross-checked by probing `max_iterations`) match the
  reimplementation in 2,712 of 2,712 previous-step scenarios.
- Warm-started results agree with cold-start results to within 2e-13 p.u., with one exception: with 5% noise on
  case300, 25 of 480 time-series runs converged and at least one of them reached a different solution,
  1.02 p.u. away.
- Times are single `calculate_power_flow` calls with the model already built, median of 15 repeats, first 40
  scenarios of each set, on one Kaggle CPU type. They are not batch timings.

## Reproduce

```bash
pip install git+https://github.com/PowerGridModel/power-grid-model@pgm/experiment/custom-starting-nodal-states-for-power-flow
# (needs GCC 14 or newer: CC=gcc-14 CXX=g++-14)
python bench/sweep_branch.py data results/branch.json results/reimplementation_v2.json
```

The reimplementation run: `python bench/sweep.py <power-grid-model checkout> results.json`, then
`python bench/analyze.py results.json summary/`.

## Limits

Synthetic load profiles with one seed, small to medium grids, symmetric Newton-Raphson only, one machine type.

Sehaj Randhir Singh. MPL-2.0, like Power Grid Model.
