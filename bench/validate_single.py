"""Quick check: external NR vs PGM on the base case of each network."""
import re, sys
from pathlib import Path
import numpy as np
from power_grid_model import PowerGridModel, CalculationMethod, CalculationInitialization
from power_grid_model.utils import json_deserialize_from_file
from power_grid_model._core.logger import Logger, LoggerType
sys.path.insert(0, str(Path(__file__).parent))
from pgm_nr import Network

ROOT = Path(__file__).resolve().parent.parent / "power-grid-model/tests/data/power_flow"
CASES = {"ieee14": "ieee14", "case300": "matpower-case300/flat-start", "vision": "vision-validation-network"}
for name, rel in CASES.items():
    data = json_deserialize_from_file(ROOT / rel / "input.json")
    net = Network(data)
    sc = net.scenario(data)
    m = PowerGridModel(data)
    for init in ("linear", "flat", "average_source"):
        try:
            with Logger(LoggerType.info) as log:
                out = m.calculate_power_flow(calculation_method=CalculationMethod.newton_raphson,
                                             error_tolerance=1e-8, max_iterations=30,
                                             calculation_initialization=CalculationInitialization[init])
            it_pgm = int(re.findall(r"Tag:3253: (\d+)", log.output)[0])
            u_pgm = np.zeros(net.n, complex)
            for nid, upu, ang in zip(out["node"]["id"], out["node"]["u_pu"], out["node"]["u_angle"]):
                u_pgm[net.id2bus[int(nid)]] = upu * np.exp(1j * ang)
        except Exception as e:
            it_pgm, u_pgm = f"fail:{type(e).__name__}", None
        u0 = getattr(net, f"start_{init}")(sc)
        u, it, ok = net.solve(sc, u0, max_iter=30)
        diff = np.max(np.abs(u - u_pgm)) if u_pgm is not None else None
        print(f"{name:8s} n={net.n:4d} {init:15s} pgm_iter={it_pgm} ext_iter={it} ext_ok={ok} max|du|={diff}")
