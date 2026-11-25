from pathlib import Path
import json
from power_grid_model import PowerGridModel, CalculationMethod
from power_grid_model.utils import json_deserialize_from_file
from power_grid_model._core.logger import Logger, LoggerType
d = json_deserialize_from_file(Path("power-grid-model/tests/data/power_flow/ieee14/input.json"))
m = PowerGridModel(d)
for lt in (LoggerType.info, LoggerType.benchmark):
    with Logger(lt) as log:
        m.calculate_power_flow(calculation_method=CalculationMethod.newton_raphson, error_tolerance=1e-8)
    print(lt, repr(log.output[:3000]))
