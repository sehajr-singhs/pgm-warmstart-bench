"""Bundle bench/pgm_nr.py, bench/sweep.py and the local ieee118 case into one Kaggle script kernel."""
import base64, json
from pathlib import Path
root = Path(__file__).resolve().parent.parent
files = {
    "bench/pgm_nr.py": root / "bench/pgm_nr.py",
    "bench/sweep.py": root / "bench/sweep.py",
    "pgm/tests/data/power_flow/ieee118/input.json": root / "power-grid-model/tests/data/power_flow/ieee118/input.json",
}
blob = {k: base64.b64encode(v.read_bytes()).decode() for k, v in files.items()}
script = f'''# Warm-start benchmark for Power Grid Model Newton-Raphson (bundled by bench/build_kernel.py)
import base64, os, subprocess, sys
from pathlib import Path
W = Path("/tmp/pgm_work")
OUT = Path("/kaggle/working")
W.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "power-grid-model==1.13.214"], check=True)
subprocess.run(["git", "clone", "-q", "https://github.com/PowerGridModel/power-grid-model.git", str(W / "pgm")], check=True)
subprocess.run(["git", "-C", str(W / "pgm"), "checkout", "-q", "4ebe96910ccf8473c11cf3bcdd30e37386a536b6"], check=True)
FILES = {json.dumps(blob)}
for rel, b64 in FILES.items():
    p = W / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(b64))
subprocess.run(["lscpu"], check=False)
r = subprocess.run([sys.executable, str(W / "bench/sweep.py"), str(W / "pgm"), str(OUT / "results.json")],
                   cwd=str(W / "bench"), check=False)
sys.exit(r.returncode)
'''
out = root / "kaggle/pgm-warmstart/run.py"
out.write_text(script)
meta = {"id": "sehajrsingh/pgm-warmstart-bench", "title": "pgm-warmstart-bench", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": []}
(root / "kaggle/pgm-warmstart/kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print("bundled", out.stat().st_size, "bytes")
