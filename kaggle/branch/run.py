import os, subprocess, sys, shutil, time
t = time.time()
subprocess.run("apt-get update -qq && apt-get install -y -qq g++-14 gcc-14", shell=True, check=True)
os.environ["CC"] = "gcc-14"; os.environ["CXX"] = "g++-14"
B = "pgm/experiment/custom-starting-nodal-states-for-power-flow"
subprocess.check_call([sys.executable, "-m", "pip", "install", f"git+https://github.com/PowerGridModel/power-grid-model@{B}"],
                      env={**os.environ, "CMAKE_BUILD_PARALLEL_LEVEL": str(os.cpu_count())})
print("build minutes", (time.time() - t) / 60, flush=True)
src = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "sweep_branch.py" in files:
        src = os.path.dirname(root); break
work = "/kaggle/working/src"; shutil.copytree(src, work, dirs_exist_ok=True)
subprocess.check_call([sys.executable, "bench/sweep_branch.py", "data", "/kaggle/working/results_branch.json", "v2_results.json"], cwd=work)
