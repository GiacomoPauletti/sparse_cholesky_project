"""Submit the parallel factorization sweep : cost against mesh size, one curve
per thread count. Plotted by plot_runs.py (figure 7).

    python3 scripts/submit_runs_par.py              # sbatch everything
    python3 scripts/submit_runs_par.py --local      # run it here, sequentially
    python3 scripts/submit_runs_par.py --dry-run    # just list the runs

Every run is Cholesky, nested ordering, one time step (the factorization
happens at setup). Besides the parallel multifrontal at each thread count, the
sequential multifrontal is run once per size : the parallel code at 1 thread
against it measures the scheduling overhead. Reports go to plots/runs_par/;
existing ones are skipped unless --force.
"""

import argparse
import os
import subprocess
import sys

from submit_runs import DEFAULT_BIN, ROOT_DIR, SCRIPT_DIR

RUNS_DIR = os.path.join(ROOT_DIR, "plots", "runs_par")
SLURM_SCRIPT = os.path.join(SCRIPT_DIR, "slurm_par.sh")

MESH_SIZES = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
# Doubling : the elimination tree caps the speedup near 2x with the current
# ordering, so intermediate counts would only add points on a flat curve.
THREADS = [1, 2, 4, 8, 16]


def sweep():
    """-> list of (n, threads, fact)."""
    runs = []
    for n in MESH_SIZES:
        runs.append((n, 1, "multifrontal"))
        runs += [(n, t, "parmultifrontal") for t in THREADS]
    return runs


def report_name(n, threads, fact):
    return f"n{n:03d}_{fact}_t{threads:02d}.txt"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bin", default=DEFAULT_BIN)
    args = ap.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)
    todo = [(n, t, f, os.path.join(RUNS_DIR, report_name(n, t, f)))
            for n, t, f in sweep()]
    total = len(todo)
    if not args.force:
        todo = [r for r in todo if not os.path.exists(r[3])]
    print(f"{total} run(s), {total - len(todo)} already done, {len(todo)} to go.")

    if args.dry_run or not todo:
        for n, t, f, out in todo:
            print(f"  n={n:<3} threads={t:<2} {f:<15} -> {os.path.basename(out)}")
        return

    if not os.access(args.bin, os.X_OK):
        sys.exit(f"profile_NS not found or not executable at {args.bin}")
    binary = os.path.abspath(args.bin)

    for i, (n, t, f, out) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] n={n} threads={t} {f}", flush=True)
        if args.local:
            env = dict(os.environ, OMP_NUM_THREADS=str(t))
            cmd = [binary, str(n), "cholesky", "1e-8", "1", out, "nested", f]
            res = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL)
        else:
            env = dict(os.environ, PROFILE_NS=binary)
            res = subprocess.run([SLURM_SCRIPT, str(n), str(t), out, f], env=env)
        if res.returncode != 0:
            sys.exit(f"  failed (exit {res.returncode})")

    print("\nOnce every job has finished, plot with :  python3 scripts/plot_runs.py")


if __name__ == "__main__":
    main()
