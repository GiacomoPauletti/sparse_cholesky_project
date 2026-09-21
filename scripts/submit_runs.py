"""Submit every profile_NS run that scripts/plot_runs.py needs.

    python3 scripts/submit_runs.py              # sbatch everything
    python3 scripts/submit_runs.py --local      # run it here instead, sequentially
    python3 scripts/submit_runs.py --dry-run    # just list what would be submitted

Reports land in plots/runs/, one file per parameter combination, named

    n<N>_<solver>_<ordering>_tol<TOL>_steps<STEPS>[_<factorization>].txt

The factorization suffix is omitted for the up-looking one, which is the
default : that keeps the names of the runs made before the multifrontal
factorization existed, so an existing plots/runs/ is not invalidated.

The name only has to be unique : every report states its own parameters in its
header, and plot_runs.py reads them from there rather than from the file name.
Runs whose report already exists are skipped, so an interrupted sweep can be
resumed by running this again (--force re-runs them).

The sweep is three groups, one per question plot_runs.py answers :

  A. phases    : the Cholesky phases against mesh size, in both the natural
                 and the nested dissection ordering. One time step is enough --
                 the phases being measured all happen once, at setup, except
                 the substitutions which happen twice per step. CG is
                 irrelevant here, and so is the tolerance.

  B. size      : total cost against mesh size, Cholesky vs CG at several
                 tolerances, at a fixed number of time steps.

  C. steps     : total cost against the number of time steps, at a fixed mesh.
                 This is the one that shows the crossover : Cholesky pays a
                 large fixed factorization then solves cheaply, CG pays nothing
                 upfront and something on every solve.

Every Cholesky group is run twice over, once per factorization algorithm, but
the multifrontal one only in the nested dissection ordering : the natural
ordering's fill in makes its fronts large enough that the run is expensive
without saying anything the nested one does not say better. The two
factorizations produce the same factor, so anything differing between them is
a pure timing result -- fill in, nnz(L) and the solve are identical.

Jobs are independent, so the scheduler runs them in parallel ; the whole sweep
is a few dozen jobs of at most a couple of minutes each.
"""

import argparse
import itertools
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
RUNS_DIR = os.path.join(ROOT_DIR, "plots", "runs")
SLURM_SCRIPT = os.path.join(SCRIPT_DIR, "slurm_seq.sh")
DEFAULT_BIN = os.path.join(ROOT_DIR, "build", "release", "profile_NS")

# Sphere subdivision levels. elements = 12 n^2, DoF = 6 n^2 + 2, so n = 64 is
# ~25k DoF : with the current ordering the factorization there already takes
# tens of seconds and a fair amount of memory. Raise with care.
MESH_SIZES = [4, 8, 12, 16, 24, 32, 48, 64]

# CG tolerances to compare against the direct solver.
TOLERANCES = ["1e-6", "1e-8", "1e-10"]

# Vertex orderings the Cholesky backend is run in. CG never factorizes, so it
# is run once, under the natural label.
ORDERINGS = ["natural", "nested"]

# (ordering, factorization) pairs the Cholesky backend is run in. Multifrontal
# is nested only, see the module docstring.
CHOLESKY_VARIANTS = [("natural", "uplooking"),
                     ("nested", "uplooking"),
                     ("nested", "multifrontal")]

# Group B : one mesh scan, at this many time steps.
STEPS_FOR_SIZE_SCAN = 20
# Group C : one time step scan, at this mesh size.
MESH_FOR_STEPS_SCAN = 32
STEPS_SCAN = [1, 2, 5, 10, 20, 50, 100]

# Group B and C only need the larger meshes to say something about the
# asymptotics ; the smallest ones are dominated by noise. Group A keeps them
# all, since its phases are cheap and the extra points help the fit.
SIZE_SCAN_SIZES = [n for n in MESH_SIZES if n >= 8]


def report_name(n, solver, tol, steps, ordering, fact):
    # No suffix for the default factorization : see the module docstring.
    suffix = "" if fact == "uplooking" else f"_{fact}"
    return (f"n{n:03d}_{solver}_{ordering}_tol{tol}"
            f"_steps{steps:04d}{suffix}.txt")


def sweep():
    """-> ordered list of (n, solver, tol, steps, ordering, fact), deduped.

    CG never factorizes, so its runs carry the up-looking label ; it is only
    there to keep every tuple the same shape, and nothing reads it back.
    """
    runs = []

    # A. Cholesky phases against mesh size. tol is ignored by the direct
    #    solver ; it is still spelled out so the file name stays uniform.
    #    This group is what the up-looking vs multifrontal scaling figure is
    #    drawn from, hence every mesh size for both factorizations.
    for n in MESH_SIZES:
        for ordering, fact in CHOLESKY_VARIANTS:
            runs.append((n, "cholesky", "1e-8", 1, ordering, fact))

    # B. Cholesky vs CG against mesh size.
    for n in SIZE_SCAN_SIZES:
        for ordering, fact in CHOLESKY_VARIANTS:
            runs.append((n, "cholesky", "1e-8", STEPS_FOR_SIZE_SCAN, ordering,
                         fact))
        for tol in TOLERANCES:
            runs.append((n, "cg", tol, STEPS_FOR_SIZE_SCAN, "natural",
                         "uplooking"))

    # C. Cholesky vs CG against the number of time steps.
    for steps in STEPS_SCAN:
        for ordering, fact in CHOLESKY_VARIANTS:
            runs.append((MESH_FOR_STEPS_SCAN, "cholesky", "1e-8", steps,
                         ordering, fact))
        for tol in TOLERANCES:
            runs.append((MESH_FOR_STEPS_SCAN, "cg", tol, steps, "natural",
                         "uplooking"))

    seen = set()
    unique = []
    for r in runs:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true",
                    help="run the sweep here, sequentially, instead of submitting it")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the runs without doing anything")
    ap.add_argument("--force", action="store_true",
                    help="re-run combinations whose report already exists")
    ap.add_argument("--bin", default=DEFAULT_BIN,
                    help="path to profile_NS, in both modes")
    args = ap.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)
    runs = sweep()

    todo = []
    skipped = 0
    for n, solver, tol, steps, ordering, fact in runs:
        out = os.path.join(RUNS_DIR,
                           report_name(n, solver, tol, steps, ordering, fact))
        if os.path.exists(out) and not args.force:
            skipped += 1
            continue
        todo.append((n, solver, tol, steps, ordering, fact, out))

    print(f"{len(runs)} run(s) in the sweep, {skipped} already done, "
          f"{len(todo)} to go.")
    if not todo:
        print("Nothing to do. Plot with : python3 scripts/plot_runs.py")
        return

    if args.dry_run:
        for n, solver, tol, steps, ordering, fact, out in todo:
            print(f"  n={n:<3} {solver:<8} {ordering:<8} {fact:<12} "
                  f"tol={tol:<6} steps={steps:<4} -> {os.path.basename(out)}")
        return

    # Checked up front in both modes : failing here beats discovering it from
    # 79 job scripts that each died on their first line.
    if not os.access(args.bin, os.X_OK):
        print(f"profile_NS not found or not executable at {args.bin}")
        print("build it first, or point --bin at it :")
        print("  cmake --build build/release --target profile_NS")
        print("  python3 scripts/submit_runs.py --bin /path/to/profile_NS")
        sys.exit(1)
    binary = os.path.abspath(args.bin)

    if args.local:
        for i, (n, solver, tol, steps, ordering, fact, out) in enumerate(todo,
                                                                         1):
            print(f"[{i}/{len(todo)}] n={n} {solver} {ordering} {fact} "
                  f"tol={tol} steps={steps}", flush=True)
            res = subprocess.run(
                [binary, str(n), solver, tol, str(steps), out, ordering, fact],
                stdout=subprocess.DEVNULL)
            if res.returncode != 0:
                print(f"  failed (exit {res.returncode})")
                sys.exit(1)
    else:
        if not os.access(SLURM_SCRIPT, os.X_OK):
            print(f"{SLURM_SCRIPT} is not executable (chmod +x it).")
            sys.exit(1)
        # slurm_seq.sh resolves the binary on its own, defaulting to
        # ./profile_NS relative to the directory it is called from. Hand it
        # our --bin through the environment so that this script stays the
        # single place where the path is configured, whichever mode is used.
        env = dict(os.environ, PROFILE_NS=binary)
        for n, solver, tol, steps, ordering, fact, out in todo:
            print(f"submitting n={n:<3} {solver:<8} {ordering:<8} "
                  f"{fact:<12} tol={tol:<6} steps={steps:<4}", flush=True)
            res = subprocess.run(
                [SLURM_SCRIPT, str(n), solver, tol, str(steps), out,
                 ordering, fact], env=env)
            if res.returncode != 0:
                print(f"  submission failed")
                sys.exit(1)

    print()
    print("Once every job has finished, plot with :")
    print("  python3 scripts/plot_runs.py")


if __name__ == "__main__":
    main()
