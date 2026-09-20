"""Empirical complexity of the sparse Cholesky phases.

Sweeps the sphere subdivision level, runs profile_NS once per size, and plots
the cost of each Cholesky phase against the number of elements, log-log, with
a fitted exponent per phase.

    # run the sweep locally, then plot :
    python3 scripts/cholesky_complexity.py

    # submit the sweep to SLURM instead, wait for the jobs, then plot :
    python3 scripts/cholesky_complexity.py --submit
    python3 scripts/cholesky_complexity.py --plot-only

The sphere is a subdivided cube : each of the 6 faces is cut into n x n quads,
each quad into 2 triangles, so

    elements = 12 * n^2        vertices (DoF) = 6 * n^2 + 2

The counts are read from the report files rather than recomputed, so a change
in the mesh generator cannot silently invalidate the x axis.

What is plotted, per phase, is the cost of ONE invocation : a run builds two
factorizations (the stream operator and the vorticity operator) and performs
two triangular solves per time step, and the totals in the report are summed
over all of them. Dividing by the call count is what makes the sizes
comparable, since the number of time steps need not be the same across a
sweep.
"""

import argparse
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt

# scripts/cholesky_complexity.py -> ../plots is the shared output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
REPORTS_DIR = os.path.join(PLOTS_DIR, "cholesky_complexity")

DEFAULT_BIN = os.path.join(ROOT_DIR, "build", "release", "profile_NS")
DEFAULT_SUBDIVS = [4, 8, 12, 16, 24, 32, 48, 64]
DEFAULT_STEPS = 20

# The Cholesky steps of interest, in pipeline order. The keys are the step
# names emitted by the instrumentation in cholesky.cpp, matched on the stripped
# (de-indented) name.
PHASES = [
    ("ordering", "ordering"),
    ("symbolic", "symbolic"),
    ("factorization", "factorization"),
    ("transpose (build L^T)", "transpose L -> L^T"),
    ("forward substitution (L y = b)", "forward substitution"),
    ("backward substitution (L^T x = y)", "backward substitution"),
]

SURFACE = "#ffffff"
# Six categorical slots. The three setup phases are blues/purples and the two
# substitutions warm, so the two families separate even before reading the
# labels ; every curve is also direct-labelled, so identity never rests on
# color alone.
COLORS = {
    "ordering": "#7a5bd6",
    "symbolic": "#2a78d6",
    "factorization": "#0f5a8f",
    "transpose L -> L^T": "#3fa2a6",
    "forward substitution": "#eb6834",
    "backward substitution": "#b8452a",
}
INK = "#0b0b0b"    # titles and axis labels
MUTED = "#52514e"  # ticks, spines, guide lines -- 7.9:1 on white
GRID = "#e3e2de"   # recessive, must stay well below the marks


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

# " key : value" lines of the header, e.g. " elements (triangles) : 3072"
INFO_RE = re.compile(r"^\s{1}([a-zA-Z][^:]*?)\s*:\s*(.*?)\s*$")

# A table row : name, then calls and five floats. The name may contain spaces,
# so the numbers are peeled off the right hand side.
ROW_RE = re.compile(
    r"^(?P<name>.*?)\s+"
    r"(?P<calls>\d+)\s+"
    r"(?P<total>[-\d.eE+]+)\s+"
    r"(?P<self>[-\d.eE+]+)\s+"
    r"(?P<avg>[-\d.eE+]+)\s+"
    r"(?P<min>[-\d.eE+]+|-)\s+"
    r"(?P<max>[-\d.eE+]+|-)\s+"
    r"(?P<pct>[-\d.eE+]+)\s*$"
)


def parse_report(path):
    """-> dict with the header info and, per step name, (calls, total_ms).

    Totals are summed over every node carrying that name : a step entered
    under two different parents (the stream solver and the vorticity solver,
    typically) appears twice in the tree and both are wanted here.
    """
    info = {}
    steps = {}

    # The report has three horizontal rules : one above the column header, one
    # below it, and one closing the table. Rows are the lines between the
    # second and the third, which is why a plain toggle is not enough.
    seen_column_header = False
    in_table = False
    # Header info lines stop at the second "====" rule, after which the legend
    # ("total : cumulated time ...") would match the same shape.
    rules = 0

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            stripped = line.strip()

            if stripped and set(stripped) == {"="}:
                rules += 1
                continue
            if stripped and set(stripped) == {"-"}:
                if seen_column_header and not in_table:
                    in_table = True
                elif in_table:
                    in_table = False
                continue

            if not in_table:
                if stripped.startswith("name") and "calls" in stripped:
                    seen_column_header = True
                    continue
                if rules == 1:
                    m = INFO_RE.match(line)
                    if m:
                        info[m.group(1).strip()] = m.group(2).strip()
                continue

            m = ROW_RE.match(line)
            if not m:
                continue
            name = m.group("name").strip()
            if name.endswith("/"):  # a section, not a measurement
                continue
            calls = int(m.group("calls"))
            total = float(m.group("total"))
            prev_calls, prev_total = steps.get(name, (0, 0.0))
            steps[name] = (prev_calls + calls, prev_total + total)

    return {"info": info, "steps": steps}


def report_row(path):
    """-> dict of the numbers this plot needs, or None if the file is unusable."""
    rep = parse_report(path)
    info = rep["info"]

    try:
        elements = int(info["elements (triangles)"])
        dofs = int(info["vertices (DoF)"])
        subdiv = int(info["sphere subdivision"])
    except (KeyError, ValueError):
        print(f"  skipping {os.path.basename(path)} : no mesh size in the header")
        return None

    if info.get("solver") != "cholesky":
        return None

    row = {"n": subdiv, "elements": elements, "dofs": dofs, "path": path}
    for step_name, label in PHASES:
        if step_name not in rep["steps"]:
            row[label] = None
            continue
        calls, total = rep["steps"][step_name]
        # Cost of a single invocation, the only size-comparable quantity.
        row[label] = (total / calls) if calls else None
    return row


# ---------------------------------------------------------------------------
# Running the sweep
# ---------------------------------------------------------------------------

def report_path(subdiv):
    return os.path.join(REPORTS_DIR, f"cholesky_n{subdiv:03d}.txt")


def run_sweep(binary, subdivs, steps):
    if not os.access(binary, os.X_OK):
        print(f"profile_NS not found or not executable at {binary}")
        print("build it first, e.g. :")
        print("  cmake --build build/release --target profile_NS")
        sys.exit(1)

    for n in subdivs:
        out = report_path(n)
        print(f"running n = {n} ...", flush=True)
        # The report path is the 5th argument : without it every run of the
        # sweep would write the same performance.txt.
        res = subprocess.run(
            [binary, str(n), "cholesky", "1e-8", str(steps), out],
            stdout=subprocess.DEVNULL,
        )
        if res.returncode != 0:
            print(f"  profile_NS failed for n = {n} (exit {res.returncode})")
            sys.exit(1)


def submit_sweep(subdivs, steps):
    script = os.path.join(SCRIPT_DIR, "slurm_seq.sh")
    for n in subdivs:
        out = report_path(n)
        print(f"submitting n = {n} -> {out}", flush=True)
        res = subprocess.run(
            [script, str(n), "cholesky", "1e-8", str(steps), out]
        )
        if res.returncode != 0:
            print(f"  submission failed for n = {n}")
            sys.exit(1)
    print()
    print("Jobs submitted. Once they have all finished, plot with :")
    print("  python3 scripts/cholesky_complexity.py --plot-only")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def fit_exponent(xs, ys):
    """Least squares slope of log(y) against log(x), i.e. p in y ~ x^p."""
    pts = [(x, y) for x, y in zip(xs, ys) if x > 0 and y is not None and y > 0]
    if len(pts) < 2:
        return None
    import math
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    n = len(pts)
    mx = sum(lx) / n
    my = sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return (num / den) if den else None


def new_axes(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=12, pad=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, which="both", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def savefig(name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150, facecolor=SURFACE)
    print(f"saved -> {path}")
    plt.close()


def write_csv(rows):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "cholesky_complexity.csv")
    labels = [label for _, label in PHASES]
    with open(path, "w") as f:
        f.write("n,elements,dofs," + ",".join(labels.copy()) + "\n")
        for r in rows:
            vals = ["" if r[l] is None else f"{r[l]:.6f}" for l in labels]
            f.write(f"{r['n']},{r['elements']},{r['dofs']}," + ",".join(vals) + "\n")
    print(f"saved -> {path}")


def declutter(labels, ax):
    """Push apart labels sharing the right edge, in log space.

    Six phases plus two guide lines end up within a decade of each other at the
    right of the plot, so the natural y of each label is only a starting point.
    """
    import math
    ymin, ymax = ax.get_ylim()
    span = math.log10(ymax) - math.log10(ymin)
    gap = 0.042 * span

    labels = sorted(labels, key=lambda e: e["y"])
    ys = [math.log10(e["y"]) for e in labels]

    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    # If the stack has grown past the top, slide the whole thing back down.
    overflow = ys[-1] - (math.log10(ymax) - 0.5 * gap)
    if overflow > 0:
        ys = [y - overflow for y in ys]

    for e, y in zip(labels, ys):
        ax.annotate(
            e["text"], xy=(e["x"], 10 ** y), xytext=(7, 0),
            textcoords="offset points", fontsize=8.5, color=e["color"],
            va="center", zorder=4,
        )


def plot(rows):
    rows = sorted(rows, key=lambda r: r["elements"])
    xs = [r["elements"] for r in rows]

    fig, ax = new_axes(
        "Sparse Cholesky : cost of each phase against mesh size",
        "elements (triangles)  [= 12 n\u00b2 for a sphere of subdivision n]",
        "time of one invocation (ms)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    labels = []
    exponents = {}
    for _, label in PHASES:
        ys = [r[label] for r in rows]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None and y > 0]
        if len(pts) < 2:
            print(f"  '{label}' : not enough nonzero data to plot")
            continue
        px = [p[0] for p in pts]
        py = [p[1] for p in pts]
        p = fit_exponent(px, py)
        exponents[label] = p

        ax.plot(px, py, marker="o", markersize=4, linewidth=1.6,
                color=COLORS[label], zorder=3)
        # Direct labels, so the reader never has to bounce between a legend
        # and the marks. declutter() sorts out the ones that would overlap.
        labels.append({
            "x": px[-1], "y": py[-1], "color": COLORS[label],
            "text": f"{label}  (\u221d E^{p:.2f})" if p is not None else label,
        })

    # Reference slopes, anchored on the factorization curve so that they sit
    # among the data rather than off in a corner.
    anchor = None
    for x, r in zip(xs, rows):
        if r.get("factorization"):
            anchor = (x, r["factorization"])
            break
    if anchor:
        x0, y0 = anchor
        xr = [x0, xs[-1]]
        for power, name in ((1.0, "E"), (1.5, "E^1.5")):
            yr = [y0 * (x / x0) ** power for x in xr]
            ax.plot(xr, yr, linestyle=":", linewidth=1.0, color=MUTED, zorder=1)
            labels.append({"x": xr[-1], "y": yr[-1], "color": MUTED,
                           "text": f"O({name})"})

    ax.set_xlim(min(xs) * 0.8, max(xs) * 4.5)
    declutter(labels, ax)
    savefig("cholesky_complexity.png")

    print()
    print("Fitted exponents, cost \u221d elements^p :")
    for _, label in PHASES:
        if label in exponents and exponents[label] is not None:
            print(f"  {label:<24} p = {exponents[label]:.2f}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subdivs", nargs="*", type=int, default=None,
                    help=f"sphere subdivision levels (default {DEFAULT_SUBDIVS})")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help=f"time steps per run (default {DEFAULT_STEPS})")
    ap.add_argument("--bin", default=DEFAULT_BIN, help="path to profile_NS")
    ap.add_argument("--submit", action="store_true",
                    help="submit the sweep through scripts/slurm_seq.sh instead of running it")
    ap.add_argument("--plot-only", action="store_true",
                    help="do not run anything, just plot the reports already collected")
    args = ap.parse_args()

    subdivs = args.subdivs if args.subdivs else DEFAULT_SUBDIVS
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if args.submit:
        submit_sweep(subdivs, args.steps)
        return
    if not args.plot_only:
        run_sweep(args.bin, subdivs, args.steps)

    paths = sorted(
        os.path.join(REPORTS_DIR, f)
        for f in os.listdir(REPORTS_DIR)
        if f.endswith(".txt")
    )
    if not paths:
        print(f"no report found in {REPORTS_DIR}")
        print("run the sweep first : python3 scripts/cholesky_complexity.py")
        sys.exit(1)

    rows = [r for r in (report_row(p) for p in paths) if r]
    if len(rows) < 2:
        print("need at least two sizes to show a trend")
        sys.exit(1)

    print(f"\n{len(rows)} Cholesky report(s) :")
    for r in sorted(rows, key=lambda r: r["elements"]):
        print(f"  n = {r['n']:>3}  elements = {r['elements']:>7}  DoF = {r['dofs']:>7}")
    print()

    write_csv(sorted(rows, key=lambda r: r["elements"]))
    plot(rows)


if __name__ == "__main__":
    main()
