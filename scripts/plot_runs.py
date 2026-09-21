"""Plot the profile_NS sweep submitted by scripts/submit_runs.py.

    python3 scripts/submit_runs.py     # (or --local) then wait for the jobs
    python3 scripts/plot_runs.py

Reads every report in plots/runs/ and writes one figure per question into
plots/. Each report states its own parameters in its header, so this script
does not care how the files are named, nor which subset of the sweep actually
finished : every figure is drawn from whatever runs support it, and one that
has too little data is skipped with a message rather than drawn misleadingly.

Figures
    1. cholesky_phases.png   cost of each Cholesky phase against mesh size
    2. total_vs_size.png     Cholesky vs CG total cost against mesh size
    3. total_vs_steps.png    Cholesky vs CG total cost against time steps
    4. cholesky_fillin.png   nnz(L)/nnz(A), the memory cost of the factor
    5. cg_iterations.png     CG iterations per solve against mesh size
    6. cholesky_factorization_scaling.png
                             up-looking vs multifrontal : how the numerical
                             factorization scales, and by how much the second
                             beats the first
    7. parallel_scaling.png  parallel multifrontal against mesh size, one curve
                             per thread count (sweep of submit_runs_par.py,
                             read from plots/runs_par/)

A Cholesky run is identified by its VARIANT, the (ordering, factorization)
pair : the two are independent knobs, the ordering deciding how much fill in
the factor suffers and the factorization only how fast the same factor is
computed. Curves are therefore keyed by the pair, not by the ordering alone,
and the factorization is dropped from the labels when the reports only contain
one of them -- which is what every report written before the multifrontal
factorization existed amounts to.

The sphere is a subdivided cube, so elements = 12 n^2 and DoF = 6 n^2 + 2 at
subdivision n ; both counts are read from the reports, not recomputed here.
"""

import math
import os
import re
import sys

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
RUNS_DIR = os.path.join(PLOTS_DIR, "runs")
PAR_RUNS_DIR = os.path.join(PLOTS_DIR, "runs_par")

# Cholesky phases of figure 1 : the step name emitted by the instrumentation in
# cholesky.cpp, and the label to plot it under. "solve" is synthesized below as
# the sum of the two substitutions, which is what one linear solve costs.
PHASES = [
    ("elimination tree", "tree assembly"),
    ("row patterns", "row pattern construction"),
    ("factorization", "factorization"),
    (None, "solve (forward + backward)"),
]
SUBSTITUTIONS = ["forward substitution (L y = b)",
                 "backward substitution (L^T x = y)"]

SURFACE = "#ffffff"
# Blues and purples for the setup phases, warm for the solve, so the two
# families separate before the labels are read. Checked against protanopia and
# deuteranopia : no pair below 3:1 on this surface, and every curve is direct
# labelled anyway, so identity never rests on color alone.
def variant(run):
    """(ordering, factorization) : what tells two Cholesky curves apart."""
    return (run["ordering"], run["factorization"])


def variant_label(key, show_fact=True):
    ordering, fact = key
    return (f"cholesky ({ordering}, {fact})" if show_fact
            else f"cholesky ({ordering})")


def variant_slug(key):
    """File name fragment for a variant.

    The default factorization contributes nothing, exactly as in the report
    names submit_runs.py writes : the figure of a variant that existed before
    the multifrontal one did keeps the file name it had, so an existing
    plots/ directory gains files rather than being left with stale ones.
    """
    ordering, fact = key
    return ordering if fact == "up-looking" else f"{ordering}_{fact}"


def variant_color(key):
    for name in (variant_label(key, True), variant_label(key, False)):
        if name in COLORS:
            return COLORS[name]
    return COLORS["cholesky"]


def show_factorization(runs):
    """True when the reports hold more than one factorization to compare."""
    return len({r["factorization"] for r in runs
                if r["solver"] == "cholesky"}) > 1


def series_label(run):
    """Name of the curve a run belongs to."""
    if run["solver"] == "cholesky":
        return variant_label(variant(run))
    return f"cg tol {run['tol']}"


COLORS = {
    "cholesky (natural)": "#8c6f1f",
    "cholesky (nested)": "#0f5a8f",
    "cholesky (natural, up-looking)": "#8c6f1f",
    "cholesky (nested, up-looking)": "#0f5a8f",
    # Violet, already in this palette for figure 1's tree assembly and so
    # already checked on this surface. The two never share a figure.
    "cholesky (nested, multifrontal)": "#7a5bd6",
    "cholesky (natural, multifrontal)": "#b0872a",
    "speedup": "#7a5bd6",
    "tree assembly": "#7a5bd6",
    "row pattern construction": "#2a78d6",
    "factorization": "#0f5a8f",
    "solve (forward + backward)": "#eb6834",
    "cholesky": "#0f5a8f",
    "cg 1e-6": "#eb6834",
    "cg 1e-8": "#c2502a",
    "cg 1e-10": "#8c3318",
    "fill in": "#3fa2a6",
}
THREAD_RAMP = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CG_FALLBACK = ["#eb6834", "#c2502a", "#8c3318", "#5c2010"]
INK = "#0b0b0b"    # titles and axis labels
MUTED = "#52514e"  # ticks, spines, guide lines -- 7.9:1 on white
GRID = "#e3e2de"   # recessive, must stay well below the marks


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

INFO_RE = re.compile(r"^\s{1}([a-zA-Z][^:]*?)\s*:\s*(.*?)\s*$")

# A table row : the name may contain spaces, so the six numeric columns are
# peeled off the right hand side.
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
    """-> (info dict, {name: (calls, total_ms)}, {section: (calls, total_ms)}).

    Totals are summed over every node carrying a given name : a step entered
    under two different parents -- the stream solver and the vorticity solver,
    typically -- appears twice in the tree and both are wanted.
    """
    info = {}
    steps = {}
    sections = {}

    # Three horizontal rules : above the column header, below it, and closing
    # the table. Rows are the lines between the second and the third.
    seen_column_header = False
    in_table = False
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
            calls = int(m.group("calls"))
            total = float(m.group("total"))
            target = sections if name.endswith("/") else steps
            key = name.rstrip("/").strip()
            prev_calls, prev_total = target.get(key, (0, 0.0))
            target[key] = (prev_calls + calls, prev_total + total)

    return info, steps, sections


def load_runs(runs_dir=RUNS_DIR):
    if not os.path.isdir(runs_dir):
        return []
    runs = []
    for name in sorted(os.listdir(runs_dir)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(runs_dir, name)
        info, steps, sections = parse_report(path)
        try:
            run = {
                "path": path,
                "n": int(info["sphere subdivision"]),
                "elements": int(info["elements (triangles)"]),
                "dofs": int(info["vertices (DoF)"]),
                "solver": info["solver"],
                "steps": int(info["time steps"]),
            }
        except (KeyError, ValueError):
            print(f"  skipping {name} : incomplete header")
            continue

        run["tol"] = info.get("tolerance")
        # Cholesky reports carry an ordering, CG does not (it never
        # factorizes). Reports written before the ordering became selectable
        # have no such line and were all natural.
        run["ordering"] = (info.get("ordering", "natural")
                           if run["solver"] == "cholesky" else None)
        # Reports written before the factorization became selectable have no
        # such line and were all up-looking. profile_NS spells it "up-looking"
        # and "multifrontal" ; anything else is normalized to the former.
        fact = info.get("factorization", "up-looking").strip().lower()
        if fact not in ("multifrontal", "parallel multifrontal"):
            fact = "up-looking"
        run["factorization"] = fact if run["solver"] == "cholesky" else None
        run["threads"] = int(info.get("threads", 1))
        run["nnz_a"] = int(info["nnz(A)"]) if "nnz(A)" in info else None
        run["nnz_l"] = int(info["nnz(L)"]) if "nnz(L)" in info else None
        run["cg_iter_psi"] = _float(info.get("cg iterations per psi solve"))
        run["cg_iter_omega"] = _float(info.get("cg iterations per omega solve"))
        run["raw_steps"] = steps
        run["sections"] = sections

        # The comparable total : everything but the mesh generation and the
        # initial condition, which are identical for both backends. For
        # Cholesky the stream factorization sits in "solver construction" and
        # the vorticity one inside the first time step, so both have to be in.
        setup = sections.get("solver construction", (0, 0.0))[1]
        loop = sections.get("time loop", (0, 0.0))[1]
        run["setup_ms"] = setup
        run["loop_ms"] = loop
        run["total_ms"] = setup + loop
        runs.append(run)
    return runs


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def phase_cost(run, step_name):
    """Cost of ONE invocation of a step, in ms, or None.

    A run performs two factorizations and two triangular solves per time step,
    and the report totals are summed over all of them ; dividing by the call
    count is what makes runs with different step counts comparable.
    """
    if step_name is None:  # the synthesized "solve" phase
        total = 0.0
        calls = 0
        for name in SUBSTITUTIONS:
            if name not in run["raw_steps"]:
                return None
            c, t = run["raw_steps"][name]
            total += t
            calls = max(calls, c)
        return (total / calls) if calls else None

    if step_name not in run["raw_steps"]:
        return None
    calls, total = run["raw_steps"][step_name]
    return (total / calls) if calls else None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def style_axes(ax, title=None, xlabel=None, ylabel=None, subtitle=None):
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, fontsize=12, pad=18 if subtitle else 10, color=INK)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9,
                color=MUTED, ha="left", va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, which="both", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return ax


def new_axes(title, xlabel, ylabel, subtitle=None):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    style_axes(ax, title, xlabel, ylabel, subtitle)
    return fig, ax


def savefig(name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150, facecolor=SURFACE)
    print(f"saved -> {path}")
    plt.close()


def fit_exponent(xs, ys):
    """Least squares slope of log(y) on log(x), i.e. p in y ~ x^p."""
    pts = [(x, y) for x, y in zip(xs, ys) if x > 0 and y and y > 0]
    if len(pts) < 2:
        return None
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    n = len(pts)
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return (num / den) if den else None


def declutter(labels, ax, logy=True, gap=0.042):
    """Push apart labels sharing the right edge of the plot.

    A label carrying side="left" is drawn to the left of its anchor instead,
    and takes no part in the spreading : it does not share the right edge, so
    it cannot collide with the stack there. That is the escape hatch for a
    curve that stops early, whose right end is in the middle of the plot and
    whose label would otherwise be written straight across its neighbours.
    """
    if not labels:
        return

    for e in [e for e in labels if e.get("side") == "left"]:
        ax.annotate(e["text"], xy=(e["x"], e["y"]), xytext=(-9, 0),
                    textcoords="offset points", fontsize=8.5,
                    color=e["color"], ha="right", va="center", zorder=4)
    labels = [e for e in labels if e.get("side") != "left"]
    if not labels:
        return

    ymin, ymax = ax.get_ylim()
    if logy:
        lo, hi = math.log10(ymin), math.log10(ymax)
        pos = [math.log10(e["y"]) for e in sorted(labels, key=lambda e: e["y"])]
    else:
        lo, hi = ymin, ymax
        pos = [e["y"] for e in sorted(labels, key=lambda e: e["y"])]
    labels = sorted(labels, key=lambda e: e["y"])
    gap = gap * (hi - lo)  # fraction of the axis height

    for i in range(1, len(pos)):
        if pos[i] - pos[i - 1] < gap:
            pos[i] = pos[i - 1] + gap
    overflow = pos[-1] - (hi - 0.5 * gap)
    if overflow > 0:
        pos = [p - overflow for p in pos]

    for e, p in zip(labels, pos):
        y = (10 ** p) if logy else p
        ax.annotate(e["text"], xy=(e["x"], y), xytext=(7, 0),
                    textcoords="offset points", fontsize=8.5,
                    color=e["color"], va="center", zorder=4)


def guide_lines(ax, anchor, x_end, powers, labels):
    """Reference slopes through `anchor`, collected into `labels`."""
    if not anchor:
        return
    x0, y0 = anchor
    for power, name in powers:
        xr = [x0, x_end]
        yr = [y0 * (x / x0) ** power for x in xr]
        ax.plot(xr, yr, linestyle=":", linewidth=1.0, color=MUTED, zorder=1)
        labels.append({"x": xr[-1], "y": yr[-1], "color": MUTED,
                       "text": f"O({name})"})


def cg_color(tol, index):
    return COLORS.get(f"cg {tol}", CG_FALLBACK[index % len(CG_FALLBACK)])


# ---------------------------------------------------------------------------
# Figure 1 : Cholesky phases against mesh size
# ---------------------------------------------------------------------------

def phase_rows(runs, key):
    """Cheapest run per mesh size for one variant, ordered by mesh size.

    One time step is all the setup phases need, but any Cholesky run carries
    them ; the smallest step count is preferred so the numbers come from the
    cheapest runs available.
    """
    by_size = {}
    for r in runs:
        if r["solver"] != "cholesky" or variant(r) != key:
            continue
        cur = by_size.get(r["elements"])
        if cur is None or r["steps"] < cur["steps"]:
            by_size[r["elements"]] = r
    return sorted(by_size.values(), key=lambda r: r["elements"])


def plot_phases(runs, key=None, show_fact=True):
    """Figure 1, for one variant : mixing two orderings would be meaningless,
    since the whole point of the permutation is that it changes these costs,
    and mixing two factorizations would average away the very difference
    figure 6 is about."""
    cholesky = [r for r in runs if r["solver"] == "cholesky"]
    if not cholesky:
        print("figure 1 skipped : no Cholesky run")
        return
    if key is None:
        available = {variant(r) for r in cholesky}
        # Nested is the interesting ordering when it is there.
        key = next((k for k in sorted(available) if k[0] == "nested"),
                   sorted(available)[0])

    name = variant_label(key, show_fact)
    rows = phase_rows(runs, key)
    if len(rows) < 2:
        print(f"figure 1 ({name}) skipped : need two mesh sizes or more")
        return

    ordering, fact = key
    slug = variant_slug(key)
    which = (f"{ordering} ordering, {fact} factorization" if show_fact
             else f"{ordering} ordering")
    xs = [r["elements"] for r in rows]
    fig, ax = new_axes(
        f"Sparse Cholesky, {which} : cost of each phase",
        "elements (triangles)   [= 12 n² at subdivision n]",
        "time of one invocation (ms)",
        f"{len(rows)} mesh sizes, {rows[0]['dofs']} to {rows[-1]['dofs']} DoF",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    labels = []
    exponents = {}
    anchor = None
    for step_name, label in PHASES:
        ys = [phase_cost(r, step_name) for r in rows]
        pts = [(x, y) for x, y in zip(xs, ys) if y and y > 0]
        if len(pts) < 2:
            print(f"  '{label}' : no usable data (all zero or missing)")
            continue
        px = [p[0] for p in pts]
        py = [p[1] for p in pts]
        p = fit_exponent(px, py)
        exponents[label] = p
        if label == "factorization":
            anchor = (px[0], py[0])
        ax.plot(px, py, marker="o", markersize=4, linewidth=1.6,
                color=COLORS[label], zorder=3)
        labels.append({
            "x": px[-1], "y": py[-1], "color": COLORS[label],
            "text": f"{label}  (∝ E^{p:.2f})" if p else label,
        })

    guide_lines(ax, anchor, xs[-1], ((1.0, "E"), (1.5, "E^1.5")), labels)
    ax.set_xlim(min(xs) * 0.8, max(xs) * 5.0)
    declutter(labels, ax)
    savefig(f"cholesky_phases_{slug}.png")

    print(f"\n  fitted exponents ({name}), cost ∝ elements^p :")
    for _, label in PHASES:
        if exponents.get(label):
            print(f"    {label:<28} p = {exponents[label]:.2f}")


# ---------------------------------------------------------------------------
# Figure 2 : total cost against mesh size
# ---------------------------------------------------------------------------

def plot_total_vs_size(runs):
    # Fix the number of time steps : the most represented one, excluding the
    # single step runs of group A which exist only for figure 1.
    counts = {}
    for r in runs:
        if r["steps"] > 1:
            counts[r["steps"]] = counts.get(r["steps"], 0) + 1
    if not counts:
        print("figure 2 skipped : no multi step run")
        return
    steps = max(counts, key=lambda s: (counts[s], s))

    show_fact = show_factorization(runs)
    chols = {}
    for r in runs:
        if r["solver"] == "cholesky" and r["steps"] == steps:
            chols.setdefault(variant(r), []).append(r)
    for o in chols:
        chols[o].sort(key=lambda r: r["elements"])
    cgs = {}
    for r in runs:
        if r["solver"] != "cholesky" and r["steps"] == steps:
            cgs.setdefault(r["tol"], []).append(r)
    for tol in cgs:
        cgs[tol].sort(key=lambda r: r["elements"])

    if not chols and not cgs:
        print("figure 2 skipped : not enough runs at a common step count")
        return

    all_x = []
    for rs in list(chols.values()) + list(cgs.values()):
        all_x += [r["elements"] for r in rs]
    if not all_x:
        print("figure 2 skipped : no usable run")
        return

    fig, ax = new_axes(
        "Cholesky vs conjugate gradient : total cost against mesh size",
        "elements (triangles)   [= 12 n² at subdivision n]",
        f"setup + {steps} time steps (ms)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    labels = []
    anchor = None
    for key in sorted(chols):
        rs = chols[key]
        if len(rs) < 2:
            continue
        px = [r["elements"] for r in rs]
        py = [r["total_ms"] for r in rs]
        p = fit_exponent(px, py)
        if anchor is None:
            anchor = (px[0], py[0])
        label = variant_label(key, show_fact)
        color = variant_color(key)
        ax.plot(px, py, marker="o", markersize=4, linewidth=1.8,
                color=color, zorder=3)
        entry = {"x": px[-1], "y": py[-1], "color": color,
                 "text": f"{label}  (∝ E^{p:.2f})" if p else label}
        # The natural curve stops where its runs stop being affordable, well
        # short of the right edge, so its label is written leftwards from that
        # last point instead of across the curves that carry on past it.
        if px[-1] < max(all_x):
            entry["side"] = "left"
        labels.append(entry)

    for i, tol in enumerate(sorted(cgs, key=lambda t: -float(t or 0))):
        rs = cgs[tol]
        if len(rs) < 2:
            continue
        px = [r["elements"] for r in rs]
        py = [r["total_ms"] for r in rs]
        p = fit_exponent(px, py)
        color = cg_color(tol, i)
        ax.plot(px, py, marker="s", markersize=4, linewidth=1.6,
                color=color, zorder=3)
        labels.append({"x": px[-1], "y": py[-1], "color": color,
                       "text": f"cg tol {tol}  (∝ E^{p:.2f})" if p
                               else f"cg tol {tol}"})

    guide_lines(ax, anchor, max(all_x), ((1.0, "E"), (1.5, "E^1.5")), labels)
    ax.set_xlim(min(all_x) * 0.8, max(all_x) * 5.0)
    declutter(labels, ax)
    savefig("total_vs_size.png")


# ---------------------------------------------------------------------------
# Figure 3 : total cost against the number of time steps
# ---------------------------------------------------------------------------

def plot_total_vs_steps(runs):
    # Fix the mesh : whichever size has the most distinct step counts.
    counts = {}
    for r in runs:
        counts.setdefault(r["elements"], set()).add(r["steps"])
    if not counts:
        print("figure 3 skipped : no run")
        return
    elements = max(counts, key=lambda e: (len(counts[e]), e))
    if len(counts[elements]) < 2:
        print("figure 3 skipped : no mesh size run at two step counts or more")
        return

    here = [r for r in runs if r["elements"] == elements]
    show_fact = show_factorization(runs)
    chols = {}
    for r in here:
        if r["solver"] == "cholesky":
            chols.setdefault(variant(r), []).append(r)
    for o in chols:
        chols[o].sort(key=lambda r: r["steps"])
    cgs = {}
    for r in here:
        if r["solver"] != "cholesky":
            cgs.setdefault(r["tol"], []).append(r)
    for tol in cgs:
        cgs[tol].sort(key=lambda r: r["steps"])

    dofs = here[0]["dofs"]
    n = here[0]["n"]

    fig, ax = new_axes(
        "Cholesky vs conjugate gradient : where the direct method pays off",
        "time steps",
        "setup + time loop (ms)",
        f"fixed mesh : subdivision {n}, {elements} elements, {dofs} DoF",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    labels = []
    for key in sorted(chols):
        rs = chols[key]
        if len(rs) < 2:
            continue
        px = [r["steps"] for r in rs]
        py = [r["total_ms"] for r in rs]
        label = variant_label(key, show_fact)
        color = variant_color(key)
        ax.plot(px, py, marker="o", markersize=4, linewidth=1.8,
                color=color, zorder=3)
        labels.append({"x": px[-1], "y": py[-1], "color": color,
                       "text": label})

    for i, tol in enumerate(sorted(cgs, key=lambda t: -float(t or 0))):
        rs = cgs[tol]
        if len(rs) < 2:
            continue
        px = [r["steps"] for r in rs]
        py = [r["total_ms"] for r in rs]
        color = cg_color(tol, i)
        ax.plot(px, py, marker="s", markersize=4, linewidth=1.6,
                color=color, zorder=3)
        labels.append({"x": px[-1], "y": py[-1], "color": color,
                       "text": f"cg tol {tol}"})

    # The crossover : the step count beyond which Cholesky is cheaper. Read off
    # the measured curves rather than a model, by linear interpolation in the
    # log-log plane of the first sign change.
    # Every ordering is checked, not just one : a permutation that cuts the
    # factorization by an order of magnitude is exactly what moves the
    # crossover into range, so reporting only the natural one would hide the
    # result. The annotation goes on the earliest crossing found.
    found = []
    for key, rs in chols.items():
        if len(rs) < 2:
            continue
        for tol in sorted(cgs, key=lambda t: -float(t or 0)):
            crossing = _crossover(rs, cgs[tol])
            if crossing:
                found.append((crossing, variant_label(key, show_fact), tol))
    if found:
        found.sort()
        crossing, name, tol = found[0]
        ax.axvline(crossing, color=MUTED, linestyle="--", linewidth=0.9,
                   zorder=1)
        ax.annotate(
            f"{name} overtakes cg tol {tol} at "
            f"≈ {crossing:.0f} steps",
            xy=(crossing, ax.get_ylim()[0]), xytext=(4, 14),
            textcoords="offset points", fontsize=8, color=MUTED, rotation=90)
        print("  crossovers (time steps beyond which Cholesky is cheaper) :")
        for c, name, t in found:
            print(f"    {name} vs cg tol {t} : {c:.1f} steps")
    else:
        print("  no crossover within the measured range of time steps")

    xs = [r["steps"] for r in here]
    ax.set_xlim(min(xs) * 0.8, max(xs) * 3.0)
    declutter(labels, ax)
    savefig("total_vs_steps.png")


def _crossover(chol, cg):
    """First step count where Cholesky becomes the cheaper of the two."""
    cg_by_steps = {r["steps"]: r["total_ms"] for r in cg}
    pts = [(r["steps"], r["total_ms"], cg_by_steps[r["steps"]])
           for r in chol if r["steps"] in cg_by_steps]
    pts.sort()
    for (s0, c0, g0), (s1, c1, g1) in zip(pts, pts[1:]):
        if (c0 - g0) > 0 >= (c1 - g1):
            d0, d1 = c0 - g0, c1 - g1
            if d0 == d1:
                return s1
            # Linear interpolation of the difference between the two samples.
            return s0 + (s1 - s0) * d0 / (d0 - d1)
    return None


# ---------------------------------------------------------------------------
# Figure 4 : fill in
# ---------------------------------------------------------------------------

def plot_fillin(runs):
    by_ordering = {}
    for r in runs:
        if r["solver"] != "cholesky" or not (r["nnz_l"] and r["nnz_a"]):
            continue
        # Keyed by ordering, not by variant : nnz(L) is decided by the
        # permutation alone, both factorizations computing the same factor
        # from the same pattern, so one curve per factorization would draw the
        # same points twice. Up-looking wins ties, for continuity with the
        # reports written before the multifrontal one existed.
        d = by_ordering.setdefault(r["ordering"], {})
        cur = d.get(r["elements"])
        if cur is None or (cur["factorization"] != "up-looking"
                           and r["factorization"] == "up-looking"):
            d[r["elements"]] = r
    by_ordering = {o: sorted(d.values(), key=lambda r: r["elements"])
                   for o, d in by_ordering.items()}
    by_ordering = {o: rs for o, rs in by_ordering.items() if len(rs) >= 2}
    if not by_ordering:
        print("figure 4 skipped : need nnz(A) and nnz(L) at two sizes or more")
        return
    # nnz(A) is a property of the operator, identical under any permutation,
    # so its reference curve is built from every size that ANY ordering
    # reached. Taking it from one ordering would silently truncate it to that
    # ordering's largest successful run.
    a_by_size = {}
    for rs in by_ordering.values():
        for r in rs:
            a_by_size[r["elements"]] = r
    rows = [a_by_size[e] for e in sorted(a_by_size)]
    xs = [r["elements"] for r in rows]

    fig, ax = new_axes(
        "Sparse Cholesky : fill in of the factor",
        "elements (triangles)   [= 12 n² at subdivision n]",
        "nonzeros",
        "memory, not time : nnz(L) is what the direct method has to store",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    labels = []
    all_x = []
    for ordering in sorted(by_ordering):
        rs = by_ordering[ordering]
        px = [r["elements"] for r in rs]
        py = [r["nnz_l"] for r in rs]
        all_x += px
        p = fit_exponent(px, py)
        label = f"cholesky ({ordering})"
        color = COLORS.get(label, COLORS["factorization"])
        ax.plot(px, py, marker="o", markersize=4, linewidth=1.8,
                color=color, zorder=3)
        labels.append({"x": px[-1], "y": py[-1], "color": color,
                       "text": f"nnz(L) {ordering}  (∝ E^{p:.2f})" if p
                               else f"nnz(L) {ordering}"})

    nnz_a = [r["nnz_a"] for r in rows]
    pa = fit_exponent(xs, nnz_a)
    ax.plot(xs, nnz_a, marker="o", markersize=4, linewidth=1.4,
            color=MUTED, zorder=3)
    labels.append({"x": xs[-1], "y": nnz_a[-1], "color": MUTED,
                   "text": f"nnz(A)  (∝ E^{pa:.2f})" if pa else "nnz(A)"})

    ax.set_xlim(min(all_x) * 0.8, max(all_x) * 5.0)
    declutter(labels, ax)
    savefig("cholesky_fillin.png")

    print("\n  fill in ratio nnz(L)/nnz(A) :")
    for ordering in sorted(by_ordering):
        print(f"    {ordering} :")
        for r in by_ordering[ordering]:
            print(f"      {r['dofs']:>7} DoF : x{r['nnz_l'] / r['nnz_a']:.1f}")


# ---------------------------------------------------------------------------
# Figure 5 : CG iterations
# ---------------------------------------------------------------------------

# The two SPD systems a time step solves. They are plotted together on
# purpose : the contrast between them is the result.
CG_SYSTEMS = [
    # key in the run dict, name, line style, marker
    ("cg_iter_psi", "stream function", "-", "s"),
    ("cg_iter_omega", "vorticity", "--", "o"),
]


def plot_cg_iterations(runs):
    """Iterations per solve, for both systems, against mesh size.

    The stream function solve is Spin * psi = M * omega, i.e. the pure
    stiffness operator : its condition number grows like h^-2, so the
    iteration count grows like sqrt(DoF).

    The vorticity solve is (M + nu*dt*S) * omega = rhs. With the defaults of
    profile_NS, nu*dt = 2e-4, so that operator is overwhelmingly the mass
    matrix, whose condition number is bounded independently of h. Its
    iteration count should therefore be close to flat -- which is why the two
    belong on the same axes rather than in two figures at different scales.
    """
    present = {}
    for key, _, _, _ in CG_SYSTEMS:
        cgs = {}
        for r in runs:
            if r["solver"] != "cholesky" and r[key]:
                cgs.setdefault(r["tol"], {})[r["elements"]] = r
        cgs = {t: sorted(d.values(), key=lambda r: r["elements"])
               for t, d in cgs.items()}
        cgs = {t: rs for t, rs in cgs.items() if len(rs) >= 2}
        if cgs:
            present[key] = cgs

    if not present:
        print("figure 5 skipped : need CG runs at two mesh sizes or more")
        return

    fig, ax = new_axes(
        "Conjugate gradient : iterations per solve against mesh size",
        "elements (triangles)",
        "iterations per solve",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    labels = []
    all_x = []
    exponents = {}
    for key, system, style, marker in CG_SYSTEMS:
        cgs = present.get(key)
        if not cgs:
            print(f"  '{system}' : no usable iteration count in the reports")
            continue
        for i, tol in enumerate(sorted(cgs, key=lambda t: -float(t or 0))):
            rs = cgs[tol]
            px = [r["elements"] for r in rs]
            py = [r[key] for r in rs]
            all_x += px
            p = fit_exponent(px, py)
            # The vorticity curve has a knee : mass dominated and flat while
            # nu*dt/h^2 is small, stiffness dominated and growing like
            # sqrt(DoF) once it is not. A single fitted exponent averages the
            # two regimes, so the tail is reported next to it.
            exponents[(system, tol)] = (p, fit_exponent(px[-3:], py[-3:]))
            color = cg_color(tol, i)
            ax.plot(px, py, marker=marker, markersize=4, linewidth=1.6,
                    linestyle=style, color=color, zorder=3)
            labels.append({
                "x": px[-1], "y": py[-1], "color": color,
                "text": f"{system}, tol {tol}  (∝ E^{p:.2f})" if p
                        else f"{system}, tol {tol}",
            })

    ax.set_xlim(min(all_x) * 0.8, max(all_x) * 6.0)
    declutter(labels, ax)
    savefig("cg_iterations.png")

    print("\n  cg iterations per solve, growth ∝ elements^p :")
    print(f"    {'system':<16} {'tol':<10} {'all sizes':>10} "
          f"{'3 largest':>10}")
    for (system, tol), (p, tail) in exponents.items():
        if p is None:
            continue
        tail_txt = f"{tail:+.2f}" if tail is not None else "-"
        print(f"    {system:<16} {tol:<10} {p:>+10.2f} {tail_txt:>10}")



# ---------------------------------------------------------------------------
# Figure 6 : up-looking vs multifrontal
# ---------------------------------------------------------------------------

def plot_factorization_scaling(runs):
    """How the two numerical factorizations scale, and by how much they differ.

    Only the factorization phase is plotted, and only it is comparable : the
    two algorithms produce the same L from the same pattern, so the ordering,
    the symbolic phase, nnz(L) and the substitutions are identical between
    them by construction. Everything visible here is therefore a pure timing
    difference, which is why the figure carries a ratio panel : an exponent
    alone would not say whether the gap widens or closes with the mesh.

    Top    : cost of one factorization against mesh size, log-log, with the
             fitted exponent of each curve.
    Bottom : up-looking / multifrontal, per ordering, on the sizes both
             reached. Above 1 means the multifrontal one is the faster.
    """
    cholesky = [r for r in runs if r["solver"] == "cholesky"]
    keys = sorted({variant(r) for r in cholesky})
    facts = {k[1] for k in keys}
    if len(facts) < 2:
        only = sorted(facts)[0] if facts else "none"
        print(f"figure 6 skipped : only the {only} factorization in the "
              f"reports")
        return

    series = {}
    for key in keys:
        rows = phase_rows(runs, key)
        pts = [(r["elements"], phase_cost(r, "factorization")) for r in rows]
        pts = [(x, y) for x, y in pts if y and y > 0]
        if len(pts) >= 2:
            series[key] = pts
    if len(series) < 2:
        print("figure 6 skipped : need two variants at two mesh sizes or more")
        return

    dofs = {r["elements"]: r["dofs"] for r in cholesky}
    all_x = sorted({x for pts in series.values() for x, _ in pts})

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(9, 7.6), facecolor=SURFACE, sharex=True,
        gridspec_kw={"height_ratios": [3, 1.35], "hspace": 0.12})
    style_axes(
        ax,
        "Sparse Cholesky : up-looking vs multifrontal factorization",
        None,
        "time of one factorization (ms)",
        f"same factor, same pattern, same fill in : only the algorithm "
        f"differs   |   {dofs[all_x[0]]} to {dofs[all_x[-1]]} DoF",
    )
    style_axes(rax, None,
               "elements (triangles)   [= 12 n\u00b2 at subdivision n]",
               "speedup\n(up-looking / multifrontal)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    rax.set_xscale("log")

    labels = []
    anchor = None
    exponents = {}
    for key in sorted(series):
        pts = series[key]
        px = [x for x, _ in pts]
        py = [y for _, y in pts]
        p = fit_exponent(px, py)
        exponents[key] = p
        color = variant_color(key)
        if anchor is None or key == ("nested", "up-looking"):
            anchor = (px[0], py[0])
        ax.plot(px, py, marker="o", markersize=4, linewidth=1.8, color=color,
                zorder=3)
        entry = {"x": px[-1], "y": py[-1], "color": color,
                 "text": f"{variant_label(key)}  (\u221d E^{p:.2f})" if p
                         else variant_label(key)}
        # A curve stopping short of the right edge -- the natural ordering
        # gives up early -- is labelled leftwards, off the stack at the edge.
        if px[-1] < all_x[-1]:
            entry["side"] = "left"
        labels.append(entry)

    guide_lines(ax, anchor, all_x[-1], ((1.0, "E"), (1.5, "E^1.5")), labels)
    ax.set_xlim(all_x[0] * 0.8, all_x[-1] * 5.0)
    declutter(labels, ax)

    # Ratio panel : one curve per ordering that has both factorizations.
    rax.axhline(1.0, color=MUTED, linestyle="--", linewidth=0.9, zorder=1)
    ratios = {}
    printed = []
    for ordering in sorted({k[0] for k in series}):
        up = dict(series.get((ordering, "up-looking"), []))
        mf = dict(series.get((ordering, "multifrontal"), []))
        common = sorted(set(up) & set(mf))
        if len(common) < 2:
            continue
        rx = common
        ry = [up[x] / mf[x] for x in common]
        ratios[ordering] = (rx, ry)
        color = variant_color((ordering, "multifrontal"))
        rax.plot(rx, ry, marker="o", markersize=4, linewidth=1.6, color=color,
                 zorder=3)
        rax.annotate(f"{ordering}", xy=(rx[-1], ry[-1]), xytext=(7, 0),
                     textcoords="offset points", fontsize=8.5, color=color,
                     va="center", zorder=4)
        printed.append((ordering, rx, ry))

    if not ratios:
        rax.text(0.5, 0.5, "no ordering run with both factorizations",
                 transform=rax.transAxes, ha="center", va="center",
                 fontsize=9, color=MUTED)
    else:
        lo = min(min(ry) for _, ry in ratios.values())
        hi = max(max(ry) for _, ry in ratios.values())
        rax.set_ylim(min(0.9, lo * 0.9), hi * 1.15)

    savefig("cholesky_factorization_scaling.png")

    print("\n  factorization cost \u221d elements^p :")
    for key in sorted(exponents):
        if exponents[key]:
            print(f"    {variant_label(key):<36} p = {exponents[key]:.2f}")
    for ordering, rx, ry in printed:
        print(f"\n  speedup of multifrontal over up-looking ({ordering}) :")
        for x, y in zip(rx, ry):
            print(f"    {dofs[x]:>7} DoF : x{y:.2f}")



# ---------------------------------------------------------------------------
# Figure 7 : parallel multifrontal, one curve per thread count
# ---------------------------------------------------------------------------

def plot_parallel_scaling(runs):
    """Factorization cost against mesh size per thread count (top) and the
    speedup over the sequential multifrontal (bottom). The speedup gets its
    own panel : a 2x gap is barely visible on a log axis spanning decades."""
    seq = {r["elements"]: phase_cost(r, "factorization") for r in runs
           if r["factorization"] == "multifrontal"}
    par = {}
    for r in runs:
        if r["factorization"] == "parallel multifrontal":
            par.setdefault(r["threads"], {})[r["elements"]] = \
                phase_cost(r, "factorization")
    par = {t: sorted((x, y) for x, y in d.items() if y) for t, d in par.items()}
    par = {t: pts for t, pts in par.items() if len(pts) >= 2}
    if not par:
        print("figure 7 skipped : need parallel runs at two mesh sizes or more")
        return

    dofs = {r["elements"]: r["dofs"] for r in runs}
    all_x = sorted({x for pts in par.values() for x, _ in pts})
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(9, 7.6), facecolor=SURFACE, sharex=True,
        gridspec_kw={"height_ratios": [3, 1.35], "hspace": 0.12})
    style_axes(ax, "Parallel multifrontal factorization : cost against mesh "
               "size", None, "time of one factorization (ms)",
               f"nested ordering, OpenMP over the elimination tree   |   "
               f"{dofs[all_x[0]]} to {dofs[all_x[-1]]} DoF")
    style_axes(rax, None, "elements (triangles)   [= 12 n\u00b2 at "
               "subdivision n]", "speedup over\nsequential multifrontal")
    for a in (ax, rax):
        a.set_xscale("log")
    ax.set_yscale("log")
    rax.axhline(1.0, color=MUTED, linestyle="--", linewidth=0.9, zorder=1)

    labels = []
    sx = sorted(x for x in seq if seq[x])
    if len(sx) >= 2:
        ax.plot(sx, [seq[x] for x in sx], linestyle="--", linewidth=1.4,
                color=MUTED, zorder=2)
        labels.append({"x": sx[-1], "y": seq[sx[-1]], "color": MUTED,
                       "text": "sequential multifrontal"})

    threads = sorted(par)
    speedups = {}
    rlabels = []
    for i, t in enumerate(threads):
        # Ramp steps spread over the whole range whatever the number of curves.
        color = THREAD_RAMP[round(i * (len(THREAD_RAMP) - 1)
                                  / max(1, len(threads) - 1))]
        px = [x for x, _ in par[t]]
        py = [y for _, y in par[t]]
        p = fit_exponent(px, py)
        name = f"{t} thread{'s' if t > 1 else ''}"
        ax.plot(px, py, marker="o", markersize=4, linewidth=1.8, color=color,
                zorder=3)
        labels.append({"x": px[-1], "y": py[-1], "color": color,
                       "text": f"{name}  (\u221d E^{p:.2f})" if p else name})
        common = [x for x in px if seq.get(x)]
        if common:
            ry = [seq[x] / dict(par[t])[x] for x in common]
            speedups[t] = list(zip(common, ry))
            rax.plot(common, ry, marker="o", markersize=4, linewidth=1.6,
                     color=color, zorder=3)
            rlabels.append({"x": common[-1], "y": ry[-1], "color": color,
                            "text": name})

    ax.set_xlim(all_x[0] * 0.8, all_x[-1] * 5.0)
    declutter(labels, ax)
    declutter(rlabels, rax, logy=False, gap=0.12)  # short panel
    savefig("parallel_scaling.png")

    print("\n  speedup over the sequential multifrontal :")
    print("    " + f"{'DoF':>8}" + "".join(f"{t:>8}t" for t in threads))
    for x in sorted({x for v in speedups.values() for x, _ in v}):
        row = [dict(speedups.get(t, [])).get(x) for t in threads]
        print("    " + f"{dofs[x]:>8}" + "".join(
            f"{v:>9.2f}" if v else f"{'-':>9}" for v in row))


# ---------------------------------------------------------------------------

def main():
    runs = load_runs()
    if not runs:
        print(f"no report found in {RUNS_DIR}")
        print("run the sweep first :")
        print("  python3 scripts/submit_runs.py          # or --local")
        sys.exit(1)

    chol = sum(1 for r in runs if r["solver"] == "cholesky")
    print(f"{len(runs)} report(s) : {chol} cholesky, {len(runs) - chol} cg")
    sizes = sorted({r["n"] for r in runs})
    steps = sorted({r["steps"] for r in runs})
    tols = sorted({r["tol"] for r in runs if r["tol"]})
    orderings = sorted({r["ordering"] for r in runs if r["ordering"]})
    facts = sorted({r["factorization"] for r in runs if r["factorization"]})
    variants = sorted({variant(r) for r in runs if r["solver"] == "cholesky"})
    print(f"  subdivisions : {sizes}")
    print(f"  time steps   : {steps}")
    print(f"  cg tolerances: {tols}")
    print(f"  orderings    : {orderings}")
    print(f"  factorizations: {facts}")
    print()

    show_fact = show_factorization(runs)
    for key in variants:
        plot_phases(runs, key, show_fact)
        print()
    plot_total_vs_size(runs)
    plot_total_vs_steps(runs)
    plot_fillin(runs)
    plot_cg_iterations(runs)
    plot_factorization_scaling(runs)
    print()
    plot_parallel_scaling(load_runs(PAR_RUNS_DIR))


if __name__ == "__main__":
    main()
