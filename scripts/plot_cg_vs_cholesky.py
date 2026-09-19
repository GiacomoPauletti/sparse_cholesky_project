"""Plot the CG -> Cholesky convergence scan.

Generate the CSV first:
    ./build/release/bench_cg_vs_cholesky 16 plots/cg_vs_cholesky.csv
then:
    python3 scripts/plot_cg_vs_cholesky.py
"""

import sys
import os
import csv
import matplotlib.pyplot as plt


# scripts/plot_cg_vs_cholesky.py -> ../plots is the shared output/input directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "plots"))
os.makedirs(PLOTS_DIR, exist_ok=True)

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PLOTS_DIR, "cg_vs_cholesky.csv")

SURFACE = "#ffffff"
# Categorical slots 1 and 2, light steps. Validated on this surface : worst CVD
# deltaE 24.7 (protan 24.7 / deutan 31.7), normal-vision 33.6, both marks above
# 3:1 against white. A red/green pair would only reach deutan ~6, so it is
# avoided ; the curves are direct-labelled as well, so identity never rests on
# color alone.
COLORS = {
    "omega": "#2a78d6",
    "psi":   "#eb6834",
}
INK = "#0b0b0b"    # titles and axis labels
MUTED = "#52514e"  # ticks, spines, guide line -- 7.9:1 on white
GRID = "#e3e2de"   # recessive, must stay well below the marks


def savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150, facecolor=SURFACE)
    print(f"saved -> {path}")
    plt.close()


def new_axes(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=12, pad=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


if not os.path.exists(CSV_PATH):
    print(f"file not found: {CSV_PATH}")
    print("generate it first, e.g.:")
    print(f"  ./build/release/bench_cg_vs_cholesky 16 {CSV_PATH}")
    sys.exit(1)

with open(CSV_PATH, newline="") as fh:
    rows = sorted(csv.DictReader(fh), key=lambda r: float(r["tol"]))

INT_COLS = {"iter_psi", "iter_omega", "converged", "n_dof"}
df = {k: [(int if k in INT_COLS else float)(r[k]) for r in rows] for k in rows[0]}

n_dof = df["n_dof"][0]
nu = df["nu"][0]
dt = df["dt"][0]
chol_step = df["chol_step_ms"][0]
subtitle = f"sphere, {n_dof} DoF   nu = {nu:g}   dt = {dt:g}   one time step"

missed = [t for t, c in zip(df["tol"], df["converged"]) if not c]
if missed:
    print(f"warning: iter_max reached at tol = {missed} (tolerance not attained)")

# ---------------------------------------------------------------------------
# 1. Convergence : distance to the Cholesky solution vs the CG tolerance.
# ---------------------------------------------------------------------------
fig, ax = new_axes(
    "CG converges to the Cholesky solution\n" + subtitle,
    "CG relative residual tolerance",
    r"relative $\ell^2$ distance to Cholesky",
)
ax.set_xscale("log")
ax.set_yscale("log")

# Slope-1 guide : error tracking the tolerance one-for-one.
lo = min(min(df["tol"]), min(df["err_omega"]), min(df["err_psi"])) * 0.4
hi = max(max(df["tol"]), max(df["err_omega"]), max(df["err_psi"])) * 6
ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
ax.text(hi, hi, "  error = tol", color=MUTED, fontsize=8.5,
        va="center", ha="left", clip_on=False)

# Direct-labelled at the loose-tolerance end, where the two curves are well
# separated ; at the tight end they sit on top of each other.
for key, label, dy in (("omega", r"vorticity  $\omega$", -16),
                       ("psi", r"stream function  $\psi$", -6)):
    ax.plot(df["tol"], df[f"err_{key}"], color=COLORS[key], linewidth=2,
            marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.2, label=label, zorder=3)
    ax.annotate(label, (df["tol"][-1], df[f"err_{key}"][-1]),
                textcoords="offset points", xytext=(-10, dy),
                ha="right", color=COLORS[key], fontsize=9, zorder=4)

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
for txt in leg.get_texts():
    txt.set_color(INK)
savefig("cg_vs_cholesky_convergence.png")

# ---------------------------------------------------------------------------
# 2. Cost in iterations.
# ---------------------------------------------------------------------------
fig, ax = new_axes(
    "CG iterations needed per system\n" + subtitle,
    "CG relative residual tolerance",
    "iterations (single time step)",
)
ax.set_xscale("log")
ax.invert_xaxis()
for key, col, label in (
    ("omega", "iter_omega", r"vorticity   $K = M + \nu\,dt\,S$"),
    ("psi", "iter_psi", r"stream function   $S_{\rm pin}$"),
):
    ax.plot(df["tol"], df[col], color=COLORS[key], linewidth=2,
            marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.2, label=label)
leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
for txt in leg.get_texts():
    txt.set_color(INK)
ax.text(0.99, 0.03,
        "tolerance tightens ->",
        transform=ax.transAxes, ha="right", va="bottom",
        color=MUTED, fontsize=8.5)
savefig("cg_vs_cholesky_iterations.png")

# ---------------------------------------------------------------------------
# 3. Cost in wall time, against the direct solver.
# ---------------------------------------------------------------------------
fig, ax = new_axes(
    "CG is far cheaper than a factorization at this size\n" + subtitle,
    "CG relative residual tolerance",
    "wall time per step (ms)",
)
ax.set_xscale("log")
ax.invert_xaxis()
ax.plot(df["tol"], df["cg_step_ms"], color=COLORS["omega"], linewidth=2,
        marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2)
ax.axhline(chol_step, color=COLORS["psi"], linewidth=2, linestyle=(0, (4, 3)))
ax.text(ax.get_xlim()[1], chol_step, f"  Cholesky  ({chol_step:.0f} ms)",
        color=COLORS["psi"], fontsize=9, va="bottom", ha="left", clip_on=False)
ax.text(df["tol"][0], df["cg_step_ms"][0], "  CG  ",
        color=COLORS["omega"], fontsize=9, va="center", ha="left")
savefig("cg_vs_cholesky_time.png")

# ---------------------------------------------------------------------------
# Table view, so identity is never carried by color alone.
# ---------------------------------------------------------------------------
cols = ["tol", "iter_psi", "iter_omega", "err_omega", "err_psi", "cg_step_ms"]
print()
print("".join(f"{c:>13}" for c in cols))
for i in range(len(rows)):
    cells = []
    for c in cols:
        v = df[c][i]
        cells.append(f"{v:>13d}" if isinstance(v, int) else f"{v:>13.3e}")
    print("".join(cells))
print(f"\nCholesky reference: setup {df['chol_setup_ms'][0]:.1f} ms, "
      f"step {chol_step:.1f} ms")
