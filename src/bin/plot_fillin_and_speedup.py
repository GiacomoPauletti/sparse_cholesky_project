import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import Patch

plt.style.use("dark_background")

CG_CSV   = sys.argv[1] if len(sys.argv) > 1 else "cg_vs_cholesky.csv"
FILL_CSV = sys.argv[2] if len(sys.argv) > 2 else "fill_in.csv"

COLORS = {
    "chol"  : "#5b9bd5",
    "cg_ok" : "#c4a85a",
    "cg_bad": "#d9534f",
    "chol_s": "#5b9bd5",
    "cg_s"  : "#5ca87a",
    "nnzA"  : "#5b9bd5",
    "nnzL"  : "#5ca87a",
    "worst" : "#d4884a",
    "ratio" : "#a07fd4",
    "vline" : "#888888",
    "annot" : "#cccccc",
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150, facecolor="#1a1a1a")
    print(f"saved -> {path}")
    plt.close()

def load(path):
    if not os.path.exists(path):
        print(f"file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    df["label"] = df["matrix"].str.split("/").str[-1]
    return df

def hbar_fig(nrows, title):
    fig, ax = plt.subplots(figsize=(10, max(5, nrows * 0.38)), facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    ax.set_title(title, fontsize=12, pad=8, color="#e0e0e0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(colors="#aaaaaa")
    ax.xaxis.label.set_color("#aaaaaa")
    ax.yaxis.label.set_color("#aaaaaa")
    return fig, ax

cg   = load(CG_CSV).sort_values("n").reset_index(drop=True)
fill = load(FILL_CSV).sort_values("n").reset_index(drop=True)

n  = len(cg)
nf = len(fill)

print("[1] plot_times.png")
y = np.arange(n)
h = 0.38
cg_colors = [COLORS["cg_ok"] if s == "OK" else COLORS["cg_bad"] for s in cg["status"]]
fig, ax = hbar_fig(n, "Total Cholesky time vs CG solve time  (log scale)")
ax.barh(y + h/2, cg["chol_total_ms"], height=h, color=COLORS["chol"])
ax.barh(y - h/2, cg["cg_solve_ms"],   height=h, color=cg_colors)
ax.set_yticks(y)
ax.set_yticklabels(cg["label"], fontsize=8, color="#cccccc")
ax.set_xscale("log")
ax.set_xlabel("time (log scale)")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda v, _: f"{v/1000:.0f}s" if v >= 1000 else f"{v:.3g}ms"))
ax.legend(handles=[
    Patch(color=COLORS["chol"],   label="Cholesky total (factorize + solve)"),
    Patch(color=COLORS["cg_ok"],  label="CG (converged)"),
    Patch(color=COLORS["cg_bad"], label="CG (did not converge)"),
], fontsize=9, loc="lower right", facecolor="#2a2a2a", edgecolor="#444444", labelcolor="#cccccc")
ax.invert_yaxis()
plt.tight_layout()
savefig("plot_times.png")

print("[2] plot_speedup.png")
ok = cg[cg["status"] == "OK"].copy()
ok["speedup"] = ok["chol_total_ms"] / ok["cg_solve_ms"]
ok = ok.sort_values("speedup")
sp_colors = [COLORS["cg_s"] if s > 1 else COLORS["chol_s"] for s in ok["speedup"]]
fig, ax = hbar_fig(len(ok), "Speedup  chol_total / cg_solve  (converged CG only)")
ax.barh(range(len(ok)), ok["speedup"], color=sp_colors)
ax.axvline(1.0, color=COLORS["vline"], linewidth=0.8, linestyle="--")
ax.set_yticks(range(len(ok)))
ax.set_yticklabels(ok["label"], fontsize=9, color="#cccccc")
ax.set_xlabel("speedup  (< 1 = Cholesky faster,  > 1 = CG faster)")
for i, (_, row) in enumerate(ok.iterrows()):
    ax.text(row["speedup"] + 0.02, i, f'{row["speedup"]:.2f}',
            va="center", fontsize=7.5, color=COLORS["annot"])
ax.legend(handles=[
    Patch(color=COLORS["chol_s"], label="Cholesky faster"),
    Patch(color=COLORS["cg_s"],   label="CG faster"),
], fontsize=9, facecolor="#2a2a2a", edgecolor="#444444", labelcolor="#cccccc")
ax.invert_yaxis()
plt.tight_layout()
savefig("plot_speedup.png")

print("[3] plot_iters.png")
iter_colors = []
for s in cg["status"]:
    if s == "OK":           iter_colors.append(COLORS["cg_s"])
    elif s == "CG_MAXITER": iter_colors.append(COLORS["cg_bad"])
    else:                   iter_colors.append(COLORS["worst"])
fig, ax = hbar_fig(n, "CG iterations per matrix  (tol=1e-6, cap=5000)")
ax.barh(range(n), cg["cg_iterations"], color=iter_colors)
ax.axvline(5000, color=COLORS["vline"], linewidth=0.8, linestyle="--")
ax.set_yticks(range(n))
ax.set_yticklabels(cg["label"], fontsize=8, color="#cccccc")
ax.set_xlabel("iterations")
ax.legend(handles=[
    Patch(color=COLORS["cg_s"],   label="OK (converged)"),
    Patch(color=COLORS["cg_bad"], label="CG_MAXITER"),
    Patch(color=COLORS["worst"],  label="CG_TIMEOUT"),
], fontsize=9, loc="lower right", facecolor="#2a2a2a", edgecolor="#444444", labelcolor="#cccccc")
ax.invert_yaxis()
plt.tight_layout()
savefig("plot_iters.png")

print("[4] plot_fillin.png")
y = np.arange(nf)
h = 0.25
fig, ax = hbar_fig(nf, "Fill-in: nnz(A)  vs  nnz(L symbolic)  vs  dense worst-case  (log scale)")
ax.barh(y + h, fill["nnz_A"],           height=h, color=COLORS["nnzA"])
ax.barh(y,     fill["nnz_L_symbolic"],  height=h, color=COLORS["nnzL"])
ax.barh(y - h, fill["nnz_L_worstcase"], height=h, color=COLORS["worst"], alpha=0.6)
ax.set_yticks(y)
ax.set_yticklabels(fill["label"], fontsize=8, color="#cccccc")
ax.set_xscale("log")
ax.set_xlabel("number of entries (log scale)")
ax.legend(handles=[
    Patch(color=COLORS["nnzA"],  label="nnz(A) lower triangle"),
    Patch(color=COLORS["nnzL"],  label="nnz(L) symbolic"),
    Patch(color=COLORS["worst"], label="dense worst-case  n(n+1)/2"),
], fontsize=9, loc="lower right", facecolor="#2a2a2a", edgecolor="#444444", labelcolor="#cccccc")
ax.invert_yaxis()
plt.tight_layout()
savefig("plot_fillin.png")

print("[5] plot_fillin_ratio.png")
fill["ratio"] = fill["nnz_L_symbolic"] / fill["nnz_A"]
fill_s = fill.sort_values("ratio")
fig, ax = hbar_fig(nf, "Fill-in ratio  nnz(L) / nnz(A)  — sorted ascending")
ax.barh(range(nf), fill_s["ratio"], color=COLORS["ratio"])
ax.axvline(1.0, color=COLORS["vline"], linewidth=0.8, linestyle="--")
ax.set_yticks(range(nf))
ax.set_yticklabels(fill_s["label"], fontsize=8, color="#cccccc")
ax.set_xlabel("nnz(L) / nnz(A)  (1 = no fill-in)")
for i, (_, row) in enumerate(fill_s.iterrows()):
    ax.text(row["ratio"] + 0.1, i, f'{row["ratio"]:.1f}x',
            va="center", fontsize=7.5, color=COLORS["annot"])
ax.invert_yaxis()
plt.tight_layout()
savefig("plot_fillin_ratio.png")

print("done.")