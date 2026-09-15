import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("dark_background")

# scripts/plot_cube_fillin.py -> ../plots is the shared output/input directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "plots"))
os.makedirs(PLOTS_DIR, exist_ok=True)

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PLOTS_DIR, "cube_fillin.csv")

COLORS = {
    "natural": "#d9534f",
    "nested":  "#5ca87a",
}

def savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150, facecolor="#1a1a1a")
    print(f"saved -> {path}")
    plt.close()

if not os.path.exists(CSV_PATH):
    print(f"file not found: {CSV_PATH}")
    print(f"generate it first, e.g.:")
    print(f"  ./build/release/test_cube_ordering --sweep {CSV_PATH} 2 4 6 8 10 12 14 16 20")
    sys.exit(1)

df = pd.read_csv(CSV_PATH).sort_values(["ordering", "n"])
nat = df[df["ordering"] == "natural"].sort_values("n")
nes = df[df["ordering"] == "nested"].sort_values("n")

def line_fig(title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    ax.set_title(title, fontsize=12, pad=8, color="#e0e0e0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(colors="#aaaaaa")
    ax.set_xlabel("n (mesh vertices)", color="#aaaaaa")
    ax.set_ylabel(ylabel, color="#aaaaaa")
    return fig, ax

print("[1] plot_cube_fillin_ratio.png")
fig, ax = line_fig(
    "Fill-in ratio  nnz(L) / nnz(A)  vs mesh size\n"
    "(P1 stiffness matrix on a cube mesh, natural vs graph-based nested-dissection ordering)",
    "nnz(L) / nnz(A)")
ax.plot(nat["n"], nat["ratio_vs_A"], marker="o", color=COLORS["natural"], label="Natural ordering (load_cube)")
ax.plot(nes["n"], nes["ratio_vs_A"], marker="o", color=COLORS["nested"],  label="Graph-based nested dissection (SparseCholeskyOrdering)")
ax.legend(fontsize=9, facecolor="#2a2a2a", edgecolor="#444444", labelcolor="#cccccc")
plt.tight_layout()
savefig("plot_cube_fillin_ratio.png")

print("[2] plot_cube_nnzL.png")
fig, ax = line_fig(
    "nnz(L) (symbolic)  vs mesh size  (log scale)",
    "nnz(L)")
ax.plot(nat["n"], nat["nnz_L_symbolic"], marker="o", color=COLORS["natural"], label="Natural ordering (load_cube)")
ax.plot(nes["n"], nes["nnz_L_symbolic"], marker="o", color=COLORS["nested"],  label="Graph-based nested dissection (SparseCholeskyOrdering)")
ax.set_yscale("log")
ax.legend(fontsize=9, facecolor="#2a2a2a", edgecolor="#444444", labelcolor="#cccccc")
plt.tight_layout()
savefig("plot_cube_nnzL.png")

print("[3] plot_cube_ratio_vs_worst.png")
fig, ax = line_fig(
    "Fill-in ratio vs dense worst-case  nnz(L) / (n(n+1)/2)",
    "nnz(L) / worst-case")
ax.plot(nat["n"], nat["ratio_vs_worst"], marker="o", color=COLORS["natural"], label="Natural ordering (load_cube)")
ax.plot(nes["n"], nes["ratio_vs_worst"], marker="o", color=COLORS["nested"],  label="Graph-based nested dissection (SparseCholeskyOrdering)")
ax.legend(fontsize=9, facecolor="#2a2a2a", edgecolor="#444444", labelcolor="#cccccc")
plt.tight_layout()
savefig("plot_cube_ratio_vs_worst.png")

print("done.")