#include <cstdio>
#include "cholesky.h"
#include "P1.h"
#include "cube.h"
#include "mesh.h"

static void report(const char *label, CSRMatrix &S, CSRPattern &P) {
    SparseCholeskySymbolic symbolic(&S);
    symbolic.buildTree();
    CSRPattern patternL, patternL_T;
    symbolic.buildPatterns(&patternL, &patternL_T);

    long n = (long)S.rows;
    long nnzA = (long)S.nnz;
    long nnzL = (long)patternL.nnz;
    printf("%-32s  n=%-6ld nnz(A)=%-7ld nnz(L)=%-8ld ratio=%.3f\n",
           label, n, nnzA, nnzL, (double)nnzL / (double)nnzA);
    (void)P;
}

int main(int argc, char **argv) {
    size_t subdiv = (argc > 1) ? (size_t)atoi(argv[1]) : 8;

    /* 1. Natural ordering */
    Mesh mNatural;
    load_cube(mNatural, subdiv);
    CSRPattern Pnat; CSRMatrix Snat;
    build_P1_CSRPattern(mNatural, Pnat);
    build_P1_stiffness_matrix(mNatural, Pnat, Snat);
    report("Natural (load_cube)", Snat, Pnat);

    /* 2. Hand-built nested dissection (mesh-level, cube2.cpp) */
    Mesh mNested;
    load_cube_nested_dissect(mNested, subdiv);
    CSRPattern Pnes; CSRMatrix Snes;
    build_P1_CSRPattern(mNested, Pnes);
    build_P1_stiffness_matrix(mNested, Pnes, Snes);
    report("Hand-built ND (load_cube_nested_dissect)", Snes, Pnes);

    /* 3. Graph-based nested dissection, computed FROM the natural matrix --
     * this is the general SparseCholeskyOrdering path, no cube-specific code. */
    SparseCholeskyOrdering ordering(&Snat);
    ordering.order();
    CSRPattern Pgraph;
    ordering.applyToPattern(Pnat, &Pgraph);

    CSRMatrix Sgraph;
    Sgraph.symmetric = true;
    Sgraph.rows = Sgraph.cols = Pgraph.rows;
    Sgraph.nnz = Pgraph.nnz;
    Sgraph.row_start = Pgraph.row_start.data;
    Sgraph.col = Pgraph.col.data;
    Sgraph.data.resize(Sgraph.nnz); /* values unused for this test, only structure matters */

    report("Graph-based ND (SparseCholeskyOrdering)", Sgraph, Pgraph);

    return 0;
}