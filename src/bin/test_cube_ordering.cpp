#include <cstdio>
#include <string>
#include <filesystem>
#include "cholesky.h"
#include "P1.h"
#include "cube.h"
#include "mesh.h"

struct FillInResult {
    long n, nnzA, nnzL, fillIn, worstCase;
    double ratioA, ratioWorst;
};

static FillInResult compute_fill_in(Mesh &m) {
    CSRPattern P;
    CSRMatrix  S;
    build_P1_CSRPattern(m, P);
    build_P1_stiffness_matrix(m, P, S);

    SparseCholeskySymbolic symbolic(&S);
    symbolic.buildTree();

    CSRPattern patternL, patternL_T;
    symbolic.buildPatterns(&patternL, &patternL_T);

    FillInResult r;
    r.n          = (long)S.rows;
    r.nnzA       = (long)S.nnz;
    r.nnzL       = (long)patternL.nnz;
    r.fillIn     = r.nnzL - r.nnzA;
    r.worstCase  = (r.n * (r.n + 1)) / 2;
    r.ratioA     = (double)r.nnzL / (double)r.nnzA;
    r.ratioWorst = (double)r.nnzL / (double)r.worstCase;
    return r;
}

static void print_result(const char *label, const FillInResult &r) {
    printf("%s\n", label);
    printf("  rows              = %ld\n", r.n);
    printf("  nnz(A) lower+diag = %ld\n", r.nnzA);
    printf("  nnz(L) symbolic   = %ld\n", r.nnzL);
    printf("  fill-in           = %ld\n", r.fillIn);
    printf("  ratio vs A        = %.3f\n", r.ratioA);
    printf("  ratio vs worst    = %.5f\n", r.ratioWorst);
    printf("\n");
}

int main(int argc, char **argv) {
    /* Single-size mode (unchanged): ./test_cube_ordering <subdiv>
     * Sweep mode (new): ./test_cube_ordering --sweep out.csv s1 s2 s3 ...   */
    if (argc > 1 && std::string(argv[1]) == "--sweep") {
        if (argc < 4) {
            printf("usage: %s --sweep <out.csv> <subdiv1> [subdiv2 ...]\n", argv[0]);
            return 1;
        }
        const char *csvPath = argv[2];

        std::filesystem::path p(csvPath);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());

        FILE *f = fopen(csvPath, "w");
        if (!f) { printf("could not open %s for writing\n", csvPath); return 1; }
        fprintf(f, "ordering,subdiv,n,nnz_A,nnz_L_symbolic,fill_in,ratio_vs_A,ratio_vs_worst\n");

        for (int i = 3; i < argc; i++) {
            size_t subdiv = (size_t)atoi(argv[i]);

            Mesh mNatural, mNested;
            if (load_cube(mNatural, subdiv) != 0) { printf("load_cube failed (subdiv=%zu)\n", subdiv); continue; }
            if (load_cube_nested_dissect(mNested, subdiv) != 0) { printf("load_cube_nested_dissect failed (subdiv=%zu)\n", subdiv); continue; }

            FillInResult rNat = compute_fill_in(mNatural);
            FillInResult rNes = compute_fill_in(mNested);

            printf("subdiv=%zu  natural: n=%ld nnzL=%ld ratio=%.3f  |  nested: n=%ld nnzL=%ld ratio=%.3f\n",
                   subdiv, rNat.n, rNat.nnzL, rNat.ratioA, rNes.n, rNes.nnzL, rNes.ratioA);

            fprintf(f, "natural,%zu,%ld,%ld,%ld,%ld,%.6f,%.6f\n",
                    subdiv, rNat.n, rNat.nnzA, rNat.nnzL, rNat.fillIn, rNat.ratioA, rNat.ratioWorst);
            fprintf(f, "nested,%zu,%ld,%ld,%ld,%ld,%.6f,%.6f\n",
                    subdiv, rNes.n, rNes.nnzA, rNes.nnzL, rNes.fillIn, rNes.ratioA, rNes.ratioWorst);
        }

        fclose(f);
        printf("\nsaved -> %s\n", csvPath);
        return 0;
    }

    /* single-size mode */
    size_t subdiv = (argc > 1) ? (size_t)atoi(argv[1]) : 8;

    Mesh mNatural, mNested;
    if (load_cube(mNatural, subdiv) != 0) { printf("load_cube failed\n"); return 1; }
    if (load_cube_nested_dissect(mNested, subdiv) != 0) { printf("load_cube_nested_dissect failed\n"); return 1; }

    printf("==================== Fill-in comparison (subdiv=%zu) =======================\n\n", subdiv);
    print_result("Natural ordering (load_cube)", compute_fill_in(mNatural));
    print_result("Nested dissection ordering (load_cube_nested_dissect)", compute_fill_in(mNested));

    return 0;
}