#include "cholesky.h"

SparseCholeskySolver::SparseCholeskySolver(CSRMatrix* A,
                                           CholeskyOrderingKind ordering,
                                           Profiler* profiler,
                                           CholeskyFactorizationKind factorization)
    : LinearSolver(profiler), symbolic{A},
      orderingKind{ordering}, factorizationKind{factorization}
{

}
CSRMatrix* SparseCholeskySolver::getFactor() {
    return this->factor;
}

void SparseCholeskySolver::release() {
    delete factor;     factor     = nullptr;
    delete factor_T;   factor_T   = nullptr;
    delete patternL;   patternL   = nullptr;
    delete patternL_T; patternL_T = nullptr;
}

void SparseCholeskySolver::initialize(CSRMatrix* A) {
    ProfileSection section(profiler, "cholesky initialize");

    /* initialize() may called whenever A changes */
    release();

    /* What is actually factorized : A itself, or its permutation. */
    CSRMatrix* target = A;
    {
        ProfileStep step(profiler, "ordering");
        if (orderingKind == CHOLESKY_ORDERING_NESTED) {
            ordering.setMatrix(A);
            ordering.order();
            perm = ordering.permutation();
            /* P A P^T, values included : the symbolic and numeric phases
             * below then know nothing about the permutation, and only solve()
             * has to undo it. */
            ordering.applyToMatrix(*A, &permPattern, &permA);
            target = &permA;

            permRhs.resize(A->rows);
            permSol.resize(A->rows);
        }
    }

    /* Point the two phases at whichever matrix was selected above. */
    symbolic = SparseCholeskySymbolic(target);
    if (factorizationKind == CHOLESKY_FACTORIZATION_MULTIFRONTAL)
        factorization.reset(new MultifrontalSparseCholeskyFactorization(target));
    else
        factorization.reset(new UplookingSparseCholeskyFactorization(target));

    patternL   = new CSRPattern();
    patternL_T = new CSRPattern();
    {
        /* buildPatterns() would build the elimination tree itself on its first
         * call. Building it explicitly here separates the two halves of the
         * symbolic phase : the tree, and the row patterns of L and L^T
         * deduced from it. The second call is a no-op (isTreeBuilt). */
        ProfileStep step(profiler, "elimination tree");
        symbolic.buildTree();
    }
    {
        ProfileStep step(profiler, "row patterns");
        symbolic.buildPatterns(patternL, patternL_T, &cscToCsr);
    }

    {
        ProfileStep step(profiler, "factorization");
        factorization->setPatterns(patternL, patternL_T, &cscToCsr);
        factor = factorization->factorize();
    }

    /* Scattering L into L^T is not one of the four textbook phases, but it is
     * a full pass over nnz(L) with a binary search per entry, so it is worth
     * its own line rather than being hidden inside the factorization. */
    ProfileStep transpose(profiler, "transpose (build L^T)");

    uint32_t n = factor->rows;

    factor_T           = new CSRMatrix();
    factor_T->symmetric = false;
    factor_T->rows      = n;
    factor_T->cols      = n;
    factor_T->nnz       = patternL_T->nnz;
    factor_T->row_start = patternL_T->row_start.data;
    factor_T->col       = patternL_T->col.data;
    factor_T->data.resize(patternL_T->nnz);

    /* cscToCsr already says, for every entry of L^T, where the same entry
     * sits in L, so this is a plain gather : no binary search per entry. */
    for (size_t q = 0; q < patternL_T->nnz; q++)
        factor_T->data[q] = factor->data[cscToCsr[q]];
}

void SparseCholeskySolver::substitute(double *__restrict x, const double *__restrict b) {
    // 1. Forward substitution of Ly=b
    ProfileStep forward(profiler, "forward substitution (L y = b)");
    TArray<double> y(factor->rows);
    for (uint32_t i=0; i < factor->rows; i++) {
        double rhs = b[i];
        for (uint32_t j_index=factor->row_start[i]; j_index<factor->row_start[i+1] - 1; j_index++) {
            uint32_t j = factor->col[j_index];
            rhs -= factor->data[j_index] * y[j];    // alternatively L[i,j] * y[j]
        }

        uint32_t diagonal_index = factor->row_start[i+1] - 1;
        y[i] = rhs / (factor->data[diagonal_index]);        // alternatively rhs / L[i,i]
    }
    forward.stop();

    // 2. Backward substitution of L^T x = y
    ProfileStep backward(profiler, "backward substitution (L^T x = y)");
    for (int i=factor_T->rows-1; i >= 0; i--) {
        double rhs = y[i];
        for (uint32_t j_index=factor_T->row_start[i]+1; j_index<factor_T->row_start[i+1]; j_index++) {
            uint32_t j = factor_T->col[j_index];
            rhs -= factor_T->data[j_index] * x[j];    // alternatively L_T[i,j] * x[j]
        }

        uint32_t diagonal_index = factor_T->row_start[i];
        x[i] = rhs / (factor_T->data[diagonal_index]);        // alternatively rhs / L_T[i,i]
    }
}

void SparseCholeskySolver::solve(double *__restrict x, const double *__restrict b) {
    ProfileSection section(profiler, "cholesky solve");

    if (orderingKind == CHOLESKY_ORDERING_NATURAL) {
        substitute(x, b);
        return;
    }

    /* The factor is that of P A P^T, so the right hand side has to enter the
     * permuted numbering and the solution has to come back out of it :
     *
     *      (P A P^T) (P x) = P b
     *
     * with perm[i] the new index of the old index i. */
    uint32_t n = (uint32_t)factor->rows;
    {
        ProfileStep step(profiler, "permute rhs");
        for (uint32_t i = 0; i < n; i++) permRhs[perm[i]] = b[i];
    }

    substitute(permSol.data, permRhs.data);

    ProfileStep step(profiler, "unpermute solution");
    for (uint32_t i = 0; i < n; i++) x[i] = permSol[perm[i]];
}

SparseCholeskySolver::~SparseCholeskySolver() {
    release();
}
