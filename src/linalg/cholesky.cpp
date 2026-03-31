#include "cholesky.h"

SparseCholeskySolver::SparseCholeskySolver(CSRMatrix* A) 
    : symbolic{A}, factorization{A}
{

}
CSRMatrix* SparseCholeskySolver::getFactor() {
    return this->factor;
}

void SparseCholeskySolver::initialize(CSRMatrix* A) {
    // 1. Ordering phase
    ordering.order();

    // 2. Symbolic phase
    CSRPattern* patternL = new CSRPattern();
    CSRPattern* patternL_T = new CSRPattern();
    this->symbolic.buildPatterns(patternL, patternL_T);

    // 3. Factorization phase
    this->factorization.setPatternL(patternL);
    this->factor = factorization.factorize();

    // TODO
    // this->factor_T
}

void SparseCholeskySolver::solve(double *__restrict x, const double *__restrict b) {
    // 1. Forward substitution of Ly=b
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
    // 2. Backward substitution of L^T x = y
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

SparseCholeskySolver::~SparseCholeskySolver() {
    free(this->factor);
    free(this->factor_T);
}