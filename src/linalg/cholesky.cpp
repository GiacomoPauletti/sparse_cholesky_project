#include "cholesky.h"

SparseCholeskySolver::SparseCholeskySolver(CSRMatrix* A) 
    : symbolic{A}, factorization{A}
{

}
CSRMatrix* SparseCholeskySolver::getFactor() {
    return this->factor;
}

void SparseCholeskySolver::initialize(CSRMatrix* A) {
    ordering.order();

    patternL   = new CSRPattern();
    patternL_T = new CSRPattern();
    symbolic.buildPatterns(patternL, patternL_T);

    factorization.setPatternL(patternL);
    factor = factorization.factorize();

    uint32_t n = factor->rows;

    factor_T           = new CSRMatrix();
    factor_T->symmetric = false;
    factor_T->rows      = n;
    factor_T->cols      = n;
    factor_T->nnz       = patternL_T->nnz;
    factor_T->row_start = patternL_T->row_start.data;
    factor_T->col       = patternL_T->col.data;
    factor_T->data.resize(patternL_T->nnz);

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t k = factor->row_start[i]; k < factor->row_start[i + 1]; k++) {
            uint32_t j = factor->col[k];
            int lo = (int)factor_T->row_start[j];
            int hi = (int)factor_T->row_start[j + 1] - 1;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                if      (factor_T->col[mid] == i) { factor_T->data[mid] = factor->data[k]; break; }
                else if (factor_T->col[mid] <  i) lo = mid + 1;
                else                               hi = mid - 1;
            }
        }
    }
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
    delete this->factor;
    delete this->factor_T;
    delete this->patternL;
    delete this->patternL_T;
}