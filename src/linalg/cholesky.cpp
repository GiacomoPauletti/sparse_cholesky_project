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
    CSRPattern* patternL = symbolic.buildPatternL();
    CSRPattern* patternL_T = symbolic.buildPatternL_T();

    // 3. Factorization phase
    this->factorization.setPatternL(patternL);
    this->factor = factorization.factorize();
}

void SparseCholeskySolver::solve(double *__restrict x, double *__restrict b) {
    // Solve LL^T*x=b
    // ...
}

SparseCholeskySolver::~SparseCholeskySolver() {
    free(this->factor);
}