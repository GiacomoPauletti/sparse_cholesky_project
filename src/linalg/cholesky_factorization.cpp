#include "cholesky.h"

SparseCholeskyFactorization::SparseCholeskyFactorization(CSRMatrix* A) 
    : A{A}
{
    // A is the matrix to factorize
    // LPattern is the pattern of factor L (A=LL^T) obtained through the symbolic phase (SparseCholeskySymbolic)
}

void SparseCholeskyFactorization::setPatternL(CSRPattern* patternL) {
    this->patternL = patternL;
}

CSRMatrix* SparseCholeskyFactorization::factorize() {
    // Implement (sequential) factorization here
    return new CSRMatrix();
}