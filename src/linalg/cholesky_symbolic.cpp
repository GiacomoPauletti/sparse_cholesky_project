#include "cholesky.h"

SparseCholeskySymbolic::SparseCholeskySymbolic(CSRMatrix* A) 
    : A{A}
{
}
CholeskyTree SparseCholeskySymbolic::buildTree() {
    // TODO
    return CholeskyTree();
}
CSRPattern* SparseCholeskySymbolic::buildPatternL() {
    // TODO
    return new CSRPattern();
}
// eventually buildPatternL and buildPatternL_T will be unified in buildPattern
CSRPattern* SparseCholeskySymbolic::buildPatternL_T() {
    // TODO
    return new CSRPattern();
}