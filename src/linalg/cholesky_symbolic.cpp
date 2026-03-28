#include "cholesky.h"
#include <iostream>

SparseCholeskySymbolic::SparseCholeskySymbolic(CSRMatrix* A) 
{
    this->A = A;
}
const CholeskyTree& SparseCholeskySymbolic::buildTree() {
    this->isTreeBuilt = true;

    this->tree = CholeskyTree(A->rows);

    std::vector<int> ancestor(A->rows);
    // Build cholesky tree

    // end of tree construction

    return this->tree;
}
CSRPattern* SparseCholeskySymbolic::buildPatternL() {
    if ( !this->isTreeBuilt ) {
        this->buildTree();
    }

    return new CSRPattern();
}
// eventually buildPatternL and buildPatternL_T will be unified in buildPattern
CSRPattern* SparseCholeskySymbolic::buildPatternL_T() {
    if ( !this->isTreeBuilt ) {
        this->buildTree();
    }

    return new CSRPattern();
}