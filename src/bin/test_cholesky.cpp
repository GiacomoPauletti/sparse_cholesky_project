#include <iostream>
#include "cholesky.h"

bool check_pattern(CSRPattern* got, CSRPattern* expected) {
    if ( got == nullptr ) {
        std::cout << "Pattern is nullptr" << std::endl;
        return false;
    }
    if (got->symmetric != expected->symmetric) {
        std::cout << "Pattern symmetry. Expected: " << expected->symmetric << "; got: " << got->symmetric << std::endl,
        return false;
    }
    if (got->rows != expected->rows) {
        std::cout << "Pattern rows. Expected: " << expected->rows << "; got: " << got->rows << std::endl,
        return false;
    }
    if (got->cols != expected->cols) {
        std::cout << "Pattern cols. Expected: " << expected->cols << "; got: " << got->cols << std::endl,
        return false;
    }
    if (got->nnz != expected->nnz) {
        std::cout << "Pattern nnz. Expected: " << expected->nnz << "; got: " << got->nnz << std::endl,
        return false;
    }

    if (got->row_start.size != expected->row_start.size) {
        std::cout << "Pattern row_start.size. Expected: " << expected->row_start.size << "; got: " << got->row_start.size << std::endl,
        return false;
    }
    for ( int i=0; i<got->row_start.size; i++ ) {
        if (got->row_start[i] != expected->row_start[i]) {
            std::cout << "Pattern row_start["<< i <<"]. Expected: " << expected->row_start[i] << "; got: " << got->row_start[i] << std::endl,
            return false;
        }
    }

    if (got->col.size != expected->col.size) {
        std::cout << "Pattern cols.size. Expected: " << expected->cols.size << "; got: " << got->cols.size << std::endl,
        return false;
    }
    for ( int i=0; i<got->col.size; i++ ) {
        if (got->col[i] != expected->col[i]) {
            std::cout << "Pattern col["<< i <<"]. Expected: " << expected->col[i] << "; got: " << got->col[i] << std::endl,
            return false;
        }
    }

    return true;
}

// TODO: add tolerance
bool check_matrix(CSRMatrix* got, CSRMatrix* expected) {
    if ( got == nullptr ) {
        std::cout << "Matrix is nullptr" << std::endl;
        return false;
    }
    if (got->symmetric != expected->symmetric) {
        std::cout << "Matrix symmetry. Expected: " << expected->symmetric << "; got: " << got->symmetric << std::endl,
        return false;
    }
    if (got->rows != expected->rows) {
        std::cout << "Matrix rows. Expected: " << expected->rows << "; got: " << got->rows << std::endl,
        return false;
    }
    if (got->cols != expected->cols) {
        std::cout << "Matrix cols. Expected: " << expected->cols << "; got: " << got->cols << std::endl,
        return false;
    }
    if (got->nnz != expected->nnz) {
        std::cout << "Matrix nnz. Expected: " << expected->nnz << "; got: " << got->nnz << std::endl,
        return false;
    }

    for ( int i=0; i<got->rows+1; i++ ) {
        if (got->row_start[i] != expected->row_start[i]) {
            std::cout << "Matrix row_start["<< i <<"]. Expected: " << expected->row_start[i] << "; got: " << got->row_start[i] << std::endl,
            return false;
        }
    }

    for ( int i=0; i<got->nnz; i++ ) {
        if (got->col[i] != expected->col[i]) {
            std::cout << "Matrix col["<< i <<"]. Expected: " << expected->col[i] << "; got: " << got->col[i] << std::endl,
            return false;
        }
    }

    for ( int i=0; i<got->nnz; i++ ) {
        if (got->data[i] != expected->data[i]) {
            std::cout << "Matrix data["<< i <<"]. Expected: " << expected->data[i] << "; got: " << got->data[i] << std::endl,
            return false;
        }
    }

    return true;
}

// TODO: add tolerance
bool check_solution(TArray<double> got, TArray<double> expected) {
    if (got.size != expected.size) {
        std::cout << "Solution size. Expected: " << expected->size << "; got: " << got->size << std::endl,
        return false;
    }
    for ( int i=0; i<got.size; i++ ) {
        std::cout << "Solution["<< i <<"]. Expected: " << expected[i] << "; got: " << got[i] << std::endl,
        return false;
    }

    return false;
}

int main() {
    /* Example matrix taken from Scott, Tuma Algorithms for Sparse Linear Systems 
     *  page 56 figure 4.2
     *  row/col 0  1  2  3  4  5  6  7 
     *  0     | *           *  *        |
     *  1     |    *     *  *           |
     *  2     |       *  *           *  | 
     *  3     |    *  *  *              |
     *  4     | *  *        *           |
     *  5     | *              *        |
     *  6     |                    *  * | 
     *  7     |    *  *            *  * |
     *
     * Entries are all set to one
     */
    
    CSRPattern patternA;
    patternA.symmetric = true;
    patternA.rows = 8;
    patternA.cols = 8;
    patternA.nnz = 8 + 8;  // diagonal elements + nnz under diagonal (not storing above diagonal)

    // patternA.row_start.resize(patternA.rows+1); --> resizing by pushing back
    patternA.row_start.push_back(0);  patternA.row_start.push_back(1);    patternA.row_start.push_back(2);
    patternA.row_start.push_back(3);  patternA.row_start.push_back(6);    patternA.row_start.push_back(9);
    patternA.row_start.push_back(11); patternA.row_start.push_back(12);   patternA.row_start.push_back(16);

    //patternA.col.resize(patternA.nnz); --> resizing by pushing back
    patternA.col.push_back(0); patternA.col.push_back(1); patternA.col.push_back(2);
    patternA.col.push_back(1); patternA.col.push_back(2); patternA.col.push_back(3);
    patternA.col.push_back(0); patternA.col.push_back(1); patternA.col.push_back(4);
    patternA.col.push_back(0); patternA.col.push_back(5); patternA.col.push_back(6);
    patternA.col.push_back(1); patternA.col.push_back(2); patternA.col.push_back(6);
    patternA.col.push_back(7);

    CSRMatrix A;
    A.symmetric = patternA.symmetric;
    A.rows = patternA.rows;
    A.cols = patternA.cols;
    A.nnz = patternA.nnz;
    A.row_start = patternA.row_start.data;
    A.col = patternA.col.data;
    A.data.push_back(1); A.data.push_back(1); A.data.push_back(1);
    A.data.push_back(1); A.data.push_back(1); A.data.push_back(1);
    A.data.push_back(1); A.data.push_back(1); A.data.push_back(1);
    A.data.push_back(1); A.data.push_back(1); A.data.push_back(1);
    A.data.push_back(1); A.data.push_back(1); A.data.push_back(1);
    A.data.push_back(1);

    CSRPattern expectedPatternL;
    expectedPatternL.symmetric = false;
    expectedPatternL.rows = 8;
    expectedPatternL.cols = 8;
    expectedPatternL.nnz = (8 + 8) + 5;  // elements of A + fill in

    expectedPatternL.row_start.push_back(0);  expectedPatternL.row_start.push_back(1);    expectedPatternL.row_start.push_back(2);
    expectedPatternL.row_start.push_back(3);  expectedPatternL.row_start.push_back(6);    expectedPatternL.row_start.push_back(10);
    expectedPatternL.row_start.push_back(13); expectedPatternL.row_start.push_back(14);   expectedPatternL.row_start.push_back(21);

    expectedPatternL.col.push_back(0); expectedPatternL.col.push_back(1); expectedPatternL.col.push_back(2); // row 0,1,2
    expectedPatternL.col.push_back(1); expectedPatternL.col.push_back(2); expectedPatternL.col.push_back(3); // row 3
    expectedPatternL.col.push_back(0); expectedPatternL.col.push_back(1); expectedPatternL.col.push_back(3); patternA.col.push_back(4); //row 4
    expectedPatternL.col.push_back(0); expectedPatternL.col.push_back(4); expectedPatternL.col.push_back(5); // row 5
    expectedPatternL.col.push_back(6); // row 6
    expectedPatternL.col.push_back(1); expectedPatternL.col.push_back(2); expectedPatternL.col.push_back(3);
    expectedPatternL.col.push_back(4); expectedPatternL.col.push_back(5); expectedPatternL.col.push_back(6);
    expectedPatternL.col.push_back(7); // row 7

    CSRMatrix expectedL;
    expectedL.symmetric = expectedPatternL.symmetric;
    expectedL.rows = expectedPatternL.rows;
    expectedL.cols = expectedPatternL.cols;
    expectedL.nnz = expectedPatternL.nnz;
    expectedL.row_start = expectedPatternL.row_start.data;
    expectedL.col = expectedPatternL.col.data;
    //expectedL.data.push_back(...);
    // still need to do the factorization by hand
    // TODO: hardcode in python the matrix and check if it is spd and compute its cholesky factor
    // TODO: find expected solution "expectedX"

    TArray<double> x(8), expectedX(8);
    TArray<double> b(0);  // 0 of size because resized through pushbacks
    b.push_back(1); b.push_back(1); b.push_back(1); b.push_back(1);
    b.push_back(1); b.push_back(1); b.push_back(1); b.push_back(1);

    SparseCholeskyOrdering ordering;
    SparseCholeskySymbolic symbolic(&A);
    SparseCholeskyFactorization factorization(&A);

    // solver.initialize(&A) is equivalent to the following code
    // 1. Ordering phase
    ordering.order();

    // 2. Symbolic phase
    CSRPattern* patternL = symbolic.buildPatternL();
    if (check_pattern(patternL, &expectedPatternL) == false) {
        return false;
    }
    CSRPattern* patternL_T = symbolic.buildPatternL_T();

    // 3. Factorization phase
    factorization.setPatternL(patternL);
    CSRMatrix* factor = factorization.factorize();
    if (check_matrix(factor, &expectedL) == false) {
        return false;
    }

    // solver.solve(x,b) is equivalent to the following code
    // ...
    if (check_solution(x, expectedSol) == false) {
        return false;
    }

    std::cout << "All tests passed" << std::endl;
}