#include <iostream>
#include "cholesky.h"

bool check_pattern(CSRPattern* got, CSRPattern* expected) {
    bool result = true; 
    result &= (got->symmetric == expected->symmetric);
    result &= (got->rows == expected->rows);
    result &= (got->cols == expected->cols);
    result &= (got->nnz == expected->nnz);

    result &= (got->row_start.size == expected->row_start.size);
    for ( int i=0; i<got->row_start.size; i++ ) {
        result &= (got->row_start[i] == expected->row_start[i]);
    }

    result &= (got->col.size == expected->col.size);
    for ( int i=0; i<got->col.size; i++ ) {
        result &= (got->col[i] == expected->col[i]);
    }

    return result;
}

// TODO: add tolerance
bool check_matrix(CSRMatrix* got, CSRMatrix* expected) {
    bool result = true; 
    result &= (got->symmetric == expected->symmetric);
    result &= (got->rows == expected->rows);
    result &= (got->cols == expected->cols);
    result &= (got->nnz == expected->nnz);

    for ( int i=0; i<got->rows+1; i++ ) {
        result &= (got->row_start[i] == expected->row_start[i]);
    }

    for ( int i=0; i<got->nnz; i++ ) {
        result &= (got->col[i] == expected->col[i]);
    }

    for ( int i=0; i<got->nnz; i++ ) {
        result &= (got->data[i] == expected->data[i]);
    }

    return result;
}

// TODO: add tolerance
bool check_solution(TArray<double> got, TArray<double> expected) {
    bool result = true; 
    result &= (got.size == expected.size);
    for ( int i=0; i<got.size; i++ ) {
        result &= (got[i] == expected[i]);
    }

    return result;
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

    SparseCholeskySolver solver(&A);
    solver.initialize(&A);
    // check_matrix(solver.getFactor(), patternExpectedL);
    solver.solve(x.data, b.data);
    // check_solution(x, expectedX);
}