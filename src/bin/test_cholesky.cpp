#include <stdio>
#include "cholesky.h"

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
    A.nnz = patternA.nnz;
    A.row_start = patternA.row_start.data();
    A.col = patternA.col.data();
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

    expectedPatternL.row_start.push_back(0);  patternL.row_start.push_back(1);    patternA.row_start.push_back(2);
    expectedPatternL.row_start.push_back(3);  patternL.row_start.push_back(6);    patternA.row_start.push_back(10);
    expectedPatternL.row_start.push_back(13); patternL.row_start.push_back(14);   patternA.row_start.push_back(21);

    expectedPatternL.col.push_back(0); patternL.col.push_back(1); patternL.col.push_back(2); // row 0,1,2
    expectedPatternL.col.push_back(1); patternL.col.push_back(2); patternL.col.push_back(3); // row 3
    expectedPatternL.col.push_back(0); patternL.col.push_back(1); patternL.col.push_back(3); patternA.col.push_back(4); //row 4
    expectedPatternL.col.push_back(0); patternL.col.push_back(4); patternL.col.push_back(5); // row 5
    expectedPatternL.col.push_back(6); // row 6
    expectedPatternL.col.push_back(1); patternL.col.push_back(2); patternL.col.push_back(3);
    expectedPatternL.col.push_back(4); patternL.col.push_back(5); patternL.col.push_back(6);
    expectedPatternL.col.push_back(7); // row 7

    SparseCholeskySolver solver(A);
}