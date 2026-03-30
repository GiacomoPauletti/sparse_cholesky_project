#include <iostream>
#include <iomanip>
#include <cmath>
#include "cholesky.h"

bool check_pattern(CSRPattern* got, CSRPattern* expected) {
    if ( got == nullptr ) {
        std::cout << "Pattern is nullptr" << std::endl;
        return false;
    }
    if (got->symmetric != expected->symmetric) {
        std::cout << "Pattern symmetry. Expected: " << expected->symmetric << "; got: " << got->symmetric << std::endl;
        return false;
    }
    if (got->rows != expected->rows) {
        std::cout << "Pattern rows. Expected: " << expected->rows << "; got: " << got->rows << std::endl;
        return false;
    }
    if (got->cols != expected->cols) {
        std::cout << "Pattern cols. Expected: " << expected->cols << "; got: " << got->cols << std::endl;
        return false;
    }
    if (got->nnz != expected->nnz) {
        std::cout << "Pattern nnz. Expected: " << expected->nnz << "; got: " << got->nnz << std::endl;
        return false;
    }

    if (got->row_start.size != expected->row_start.size) {
        std::cout << "Pattern row_start.size. Expected: " << expected->row_start.size << "; got: " << got->row_start.size << std::endl;
        return false;
    }
    for ( int i=0; i<got->row_start.size; i++ ) {
        if (got->row_start[i] != expected->row_start[i]) {
            std::cout << "Pattern row_start["<< i <<"]. Expected: " << expected->row_start[i] << "; got: " << got->row_start[i] << std::endl;
            return false;
        }
    }

    if (got->col.size != expected->col.size) {
        std::cout << "Pattern col.size. Expected: " << expected->col.size << "; got: " << got->col.size << std::endl;
        return false;
    }
    for ( int i=0; i<got->col.size; i++ ) {
        if (got->col[i] != expected->col[i]) {
            std::cout << "Pattern col["<< i <<"]. Expected: " << expected->col[i] << "; got: " << got->col[i] << std::endl;
            return false;
        }
    }

    return true;
}

bool check_matrix(CSRMatrix* got, CSRMatrix* expected, double tol=1e-5) {
    if ( got == nullptr ) {
        std::cout << "Matrix is nullptr" << std::endl;
        return false;
    }
    if (got->symmetric != expected->symmetric) {
        std::cout << "Matrix symmetry. Expected: " << expected->symmetric << "; got: " << got->symmetric << std::endl;
        return false;
    }
    if (got->rows != expected->rows) {
        std::cout << "Matrix rows. Expected: " << expected->rows << "; got: " << got->rows << std::endl;
        return false;
    }
    if (got->cols != expected->cols) {
        std::cout << "Matrix cols. Expected: " << expected->cols << "; got: " << got->cols << std::endl;
        return false;
    }
    if (got->nnz != expected->nnz) {
        std::cout << "Matrix nnz. Expected: " << expected->nnz << "; got: " << got->nnz << std::endl;
        return false;
    }

    for ( int i=0; i<got->rows+1; i++ ) {
        if (got->row_start[i] != expected->row_start[i]) {
            std::cout << "Matrix row_start["<< i <<"]. Expected: " << expected->row_start[i] << "; got: " << got->row_start[i] << std::endl;
            return false;
        }
    }

    for ( int i=0; i<got->nnz; i++ ) {
        if (got->col[i] != expected->col[i]) {
            std::cout << "Matrix col["<< i <<"]. Expected: " << expected->col[i] << "; got: " << got->col[i] << std::endl;
            return false;
        }
    }

    for ( int i=0; i<got->nnz; i++ ) {
        if (std::abs(got->data[i] != expected->data[i]) > tol) {
            std::cout << "Matrix data["<< i <<"]. Expected: " << expected->data[i] << "; got: " << got->data[i] << std::endl;
            return false;
        }
    }

    return true;
}
 
template<int d> 
std::ostream& fixed(std::ostream& os){
    os.setf(std::ios_base::fixed, std::ios_base::floatfield); 
    os.precision(d); 
    return os; 
}

bool check_solution(TArray<double>* got, TArray<double>* expected, double tol=1e-5) {
    if (got->size != expected->size) {
        std::cout << "Solution size. Expected: " << expected->size << "; got: " << got->size << std::endl;
        return false;
    }
    for ( int i=0; i<got->size; i++ ) {
        if (std::abs((*got)[i] - (*expected)[i]) > tol) {
            std::cout << fixed<10> << "Solution["<< i <<"]. Expected: " << (*expected)[i] << "; got: " << (*got)[i] << std::endl;
            return false;
        }
    }

    return true;
}

int main() {
    /* Example matrix taken from Scott, Tuma Algorithms for Sparse Linear Systems 
     *  page 56 figure 4.2
     *  row/col 0  1  2  3  4  5  6  7 
     *  0     | *           *  *       |
     *  1     |    *     *  *        * |
     *  2     |       *  *           * | 
     *  3     |    *  *  *             |
     *  4     | *  *        *          |
     *  5     | *              *       |
     *  6     |                   *  * | 
     *  7     |    *  *           *  * |
     *
     * Entries are all set to one.
     *
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
    // STORING ONLY LOW TRIANGULAR
    patternA.col.push_back(0); // row 0
    patternA.col.push_back(1); // row 1
    patternA.col.push_back(2); // row 2
    patternA.col.push_back(1); patternA.col.push_back(2); patternA.col.push_back(3); // row 3
    patternA.col.push_back(0); patternA.col.push_back(1); patternA.col.push_back(4); // row 4
    patternA.col.push_back(0); patternA.col.push_back(5); // row 5
    patternA.col.push_back(6); // row 6
    patternA.col.push_back(1); patternA.col.push_back(2); patternA.col.push_back(6); patternA.col.push_back(7); // row 7

    CSRMatrix A;
    A.symmetric = patternA.symmetric;
    A.rows = patternA.rows;
    A.cols = patternA.cols;
    A.nnz = patternA.nnz;
    A.row_start = patternA.row_start.data;
    A.col = patternA.col.data;
    A.data.push_back(10); // row 0
    A.data.push_back(10); // row 1
    A.data.push_back(10); // row 2
    A.data.push_back(1); A.data.push_back(1); A.data.push_back(10); // row 3
    A.data.push_back(1); A.data.push_back(1); A.data.push_back(10); // row 4
    A.data.push_back(1); A.data.push_back(10); 
    A.data.push_back(10); // row 6
    A.data.push_back(1); A.data.push_back(1); A.data.push_back(1); A.data.push_back(10);

    /*
     * Expected pattern for L is the following (f=new entry added, which is the fill-in)
     *  row/col 0  1  2  3  4  5  6  7 
     *  0     | *                      |
     *  1     |    *                   |
     *  2     |       *                | 
     *  3     |    *  *  *             |
     *  4     | *  *     f  *          |
     *  5     | *           f  *       |
     *  6     |                   *    | 
     *  7     |    *  *  f  f  f  *  * |
    */
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
    expectedPatternL.col.push_back(0); expectedPatternL.col.push_back(1); expectedPatternL.col.push_back(3); expectedPatternL.col.push_back(4); //row 4
    expectedPatternL.col.push_back(0); expectedPatternL.col.push_back(4); expectedPatternL.col.push_back(5); // row 5
    expectedPatternL.col.push_back(6); // row 6
    expectedPatternL.col.push_back(1); expectedPatternL.col.push_back(2); expectedPatternL.col.push_back(3);
    expectedPatternL.col.push_back(4); expectedPatternL.col.push_back(5); expectedPatternL.col.push_back(6);
    expectedPatternL.col.push_back(7); // row 7

    /* 
     * Using cholesky solver on python we deduce the following L:
       row/col 0               1               2               3              4               5               6               7
       0     | 3.16227766e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 |
       1     | 0.00000000e+00  3.16227766e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 |
       2     | 0.00000000e+00  0.00000000e+00  3.16227766e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 |
       3     | 0.00000000e+00  3.16227766e-01  3.16227766e-01  3.13049517e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 |
       4     | 3.16227766e-01  3.16227766e-01  0.00000000e+00 -3.19438282e-02  3.13033219e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 |
       5     | 3.16227766e-01  0.00000000e+00  0.00000000e+00  0.00000000e+00 -3.19454914e-02  3.14626437e+00  0.00000000e+00  0.00000000e+00 |
       6     | 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  3.16227766e+00  0.00000000e+00 |
       7     | 0.00000000e+00  3.16227766e-01  3.16227766e-01 -6.38876565e-02 -3.25974402e-02 -3.30977033e-04  3.16227766e-01  3.11365632e+00 |

    */

    CSRMatrix expectedL;
    expectedL.symmetric = expectedPatternL.symmetric;
    expectedL.rows = expectedPatternL.rows;
    expectedL.cols = expectedPatternL.cols;
    expectedL.nnz = expectedPatternL.nnz;
    expectedL.row_start = expectedPatternL.row_start.data;
    expectedL.col = expectedPatternL.col.data;
    expectedL.data.push_back(3.16227766); // row 0
    expectedL.data.push_back(3.16227766); // row 1
    expectedL.data.push_back(3.16227766); // row 2
    expectedL.data.push_back(3.16227766e-01); expectedL.data.push_back(3.16227766e-01); expectedL.data.push_back(3.13049517e+00); // row 3
    expectedL.data.push_back(3.16227766e-01); expectedL.data.push_back(3.16227766e-01); expectedL.data.push_back(-3.19438282e-02);
     expectedL.data.push_back(3.13033219e+00); // row 4
    expectedL.data.push_back(3.16227766e-01); expectedL.data.push_back(-3.19454914e-02); expectedL.data.push_back(3.14626437e+00); // row 5
    expectedL.data.push_back(3.16227766e+00); // row 6
             
    expectedL.data.push_back(3.16227766e-01); expectedL.data.push_back(3.16227766e-01); expectedL.data.push_back(-6.38876565e-02); // row 5
    expectedL.data.push_back(-3.25974402e-02); expectedL.data.push_back(-3.30977033e-04); expectedL.data.push_back(3.16227766e-01); // row 5
    expectedL.data.push_back(3.11365632e+00);

    /* 
     * Consequently the pattern for L_T is the following
     *  row/col 0  1  2  3  4  5  6  7 
     *  0     | *           *  *       |
     *  1     |    *     *  *        * |
     *  2     |       *  *           * | 
     *  3     |          *  f        f |
     *  4     |             *  f     f |
     *  5     |                *     f |
     *  6     |                   *  * | 
     *  7     |                      * |
    */
    CSRPattern expectedPatternL_T;
    expectedPatternL_T.symmetric = false;
    expectedPatternL_T.rows = 8;
    expectedPatternL_T.cols = 8;
    expectedPatternL_T.nnz = (8 + 8) + 5;  // elements of A + fill in

    expectedPatternL_T.row_start.push_back(0);  expectedPatternL_T.row_start.push_back(3);    expectedPatternL_T.row_start.push_back(7);
    expectedPatternL_T.row_start.push_back(10); expectedPatternL_T.row_start.push_back(13);   expectedPatternL_T.row_start.push_back(16);
    expectedPatternL_T.row_start.push_back(18); expectedPatternL_T.row_start.push_back(20);   expectedPatternL_T.row_start.push_back(21);

    expectedPatternL_T.col.push_back(0); expectedPatternL_T.col.push_back(4); expectedPatternL_T.col.push_back(5); // row 0
    expectedPatternL_T.col.push_back(1); expectedPatternL_T.col.push_back(3); expectedPatternL_T.col.push_back(4); expectedPatternL_T.col.push_back(7); // row 1
    expectedPatternL_T.col.push_back(2); expectedPatternL_T.col.push_back(3); expectedPatternL_T.col.push_back(7); // row 2
    expectedPatternL_T.col.push_back(3); expectedPatternL_T.col.push_back(4); expectedPatternL_T.col.push_back(7); // row 3
    expectedPatternL_T.col.push_back(4); expectedPatternL_T.col.push_back(5); expectedPatternL_T.col.push_back(7); // row 4
    expectedPatternL_T.col.push_back(5); expectedPatternL_T.col.push_back(7); // row 5
    expectedPatternL_T.col.push_back(6); expectedPatternL_T.col.push_back(7); // row 6
    expectedPatternL_T.col.push_back(7); // row 7

    CSRMatrix expectedL_T;
    expectedL_T.symmetric = expectedPatternL_T.symmetric;
    expectedL_T.rows = expectedPatternL_T.rows;
    expectedL_T.cols = expectedPatternL_T.cols;
    expectedL_T.nnz = expectedPatternL_T.nnz;
    expectedL_T.row_start = expectedPatternL_T.row_start.data;
    expectedL_T.col = expectedPatternL_T.col.data;
    expectedL_T.data.push_back(3.16227766); expectedL_T.data.push_back(3.16227766e-01); expectedL_T.data.push_back(3.16227766e-01);// row 0
    expectedL_T.data.push_back(3.16227766e+00); expectedL_T.data.push_back(3.16227766e-01); expectedL_T.data.push_back(3.16227766e-01); expectedL_T.data.push_back(3.16227766e-01);// row 1
    expectedL_T.data.push_back(3.16227766e+00); expectedL_T.data.push_back(3.16227766e-01); expectedL_T.data.push_back(3.16227766e-01); // row 2
    expectedL_T.data.push_back(3.13049517e+00); expectedL_T.data.push_back(-3.19438282e-02); expectedL_T.data.push_back(-6.38876565e-02); // row 3
    expectedL_T.data.push_back(3.13033219e+00); expectedL_T.data.push_back(-3.19454914e-02); expectedL_T.data.push_back(-3.25974402e-02); // row 4
    expectedL_T.data.push_back(3.14626437e+00); expectedL_T.data.push_back(-3.30977033e-04); // row 5
    expectedL_T.data.push_back(3.16227766e+00); expectedL_T.data.push_back(3.16227766e-01); // row 6
    expectedL_T.data.push_back(3.11365632e+00); // row 7

    TArray<double> x(8), expectedX(0);
    expectedX.push_back(0.08240513); expectedX.push_back(0.0757028); expectedX.push_back(0.08412173);
    expectedX.push_back(0.08401755); expectedX.push_back(0.08418921); expectedX.push_back(0.09175949); 
    expectedX.push_back(0.09252348); expectedX.push_back(0.0747652);
    TArray<double> b(0);  // 0 of size because resized through pushbacks
    b.push_back(1); b.push_back(1); b.push_back(1); b.push_back(1);
    b.push_back(1); b.push_back(1); b.push_back(1); b.push_back(1);

    SparseCholeskyOrdering ordering;
    SparseCholeskySymbolic symbolic(&A);
    SparseCholeskyFactorization factorization(&A);

    std::cout << "==================== 1. METATEST TESTING =======================" << std::endl;
    std::cout << "That is, checking that the test is correct" << std::endl;
    std::cout << "1. Checking that Ax=b holds" << std::endl;
    A.mvp(expectedX.data, x.data);
    if ( check_solution(&x, &b) == false ) {
        std::cout << "Ax=b does not hold. Not going further. Check that you filled"
            "A and patternA correctly" << std::endl;
        return 1;
    } else {
        std::cout << "Ax=b holds." << std::endl;
    }

    // checking that LL^T*x=b holds
    TArray<double> tmpX(8);
    expectedL_T.mvp(expectedX.data, tmpX.data);
    expectedL.mvp(tmpX.data, x.data);
    if ( check_solution(&x, &b) == false ) {
        std::cout << "LL^T*x=b does not hold. Not going further. Check that you filled"
            "expectedL, expectedL_T and their patterns correctly" << std::endl;
        return 1;
    } else {
        std::cout << "LL^T*x=b holds." << std::endl;
    }
    

    // solver.initialize(&A) is equivalent to the following code
    // ==================== 2. ORDERING PHASE TESTING =======================
    ordering.order();

    // ==================== 3. SYMBOLIC PHASE TESTING =======================
    CSRPattern* patternL = symbolic.buildPatternL();
    if (check_pattern(patternL, &expectedPatternL) == false) {
        std::cout << "Failed Symbolic (L) test" << std::endl;
        patternL = &expectedPatternL;
    } else {
        std::cout << "Succeded Symbolic (L) test" << std::endl;
    }
    CSRPattern* patternL_T = symbolic.buildPatternL_T();
    if (check_pattern(patternL_T, &expectedPatternL_T) == false) {
        std::cout << "Failed Symbolic (L^T) test" << std::endl;
        patternL = &expectedPatternL;
    } else {
        std::cout << "Succeded Symbolic (L_T) test" << std::endl;
    }

    // ==================== 4. FACTORIZATION PHASE TESTING =======================
    // 3. Factorization phase
    factorization.setPatternL(patternL);
    CSRMatrix* factor = factorization.factorize();
    if (check_matrix(factor, &expectedL) == false) {
        std::cout << "Failed Factorization test" << std::endl;
        factor = &expectedL;
    } else {
        std::cout << "Succeded Factorization test" << std::endl;
    }

    // TODO: obtain factor_T (that is, L transposed) given factor
    // ...

    // ==================== 4. SOLVE PHASE TESTING =======================
    // solver.solve(x,b) is equivalent to the following code
    // TODO: implement resolution of LL^T*x = b

    if (check_solution(&x, &expectedX) == false) {
        std::cout << "Failed Solution test" << std::endl;
        return 1;
    } else {
        std::cout << "Succeded Solution test" << std::endl;
    }

    return 0;
}