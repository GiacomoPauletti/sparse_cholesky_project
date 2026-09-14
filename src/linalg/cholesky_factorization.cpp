#include "cholesky.h"
#include "sys_utils.h"
#include <cmath>
#include <vector>

SparseCholeskyFactorization::SparseCholeskyFactorization(CSRMatrix* A)
    : A{A}
{
}

void SparseCholeskyFactorization::setPatternL(CSRPattern* patternL) {
    this->patternL = patternL;
}

CSRMatrix* SparseCholeskyFactorization::factorize() {
    ASSERT(A != nullptr);
    ASSERT_ALWAYS(A->rows == A->cols);
    ASSERT(patternL != nullptr);

    const uint32_t n   = A->rows;
    const size_t   nnz = patternL->nnz;

    CSRMatrix* L = new CSRMatrix();
    L->symmetric = false;
    L->rows      = n;
    L->cols      = n;
    L->nnz       = nnz;
    L->row_start = patternL->row_start.data;
    L->col       = patternL->col.data;
    L->data.resize(nnz);

    // Step 1: seed L's data with A's values where the pattern matches, 0.0
    // (i.e. pure fill-in) otherwise. Same merge-scan idea as before: both A
    // and L have sorted column indices per row, so a single forward pass
    // over each row does the matching in O(nnz_A_row + nnz_L_row).
    for (size_t i = 0; i < nnz; i++)
        L->data[i] = 0.0;

    for (uint32_t row = 0; row < n; row++) {
        uint32_t lPos    = patternL->row_start[row];
        uint32_t lRowEnd = patternL->row_start[row + 1];

        for (uint32_t aPos = A->row_start[row]; aPos < A->row_start[row + 1]; aPos++) {
            uint32_t aCol = A->col[aPos];

            // advance lPos through L's row until it reaches a column >= aCol
            while (lPos < lRowEnd && patternL->col[lPos] < aCol)
                lPos++;

            // if L's pattern has a matching entry at this column, copy A's value in
            if (lPos < lRowEnd && patternL->col[lPos] == aCol)
                L->data[lPos] = A->data[aPos];
        }
    }

    // Step 2 (up-looking, Algorithm 5.7): compute L one row at a time.
    // Row i only ever needs rows 0..i-1, which are already fully finished
    for (uint32_t row = 0; row < n; row++) {
        uint32_t rowStart    = L->row_start[row];
        uint32_t rowEnd      = L->row_start[row + 1];
        uint32_t diagPos     = rowEnd - 1;   // diagonal is always the last entry
                                              // in the row (buildPatterns pushes it
                                              // last, after sorting the rest)
        ASSERT(L->col[diagPos] == row);

        // off-diagonal entries of this row, ascending column order
        for (uint32_t entry = rowStart; entry < diagPos; entry++) {
            uint32_t col        = L->col[entry];        // the column j < row we're solving for
            uint32_t colDiagPos = L->row_start[col + 1] - 1;
            double   ljj        = L->data[colDiagPos];   // L_jj, already finalized
            ASSERT(ljj != 0.0);

            // sparse dot product: this row's already-computed prefix
            // (columns < col, positions rowStart..entry-1) against row
            // `col` itself (columns < col, positions row_start[col]..colDiagPos-1).
            // Both rows are sorted ascending, so this is a standard
            // two-pointer merge.
            double   accum      = L->data[entry];   // starts as a_ij (0.0 if pure fill-in)
            uint32_t ownPos     = rowStart;
            uint32_t otherPos   = L->row_start[col];
            while (ownPos < entry && otherPos < colDiagPos) {
                uint32_t ownCol   = L->col[ownPos];
                uint32_t otherCol = L->col[otherPos];
                if      (ownCol < otherCol) ownPos++;
                else if (ownCol > otherCol) otherPos++;
                else {
                    accum -= L->data[ownPos] * L->data[otherPos];
                    ownPos++;
                    otherPos++;
                }
            }

            L->data[entry] = accum / ljj;
        }

        // diagonal: a_ii minus the sum of squares of the row's off-diagonal
        // entries just computed above
        double diagAccum = L->data[diagPos];   // starts as a_ii
        for (uint32_t entry = rowStart; entry < diagPos; entry++)
            diagAccum -= L->data[entry] * L->data[entry];

        ASSERT_ALWAYS(diagAccum > 0.0);
        L->data[diagPos] = std::sqrt(diagAccum);
    }

    return L;
}