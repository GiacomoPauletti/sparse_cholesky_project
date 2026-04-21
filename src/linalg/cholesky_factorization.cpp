// since patternL is CSR

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

    // allocate L
    CSRMatrix* L = new CSRMatrix();
    L->symmetric = false;
    L->rows      = n;
    L->cols      = n;
    L->nnz       = nnz;
    L->row_start = patternL->row_start.data;
    L->col       = patternL->col.data;
    L->data.resize(nnz);
    
    // fill L's data with A's values where the pattern matches, and 0.0 otherwise
    for (size_t i = 0; i < nnz; i++)
    L->data[i] = 0.0;

    // we exploit the fact that both A and L have their column indices
    // sorted in increasing order within each row. Instead of binary searching
    // for each entry of A in L's row we walk both rows simultaneously with two
    // pointers, advancing each only forward - this is a merge scan.
    for (uint32_t i = 0; i < n; i++) {
        uint32_t lp = patternL->row_start[i];
        // ap walks through A's entries in row i one column j at a time
        for (uint32_t ap = A->row_start[i]; ap < A->row_start[i + 1]; ap++) {
            uint32_t j = A->col[ap];
            // for each j in A, lp advances through L's entries in the same row
            // until it either finds j or passes it - all entries of L
            // whose column is strictly less than j are skipped
            while (lp < patternL->row_start[i + 1] && patternL->col[lp] < j)
                lp++; // either past the end of L's row or sitting on a col >= j

            // the if checks if it landed exactly on j - meaning L has an
            // entry at (i, j).
            // note that lp is never reset between iterations of ap - it only moves
            // forward. So the total work for the whole row is O(nnz_A_row + nnz_L_row)
            // rather than O(nnz_A_row * log(nnz_L_row)).
            if (lp < patternL->row_start[i + 1] && patternL->col[lp] == j)
                L->data[lp] = A->data[ap];
        }
    }

    // step 2: build CSC index of L from patternL
    std::vector<uint32_t> col_start(n + 1, 0);
    std::vector<uint32_t> row_in_col(nnz);
    std::vector<uint32_t> data_pos(nnz);

    // we scan every nz and increment the count for its column.
    // the off-by-one is to set up the prefix sum cleanly.
    for (size_t p = 0; p < nnz; p++)
        col_start[patternL->col[p] + 1]++;

    // we accumulate the counts so that col_start[j] becomes the starting
    // position of column j in the output arrays. After this, col_start[j]
    // to col_start[j+1] is the range belonging to column j.
    // This is exactly the same structure as row_start in CSR, but for columns.
    for (uint32_t j = 0; j < n; j++)
        col_start[j + 1] += col_start[j];

    // col_cursor is a working copy of col_start - it starts pointing at the
    // beginning of each column's slot and advances as we fill entries in.
    // iterating rows 0..n-1 ensures entries within each column end up row-sorted.
    std::vector<uint32_t> col_cursor(col_start.begin(), col_start.end());
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t p = patternL->row_start[i]; p < patternL->row_start[i + 1]; p++) {
            uint32_t j       = patternL->col[p];
            uint32_t slot    = col_cursor[j]++;
            row_in_col[slot] = i;
            data_pos[slot]   = p;
        }
    }

    auto csc_lower_bound = [&](uint32_t k, uint32_t j) -> uint32_t {
        uint32_t lo = col_start[k], hi = col_start[k + 1];
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (row_in_col[mid] < j) lo = mid + 1;
            else                     hi = mid;
        }
        return lo;
    };

    auto find_in_row = [&](uint32_t i, uint32_t j) -> int {
        int lo = (int)L->row_start[i], hi = (int)L->row_start[i + 1] - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if      (L->col[mid] == j) return mid;
            else if (L->col[mid] <  j) lo = mid + 1;
            else                       hi = mid - 1;
        }
        return -1;
    };

    for (uint32_t j = 0; j < n; j++) {
        uint32_t lj_s   = L->row_start[j];
        uint32_t lj_e   = L->row_start[j + 1];
        uint32_t diag_j = lj_e - 1;
        ASSERT(L->col[diag_j] == j);

        for (uint32_t p = lj_s; p < diag_j; p++) {
            uint32_t k   = L->col[p];
            double   ljk = L->data[p];
            if (ljk == 0.0) continue;

            for (uint32_t q = csc_lower_bound(k, j); q < col_start[k + 1]; q++) {
                uint32_t i = row_in_col[q];
                if (i == j) {
                    L->data[diag_j] -= L->data[data_pos[q]] * ljk;
                } else {
                    int lij = find_in_row(i, j);
                    if (lij >= 0) L->data[lij] -= L->data[data_pos[q]] * ljk;
                }
            }
        }
 
    ASSERT_ALWAYS(L->data[diag_j] > 0.0);
        L->data[diag_j] = std::sqrt(L->data[diag_j]);
        double ljj = L->data[diag_j];

        for (uint32_t q = col_start[j]; q < col_start[j + 1]; q++) {
            if (row_in_col[q] > j) L->data[data_pos[q]] /= ljj;
        }
    }

    return L;
}