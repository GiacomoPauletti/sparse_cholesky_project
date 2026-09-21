#include "cholesky.h"
#include "sys_utils.h"

#include <atomic>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads() { return 1; }
static int omp_get_thread_num() { return 0; }
#endif


// ===================== COMMON FACTORIZATION STATE =======================

SparseCholeskyFactorization::SparseCholeskyFactorization(CSRMatrix* A)
    : A{A}
{
}

void SparseCholeskyFactorization::setPatternL(CSRPattern* patternL) {
    this->patternL = patternL;
}

void SparseCholeskyFactorization::setPatterns(CSRPattern* patternL,
                                              CSRPattern* patternL_T,
                                              const TArray<uint32_t>* cscToCsr) {
    this->patternL   = patternL;
    this->patternL_T = patternL_T;
    this->cscToCsr   = cscToCsr;
}



// ===================== UPLOOKING SPARSE CHOLESKY FACTORIZATION =======================

UplookingSparseCholeskyFactorization::UplookingSparseCholeskyFactorization(CSRMatrix* A)
    : SparseCholeskyFactorization{A}
{
}

CSRMatrix* UplookingSparseCholeskyFactorization::factorize() {
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



// ===================== MULTIFRONTAL SPARSE CHOLESKY FACTORIZATION =======================

GeneratedElement::GeneratedElement(uint32_t m, const uint32_t* rel)
    : m{m}, data((size_t)m * m), rel{rel}
{
}

double& GeneratedElement::at(uint32_t i, uint32_t j) {
    ASSERT(j <= i && i < m);
    return data.data[(size_t)j * m + i];
}

double GeneratedElement::at(uint32_t i, uint32_t j) const {
    ASSERT(j <= i && i < m);
    return data.data[(size_t)j * m + i];
}

void GeneratedElement::extendAdd(SymFrontMatrix& F_k) const {
    /* rel is ascending, so rel[i] >= rel[j] whenever i >= j : the lower
     * triangle of the child always lands in the lower triangle of the
     * parent, never straddles its diagonal. That is what lets this loop
     * touch half of the child only. */
    for (uint32_t j = 0; j < m; j++) {
        uint32_t jp = rel[j];
        for (uint32_t i = j; i < m; i++)
            F_k.at(rel[i], jp) += at(i, j);
    }
}


MultifrontalSparseCholeskyFactorization::MultifrontalSparseCholeskyFactorization(CSRMatrix* A)
    : SparseCholeskyFactorization{A}
{
}

/* Pseudocode
 * do k = 0:n-1
 *    Assemble F_k = [a_kk, a_*k^T ; a_*k, 0] (+) V_{c_1} (+) ... (+) V_{c_r}
 *    One Cholesky step on F_k -> L(:,k) and V_k
 * setup() is everything before the loop, processNode(k) is one iteration. */
void MultifrontalSparseCholeskyFactorization::setup() {
    ASSERT(A != nullptr);
    ASSERT_ALWAYS(A->rows == A->cols);
    ASSERT(patternL != nullptr);
    ASSERT(patternL_T != nullptr);   // the column structure IS the front structure
    ASSERT(cscToCsr != nullptr);     // needed to scatter columns into a row major L

    const uint32_t n   = A->rows;
    const size_t   nnz = patternL->nnz;

    // === 1. Sparsity pattern setup
    L = new CSRMatrix();
    L->symmetric = false;
    L->rows      = n;
    L->cols      = n;
    L->nnz       = nnz;
    L->row_start = patternL->row_start.data;
    L->col       = patternL->col.data;
    L->data.resize(nnz);
    /* No zero fill : the scatter at 3.5 below writes every entry of L exactly
     * once, cscToCsr being a bijection between the two patterns. */

    /* Row k of patternL_T is column k of L : the global row indices of front
     * k, ascending, starting with k itself. */
    const uint32_t* rsT  = patternL_T->row_start.data;
    const uint32_t* colT = patternL_T->col.data;

    // === 2. Precomputation, all of it O(nnz(A) + nnz(L)) and done once

    /* 2.1 Transpose of the lower triangle of A.
     * A is stored by ROWS, but a front needs COLUMN k of A, which is
     * scattered across rows k..n-1. One counting pass makes every column
     * contiguous and ascending in row index. (Entries above the diagonal, if
     * A happens to store them, are dropped : the lower triangle is all the
     * factorization ever needs.) */
    atStart.assign(n + 1, 0);
    size_t nnzLower = 0;
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t p = A->row_start[i]; p < A->row_start[i + 1]; p++) {
            uint32_t j = A->col[p];
            if (j > i) continue;
            atStart[j]++;
            nnzLower++;
        }
    }
    {
        uint32_t acc = 0;
        for (uint32_t j = 0; j < n; j++) {
            uint32_t c = atStart[j];
            atStart[j] = acc;
            acc += c;
        }
        atStart[n] = acc;
    }
    atRow.resize(nnzLower);
    atVal.resize(nnzLower);
    {
        TArray<uint32_t> cursor(n);
        for (uint32_t j = 0; j < n; j++) cursor[j] = atStart[j];
        /* i ascending -> each column comes out sorted by row index for free */
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t p = A->row_start[i]; p < A->row_start[i + 1]; p++) {
                uint32_t j = A->col[p];
                if (j > i) continue;
                uint32_t q = cursor[j]++;
                atRow[q] = i;
                atVal[q] = A->data[p];
            }
        }
    }

    /* 2.2 Children of each node.
     * parent(k) = min { i > k : L(i,k) != 0 }, which is simply the second
     * entry of column k -- the elimination tree is already in patternL_T, so
     * nothing has to be passed in from the symbolic phase. A column with no
     * off-diagonal entry is a root. */
    childStart.assign(n + 1, 0);
    for (uint32_t k = 0; k < n; k++)
        if (rsT[k + 1] - rsT[k] > 1) childStart[colT[rsT[k] + 1]]++;
    {
        uint32_t acc = 0;
        for (uint32_t k = 0; k < n; k++) {
            uint32_t c = childStart[k];
            childStart[k] = acc;
            acc += c;
        }
        childStart[n] = acc;
    }
    childList.resize(childStart[n]);
    {
        std::vector<uint32_t> cursor(childStart.begin(), childStart.end());
        for (uint32_t k = 0; k < n; k++)
            if (rsT[k + 1] - rsT[k] > 1)
                childList[cursor[colT[rsT[k] + 1]]++] = k;
    }

    /* 2.3 Extend-add maps and the largest front.
     * struct(V_k) is a subset of struct(L(:,parent)) and both are ascending,
     * so each map is a two pointer merge -- no search, no hash map, and it is
     * paid once here rather than at every extend-add.
     * relIdx is indexed exactly like patternL_T : the map of V_k lives at
     * positions rsT[k]+1 .. rsT[k+1]-1, and the unused slot rsT[k] keeps the
     * addressing trivial. */
    relIdx.resize(nnz);
    maxM = 0;
    for (uint32_t k = 0; k < n; k++) {
        uint32_t m = rsT[k + 1] - rsT[k];
        if (m > maxM) maxM = m;
        if (m == 1) continue;   // root : no parent, no generated element

        uint32_t p = colT[rsT[k] + 1];
        uint32_t b = rsT[p];
        for (uint32_t a = rsT[k] + 1; a < rsT[k + 1]; a++) {
            while (b < rsT[p + 1] && colT[b] < colT[a]) b++;
            ASSERT(b < rsT[p + 1] && colT[b] == colT[a]);
            relIdx[a] = b - rsT[p];
            b++;
        }
    }

    /* V[j] is the generated element of j, alive from the step that produced
     * it to the step that consumes it, which frees it. */
    V.assign(n, nullptr);
}

void MultifrontalSparseCholeskyFactorization::processNode(uint32_t k,
                                                          double* scratch,
                                                          double* L_k) {
    const uint32_t* rsT  = patternL_T->row_start.data;
    const uint32_t* colT = patternL_T->col.data;

    // === 3.1 F_k setup. Dense and symmetric : no pattern, lower half only
    uint32_t n_rows_k = rsT[k + 1] - rsT[k];   // n_rows_k = |col(k)|
    ASSERT(colT[rsT[k]] == k);                 // the diagonal comes first
    SymFrontMatrix F_k(n_rows_k, scratch);
    F_k.zeroLower();

    // === 3.2 Assemble F_k
    /* -- Column k of A goes into the front's first column. Only the first
     *    column : an entry A(k_i, k_j) with both indices > k is assembled
     *    at ITS own front, not here.
     *    struct(A(:,k)) is a subset of the front's rows, and both are
     *    ascending, so a single forward walk places every value. */
    uint32_t t = 0;
    for (uint32_t q = atStart[k]; q < atStart[k + 1]; q++) {
        uint32_t i = atRow[q];   // i >= k
        while (t < n_rows_k && colT[rsT[k] + t] < i) t++;
        ASSERT(t < n_rows_k && colT[rsT[k] + t] == i);
        F_k.at(t, 0) += atVal[q];
    }

    // -- extend add with the children's generated elements
    for (uint32_t c = childStart[k]; c < childStart[k + 1]; c++) {
        uint32_t j = childList[c];
        ASSERT(V[j] != nullptr);
        V[j]->extendAdd(F_k);
        delete V[j];          // consumed : nothing will ever read it again
        V[j] = nullptr;
    }

    // === 3.3 One step of Cholesky
    double diag = F_k.at(0, 0);
    ASSERT_ALWAYS(diag > 0.0);
    L_k[0] = std::sqrt(diag);
    for (uint32_t i = 1; i < n_rows_k; i++)
        L_k[i] = F_k.at(i, 0) / L_k[0];

    // === 3.4 V_k = F_k[1:,1:] - L_k[1:] L_k[1:]^T
    /* Symmetric rank 1 update, lower triangle only : this is dsyr with
     * uplo = 'L', written out by hand since the project links no BLAS. */
    if (n_rows_k > 1) {
        uint32_t mv = n_rows_k - 1;
        GeneratedElement* V_k = new GeneratedElement(mv, &relIdx[rsT[k] + 1]);
        for (uint32_t j = 0; j < mv; j++) {
            double lj = L_k[j + 1];
            for (uint32_t i = j; i < mv; i++)
                V_k->at(i, j) = F_k.at(i + 1, j + 1) - L_k[i + 1] * lj;
        }
        V[k] = V_k;
    }

    // === 3.5 Scatter L_k into L
    /* L is row major but L_k is a column, so each value has to find its
     * slot in a different row of L. cscToCsr, built with the patterns,
     * makes that one indirect store per entry instead of a binary
     * search. */
    for (uint32_t i = 0; i < n_rows_k; i++)
        L->data[(*cscToCsr)[rsT[k] + i]] = L_k[i];
}

CSRMatrix* MultifrontalSparseCholeskyFactorization::factorize() {
    setup();
    /* One front buffer, sized for the largest front, reused by every column. */
    std::vector<double> scratch((size_t)maxM * maxM + maxM);
    for (uint32_t k = 0; k < A->rows; k++)
        processNode(k, scratch.data(), scratch.data() + (size_t)maxM * maxM);
    /* Each generated element is consumed by its parent; roots produce none. */
    for (GeneratedElement* v : V) ASSERT(v == nullptr);
    return L;
}



// ===================== PARALLEL MULTIFRONTAL SPARSE CHOLESKY FACTORIZATION =======================

ParMultifrontalSparseCholeskyFactorization::ParMultifrontalSparseCholeskyFactorization(CSRMatrix* A)
    : MultifrontalSparseCholeskyFactorization{A}
{
}

CSRMatrix* ParMultifrontalSparseCholeskyFactorization::factorize() {
    setup();
    const uint32_t n = A->rows;
    const uint32_t* rsT  = patternL_T->row_start.data;
    const uint32_t* colT = patternL_T->col.data;
    const uint32_t NONE = UINT32_MAX;
    auto parent = [&](uint32_t k) { return rsT[k + 1] - rsT[k] > 1 ? colT[rsT[k] + 1] : NONE; };

    /* Subtree work, m_k^2 per front. parent(k) > k, so one ascending pass. */
    std::vector<double> sub(n, 0.0);
    double W = 0.0;
    for (uint32_t k = 0; k < n; k++) {
        double m = rsT[k + 1] - rsT[k];
        sub[k] += m * m;
        W += m * m;
        if (parent(k) != NONE) sub[parent(k)] += sub[k];
    }

    /* A node is small if its whole subtree is one task's worth of work (or it
     * is a leaf). owner[k] = root of the task processing k; parents first, so
     * descending k. Nodes above the cut own themselves. */
    const int nthreads = omp_get_max_threads();
    const double T = W / (8.0 * nthreads);
    auto small = [&](uint32_t k) { return sub[k] <= T || childStart[k] == childStart[k + 1]; };
    std::vector<uint32_t> owner(n);
    for (uint32_t k = n; k-- > 0; ) {
        uint32_t p = parent(k);
        owner[k] = (p != NONE && small(k) && small(p)) ? owner[p] : k;
    }

    /* Nodes of each task, ascending so children come first (counting sort). */
    std::vector<uint32_t> first(n + 1, 0), order(n);
    for (uint32_t k = 0; k < n; k++) first[owner[k] + 1]++;
    for (uint32_t k = 0; k < n; k++) first[k + 1] += first[k];
    {
        std::vector<uint32_t> cur(first.begin(), first.end() - 1);
        for (uint32_t k = 0; k < n; k++) order[cur[owner[k]]++] = k;
    }

    /* Children still to finish, for the nodes above the cut. acq_rel makes
     * each child's V visible to the thread that processes the parent. */
    std::vector<std::atomic<uint32_t>> pending(n);
    for (uint32_t k = 0; k < n; k++)
        pending[k].store(childStart[k + 1] - childStart[k], std::memory_order_relaxed);

    std::vector<std::vector<double>> buf(nthreads);
    #pragma omp parallel
    #pragma omp single
    for (uint32_t r = 0; r < n; r++) {
        if (owner[r] != r || !small(r)) continue;
        #pragma omp task firstprivate(r)
        {
            /* Tied task without scheduling points : the thread's buffer is ours. */
            std::vector<double>& b = buf[omp_get_thread_num()];
            if (b.empty()) b.resize((size_t)maxM * maxM + maxM);
            double* scratch = b.data();
            double* L_k = scratch + (size_t)maxM * maxM;

            for (uint32_t i = first[r]; i < first[r + 1]; i++)
                processNode(order[i], scratch, L_k);
            /* Last child in climbs into the parent. */
            for (uint32_t k = r, p; (p = parent(k)) != NONE; k = p) {
                if (pending[p].fetch_sub(1, std::memory_order_acq_rel) != 1) break;
                processNode(p, scratch, L_k);
            }
        }
    }

    for (GeneratedElement* v : V) ASSERT(v == nullptr);
    return L;
}
