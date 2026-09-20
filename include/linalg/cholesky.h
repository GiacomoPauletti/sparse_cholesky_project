#pragma once

#include <vector>

#include "array.h"

#include "sparse_matrix.h"
#include "linear_solver.h"


// each class corresponds to a phase of the Sparse_Cholesky Pipeline


class CholeskyTree {
    public:
        static constexpr int NO_PARENT = -1;
        CholeskyTree(); // elimination tree
        CholeskyTree(int num_nodes); // for each node
        int& operator[](int index);
        int& parent(int index);
        int  parent(int index) const;
    private:
        std::vector<int> parentship; // it stores its parent in the tree
};

/* Which permutation SparseCholeskySolver factorizes in. The matrix itself is
 * the same either way : reordering only changes how much fill in the factor
 * suffers, never the solution. */
enum CholeskyOrderingKind {
    CHOLESKY_ORDERING_NATURAL, // factorize A as handed in
    CHOLESKY_ORDERING_NESTED   // factorize P A P^T, P from nested dissection
};

class SparseCholeskyOrdering {
    private:
        CSRMatrix* A = nullptr;
        std::vector<uint32_t> perm;   // old index -> new index
        bool computed = false;
    public:
        SparseCholeskyOrdering();              // kept for backward compatibility
        SparseCholeskyOrdering(CSRMatrix* A);  // preferred: operates on any CSR matrix
        void setMatrix(CSRMatrix* A);
        void order();                           // computes a nested-dissection permutation
        bool isComputed() const;
        const std::vector<uint32_t>& permutation() const; // old index -> new index
        void applyToPattern(const CSRPattern& P, CSRPattern* out) const; // build the permuted pattern
        /* Builds P A P^T, pattern AND values, in the same lower triangular
         * convention as A (columns ascending, diagonal last). outPattern owns
         * the index arrays that out points into, so it must outlive out. */
        void applyToMatrix(const CSRMatrix& A, CSRPattern* outPattern,
                           CSRMatrix* out) const;
};

class SparseCholeskySymbolic {
    private:
        CSRMatrix* A;
        CholeskyTree tree;
        bool isTreeBuilt = false;
    public:
        SparseCholeskySymbolic(CSRMatrix* A);
        const CholeskyTree& buildTree(); // constructs the elimination tree
        void buildPatterns(CSRPattern* patternL, CSRPattern* patternL_T);
};
        
class SparseCholeskyFactorization {
    private:
        CSRMatrix* A;
        CSRPattern* patternL; // sparsity pattern
        CSRPattern* patternL_T = nullptr;
    public:
        SparseCholeskyFactorization(CSRMatrix* A);
        void setPatternL(CSRPattern* patternL); // computes the numerical values of L 
        CSRMatrix* factorize();
};

class SparseCholeskySolver : public LinearSolver {
    private:
        CSRMatrix*  factor     = nullptr;
        CSRMatrix*  factor_T   = nullptr;
        CSRPattern* patternL   = nullptr;
        CSRPattern* patternL_T = nullptr;
        SparseCholeskyOrdering ordering;
        SparseCholeskySymbolic symbolic;
        SparseCholeskyFactorization factorization;

        CholeskyOrderingKind orderingKind = CHOLESKY_ORDERING_NATURAL;
        /* Only used when orderingKind is NESTED : P A P^T, which is what is
         * actually factorized, and the scratch the substitutions run on. */
        CSRPattern permPattern;
        CSRMatrix  permA;
        std::vector<uint32_t> perm;   // old index -> new index
        TArray<double> permRhs;
        TArray<double> permSol;

        /* L y = b then L^T x = y, both in the factorized numbering. */
        void substitute(double *__restrict x, const double *__restrict b);
        /* Frees the factor and the patterns, so initialize() can be called
         * again (the LinearSolver contract allows it when A's values change). */
        void release();

    public:
        /* ordering selects what is factorized : the matrix as given, or its
         * nested dissection permutation, which is the same solve but with far
         * less fill in. profiler is optional : nullptr disables the per-phase
         * instrumentation of this solver. */
        SparseCholeskySolver(CSRMatrix* A,
                             CholeskyOrderingKind ordering = CHOLESKY_ORDERING_NATURAL,
                             Profiler* profiler = nullptr);
        CSRMatrix* getFactor();
        CholeskyOrderingKind orderingUsed() const { return orderingKind; }
        void initialize(CSRMatrix* A) override;
        // forward substitution of L, backward substitution using L^T
        void solve(double *__restrict x, const double *__restrict b) override;  
        ~SparseCholeskySolver() override;
};