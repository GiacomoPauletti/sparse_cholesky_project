#pragma once

#include <vector>

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
        const std::vector<uint32_t>& permutation() const; // old index -> new index
        void applyToPattern(const CSRPattern& P, CSRPattern* out) const; // build the permuted pattern
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

    public:
        /* profiler is optional : nullptr (the default) disables the
         * per-phase instrumentation of this solver. */
        SparseCholeskySolver(CSRMatrix* A, Profiler* profiler = nullptr);
        CSRMatrix* getFactor();
        void initialize(CSRMatrix* A) override;
        // forward substitution of L, backward substitution using L^T
        void solve(double *__restrict x, const double *__restrict b) override;  
        ~SparseCholeskySolver() override;
};