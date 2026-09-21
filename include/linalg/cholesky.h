#pragma once

#include <vector>
#include <memory>

#include "array.h"

#include "sparse_matrix.h"
#include "dense_matrix.h"
#include "frontal_matrix.h"
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
        /* Builds the row pattern of L and its transpose (= the column
         * pattern of L). cscToCsr, when asked for, is the permutation that
         * sends a position in patternL_T to the position of the same entry
         * in patternL : it turns "scatter a column of L into a row major L"
         * into one indirect store per entry, instead of a binary search. */
        void buildPatterns(CSRPattern* patternL, CSRPattern* patternL_T,
                           TArray<uint32_t>* cscToCsr = nullptr);
};
        
/* Common state of every numerical factorization : the matrix to factorize
 * and the symbolic phase's output. Derived classes differ only in how they
 * fill L's values. */
class SparseCholeskyFactorization {
    protected:
        CSRMatrix* A = nullptr;
        CSRPattern* patternL = nullptr;   // row pattern of L
        CSRPattern* patternL_T = nullptr; // column pattern of L
        /* position in patternL_T -> position in patternL, see buildPatterns */
        const TArray<uint32_t>* cscToCsr = nullptr;
    public:
        SparseCholeskyFactorization(CSRMatrix* A);
        virtual ~SparseCholeskyFactorization() = default;
        void setPatternL(CSRPattern* patternL);
        /* Multifrontal needs the column structure of L and the scatter map
         * as well; up-looking ignores both. */
        void setPatterns(CSRPattern* patternL, CSRPattern* patternL_T,
                         const TArray<uint32_t>* cscToCsr);
        virtual CSRMatrix* factorize() = 0; // computes the numerical values of L
};

class UplookingSparseCholeskyFactorization : public SparseCholeskyFactorization {
    public:
        UplookingSparseCholeskyFactorization(CSRMatrix* A);
        CSRMatrix* factorize() override;
};

/******************************************************************************
 *
 * GeneratedElement : the update matrix V_k a multifrontal step hands to its
 * parent. Symmetric, dense, of order m = |struct(L(:,k))| - 1, stored like a
 * front (column major, lower triangle only, see SymFrontMatrix) in storage it
 * owns, since it has to outlive the step that produced it.
 *
 * It does NOT derive from DenseMatrix : half of its buffer is garbage, which
 * would silently break DenseMatrix::sum() and ::mvp().
 *
 *****************************************************************************/
class GeneratedElement {
    private:
        uint32_t m;
        TArray<double> data;
        /* rel[t] = position, in the PARENT front, of this element's row t.
         * Ascending, because both index lists are, which is what makes the
         * lower triangle of the child land in the lower triangle of the
         * parent. Filled once in the symbolic precomputation of the
         * factorization, never searched for at factorization time. */
        const uint32_t* rel = nullptr;
    public:
        GeneratedElement(uint32_t m, const uint32_t* rel);
        uint32_t order() const { return m; }
        double& at(uint32_t i, uint32_t j);
        double  at(uint32_t i, uint32_t j) const;
        /* F_k += this, scattered through rel. Lower triangle only. */
        void extendAdd(SymFrontMatrix& F_k) const;
};

class MultifrontalSparseCholeskyFactorization : public SparseCholeskyFactorization {
    protected:
        /* Filled by setup(), read by processNode() */
        CSRMatrix* L = nullptr;
        std::vector<uint32_t> atStart, atRow;   // lower triangle of A, by column
        std::vector<double> atVal;
        std::vector<uint32_t> childStart, childList; // elimination tree children
        std::vector<uint32_t> relIdx;           // extend-add maps, indexed like patternL_T
        uint32_t maxM = 0;                      // largest front order
        std::vector<GeneratedElement*> V;       // generated elements awaiting their parent

        void setup();
        /* One front : assemble, factor, hand V_k to the parent, scatter L(:,k).
         * scratch holds maxM^2 doubles, L_k maxM. Safe to run concurrently on
         * distinct k once all the children of each k are done. */
        void processNode(uint32_t k, double* scratch, double* L_k);
    public:
        MultifrontalSparseCholeskyFactorization(CSRMatrix* A);
        CSRMatrix* factorize() override;
};

/* Multifrontal over OpenMP tasks, the elimination tree as the DAG. The tree is
 * cut into subtrees of bounded work, one task each; above the cut, the last
 * child to finish carries on into its parent. */
class ParMultifrontalSparseCholeskyFactorization : public MultifrontalSparseCholeskyFactorization {
    public:
        ParMultifrontalSparseCholeskyFactorization(CSRMatrix* A);
        CSRMatrix* factorize() override;
};

/* Which numerical factorization SparseCholeskySolver runs. Both produce the
 * same L, to rounding. */
enum CholeskyFactorizationKind {
    CHOLESKY_FACTORIZATION_UPLOOKING,    // Algorithm 5.7, one row of L at a time
    CHOLESKY_FACTORIZATION_MULTIFRONTAL,     // one dense front per column of L
    CHOLESKY_FACTORIZATION_PAR_MULTIFRONTAL  // same, OpenMP over the elimination tree
};

class SparseCholeskySolver : public LinearSolver {
    private:
        CSRMatrix*  factor     = nullptr;
        CSRMatrix*  factor_T   = nullptr;
        CSRPattern* patternL   = nullptr;
        CSRPattern* patternL_T = nullptr;
        SparseCholeskyOrdering ordering;
        SparseCholeskySymbolic symbolic;
        std::unique_ptr<SparseCholeskyFactorization> factorization;
        /* position in patternL_T -> position in patternL. Used to scatter the
         * multifrontal columns into the row major factor, and to build L^T
         * below without a binary search per entry. */
        TArray<uint32_t> cscToCsr;

        CholeskyOrderingKind orderingKind = CHOLESKY_ORDERING_NATURAL;
        CholeskyFactorizationKind factorizationKind = CHOLESKY_FACTORIZATION_UPLOOKING;
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
                             Profiler* profiler = nullptr,
                             CholeskyFactorizationKind factorization =
                                 CHOLESKY_FACTORIZATION_UPLOOKING);
        CSRMatrix* getFactor();
        CholeskyOrderingKind orderingUsed() const { return orderingKind; }
        CholeskyFactorizationKind factorizationUsed() const { return factorizationKind; }
        void initialize(CSRMatrix* A) override;
        // forward substitution of L, backward substitution using L^T
        void solve(double *__restrict x, const double *__restrict b) override;  
        ~SparseCholeskySolver() override;
};