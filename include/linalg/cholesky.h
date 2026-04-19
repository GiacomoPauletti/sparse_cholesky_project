#include <vector>

#include "sparse_matrix.h"


// each class corresponds to a phase of the Sparse_Cholesky Pipeline


class CholeskyTree {
    public:
        CholeskyTree(); // elimination tree
        CholeskyTree(int num_nodes); // for each node
        int& operator[](int index);
        int& parent(int index);
        int  parent(int index) const;
    private:
        std::vector<int> parentship; // it stores its parent in the tree
};

class SparseCholeskyOrdering {
    public:
        SparseCholeskyOrdering(); // meant to reduce fill-in
        void order();    // Still to be figured out...
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
    public:
        SparseCholeskyFactorization(CSRMatrix* A);
        void setPatternL(CSRPattern* patternL); // computes the numerical values of L 
        CSRMatrix* factorize();
};

class SparseCholeskySolver {
    private:
        CSRMatrix* factor;
        CSRMatrix* factor_T = nullptr;
        SparseCholeskyOrdering ordering;
        SparseCholeskySymbolic symbolic;
        SparseCholeskyFactorization factorization;

    public:
        SparseCholeskySolver(CSRMatrix* A);
        CSRMatrix* getFactor();
        void initialize(CSRMatrix* A);
        // forward substitution of L, backward substitution using L^T
        void solve(double *__restrict x, const double *__restrict b);  
        ~SparseCholeskySolver();
};