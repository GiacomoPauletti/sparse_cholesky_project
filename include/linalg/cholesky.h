#include <vector>

#include "sparse_matrix.h"

class CholeskyTree {
    public:
        CholeskyTree();
        CholeskyTree(int num_nodes);
        int& operator[](int index);
        int& parent(int index);
        int parent(int index) const;
    private:
        std::vector<int> parentship;
};

class SparseCholeskyOrdering {
    public:
        SparseCholeskyOrdering();
        void order();    // Still to be figured out...
};

class SparseCholeskySymbolic {
    private:
        CSRMatrix* A;
        CholeskyTree tree;
        bool isTreeBuilt = false;
    public:
        SparseCholeskySymbolic(CSRMatrix* A);
        const CholeskyTree& buildTree();
        void buildPatterns(CSRPattern* patternL, CSRPattern* patternL_T);
};
        
class SparseCholeskyFactorization {
    private:
        CSRMatrix* A;
        CSRPattern* patternL;
    public:
        SparseCholeskyFactorization(CSRMatrix* A);
        void setPatternL(CSRPattern* patternL);
        CSRMatrix* factorize();
};

class SparseCholeskySolver {
    private:
        CSRMatrix* factor;
        SparseCholeskyOrdering ordering;
        SparseCholeskySymbolic symbolic;
        SparseCholeskyFactorization factorization;

    public:
        SparseCholeskySolver(CSRMatrix* A);
        CSRMatrix* getFactor();
        void initialize(CSRMatrix* A);
        void solve(double *__restrict x, double *__restrict b);  
        ~SparseCholeskySolver();
};