#include "sparse_matrix.h"

class CholeskyTree {
}

class SparseCholeskySolver {
    public:
        SparseCholeskySolver();
        CSRMatrix getFactor();
        void initialize(const CSRMatrix& A);
        solve(double *__restrict x, double *__restrict b);  
}

class SparseCholeskyOrdering {
    public:
        SparseCholeskyOrdering();
        void order();    // Still to be figured out...
}

class SparseCholeskyFactorization {
    public:
        SparseCholeskyFactorization(const CSRMatrix& A, const CSRPattern& LPattern);
        CSRMatrix factorize();
}

class SparseCholeskySymbolic {
    public:
        SparseCholeskySymbolic(const CSRMatrix& A);
        CholeskyTree buildTree();
        CSRPattern buildPatternL();
        CSRPattern buildPatternL_T();  // eventually buildPatternL and buildPatternL_T will be unified in buildPattern
}
        
        
