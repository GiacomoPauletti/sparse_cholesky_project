``` mermaid
classDiagram
    class CSRMatrix {}
    class CSRPattern {}

    class SparseCholeskySolver {
        +SparseCholeskySolver()
        +getFactor() CSRMatrix
        +initialize(CSRMatrix m)
        +solve()
    }

    class SparseCholeskyOrdering {
        +SparseCholeskyOrdering(CSRMatrix m)
        +order()
    }

    class SparseCholeskySymbolic {
        +SparseCholeskySymbolic(CSRMatrix m)
        +build_tree() CholeskyTree
        +build_sparsity() CSRPattern
    }

    class SparseCholeskyFactorization {
        +SparseCholeskyFactorization(CSRMatrix m, CSRPattern L_pattern)
        +factorize() CSRMatrix
    }

    class CholeskyTree {}

    SparseCholeskySolver --> CSRMatrix
    SparseCholeskySolver --> SparseCholeskyOrdering
    SparseCholeskySolver --> SparseCholeskySymbolic
    SparseCholeskySolver --> SparseCholeskyFactorization

    SparseCholeskySymbolic --> CSRPattern
    SparseCholeskyFactorization --> CSRPattern
    SparseCholeskySymbolic --> CholeskyTree
    CSRMatrix --> CSRPattern
```

### Classes description
#### SparseCholeskySolver
The `initialize(...)` method does the heavy job, that is:
1. Permutation phase: computes a permutation of $A$ to improve the performances of the solver (optional step)
2. Symbolic phase: computes the sparsity pattern of $L$
3. Factorization phase: computes the $L$ based on the results of Symbolic phase. TODO: exept the sparsity pattern, what is needed?

At the end of the `initialize(...)` method, the factor `L` is saved locally and exposed through `getFactor()` interface.

The `solve(...)` method performs the Solve phase, which consists in solving $Ax=b$ using the equivalent system $LL^Tx=b$. Here forward substitution (to get y:Ly=b) and then backward substitution (to get x:L^Tx=y) are used.

At the momento only a sequential version is considered. 
A parallel version can be considered since all 4 phases are parallelizable. 
A better performance of the parallel code can be obtained using _supernodes_ in the symbolic phase. 
