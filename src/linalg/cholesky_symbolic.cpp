#include <vector>
#include <algorithm>
#include "cholesky.h"
#include "adjacency_graph.h"

SparseCholeskySymbolic::SparseCholeskySymbolic(CSRMatrix* A) 
{
    this->A = A;
}
const CholeskyTree& SparseCholeskySymbolic::buildTree() {
    this->isTreeBuilt = true;

    this->tree = CholeskyTree(A->rows);

    AdjacencyGraph G(A);
    std::vector<uint32_t> ancestor(A->rows);
    // Build cholesky tree
    for (uint32_t i=0; i < A->rows; i++) {
        tree.parent(i) = 0; ancestor[i] = 0;

        const std::vector<uint32_t> adj_i = G.adj(i);
        for (uint32_t j: adj_i) {
            if (j >= i) continue;

            uint32_t jroot = j;
            while (ancestor[jroot] != 0 && ancestor[jroot] != i) {
                uint32_t l = ancestor[jroot];
                ancestor[jroot] = i;    // path compression -> next time start from i and 
                                        //                     don't go through the path 
                                        //                     from jroot to i 
                jroot = l;
            }

            if (ancestor[jroot] == 0 ) {
                ancestor[jroot] = i;
                tree.parent(jroot) = i;
            }
        }
    }

    // end of tree construction

    return this->tree;
}
CSRPattern* SparseCholeskySymbolic::buildPatternL() {
    if ( !this->isTreeBuilt ) {
        this->buildTree();
    }
    CSRPattern* patternL = new CSRPattern();
    patternL->rows = A->rows;
    patternL->cols = A->cols;
    patternL->symmetric = 0;
    patternL->nnz = 0;
    patternL->row_start.resize(A->rows+1);
    patternL->col.resize(0);

    std::vector<uint32_t> mark(A->rows, 0);

    AdjacencyGraph G(A);
    for (uint32_t i=0; i < A->rows; i++) {
        patternL->row_start[i] = patternL->col.size;
        mark[i] = i;

        const std::vector<uint32_t> adj_i = G.adj(i);
        for (uint32_t k : adj_i) {
            if (k >= i) continue;

            uint32_t j = k;

            while (mark[j] != i) {
                mark[j] = i;
                patternL->col.push_back(j);
                j = tree.parent(j);
            }
        }
        std::sort(&patternL->col.data[patternL->row_start[i]], 
                  patternL->col.data + patternL->col.size);
        patternL->col.push_back(i);     // adding the diagonal
    }

    patternL->row_start[A->rows] = patternL->col.size;
    patternL->nnz = patternL->col.size;

    return patternL;
}
// eventually buildPatternL and buildPatternL_T will be unified in buildPattern
CSRPattern* SparseCholeskySymbolic::buildPatternL_T() {
    if ( !this->isTreeBuilt ) {
        this->buildTree();
    }

    return nullptr;
}