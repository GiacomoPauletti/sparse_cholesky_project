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
void SparseCholeskySymbolic::buildPatterns(CSRPattern* patternL, CSRPattern* patternL_T) {
    if ( !this->isTreeBuilt ) {
        this->buildTree();
    }

    // patternL setup
    patternL->rows = A->rows; 
    patternL->cols = A->cols; 
    patternL->symmetric = 0; 
    patternL->nnz = 0; 
    patternL->row_start.resize(A->rows+1); 
    patternL->col.resize(0); 

    // patternL_T setup
    patternL_T->rows = A->cols;
    patternL_T->cols = A->rows;
    patternL_T->symmetric = 0;
    patternL_T->nnz = 0;
    patternL_T->row_start.resize(A->rows+1);
    patternL_T->col.resize(0);
    std::vector<std::vector<uint32_t>> patternL_T_rows(A->cols);

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
                patternL->col.push_back(j);     // (i,j) -> row i, col j (L)
                patternL_T_rows[j].push_back(i); // (i,j) -> row j, col i (L_T)
                j = tree.parent(j);
            }
        }
        std::sort(&patternL->col.data[patternL->row_start[i]], 
                  patternL->col.data + patternL->col.size);
        patternL->col.push_back(i);     // adding the diagonal
        patternL_T_rows[i].push_back(i);
    }

    patternL->row_start[A->rows] = patternL->col.size;
    patternL->nnz = patternL->col.size;

    // Filling patternL_T->row_start and patternL_T->col
    uint32_t counter = 0;
    for (uint32_t i=0; i < A->rows; i++) {
        // Filling row_start
        patternL_T->row_start[i] = counter;
        counter += patternL_T_rows[i].size();

        // Filling col
        std::sort(patternL_T_rows[i].begin(), patternL_T_rows[i].end());
        for (uint32_t j : patternL_T_rows[i]) {
            patternL_T->col.push_back(j);
        }
    }
    patternL_T->row_start[A->rows] = counter;
    patternL_T->nnz = counter;

}