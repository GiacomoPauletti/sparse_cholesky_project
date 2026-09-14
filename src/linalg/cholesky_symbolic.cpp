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

    for (uint32_t i = 0; i < A->rows; i++) {
        tree.parent(i) = i;   /* default: node points to itself */
        ancestor[i] = i;      /* sentinel for this iteration */

        const std::vector<uint32_t> adj_i = G.adj(i);
        for (uint32_t j : adj_i) {
            if (j >= i) continue;

            // path compression
            uint32_t jroot = j;
            while ((uint32_t)ancestor[jroot] != i) {
                uint32_t l = ancestor[jroot];
                ancestor[jroot] = i;
                jroot = l;
            }

            if ((uint32_t)tree.parent(jroot) == jroot) {
                ancestor[jroot] = i; 
                tree.parent(jroot) = i;
            }
        }
    }

    /* root node(s) have no parent — marked with NO_PARENT */
    for (uint32_t i = 0; i < A->rows; i++)
        if ((uint32_t)tree.parent(i) == i)
            tree.parent(i) = CholeskyTree::NO_PARENT;

    return this->tree;
}

void SparseCholeskySymbolic::buildPatterns(CSRPattern* patternL, 
    CSRPattern* patternL_T) {
    
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
    
    // row_L (i) = { }
    std::vector<std::vector<uint32_t>> patternL_T_rows(A->cols);
    
    // mark [i] = 0
    std::vector<uint32_t> mark(A->rows, 0);

    AdjacencyGraph G(A);
    
    // loop over all rows of A
    for (uint32_t i=0; i < A->rows; i++) {
       
        patternL->row_start[i] = patternL->col.size;
        mark[i] = i;
   
        // loop over the below-diagonal entries of row i of A
        const std::vector<uint32_t> adj_i = G.adj(i);
        for (uint32_t k : adj_i) {
            if (k >= i) continue;

            uint32_t j = k;

        // while column j not yet encountered in row i
        while (j != (uint32_t)CholeskyTree::NO_PARENT && mark[j] != i) {
            mark[j] = i; // mark column j as encountered in row i
            patternL->col.push_back(j);   // add column j to row i of L's pattern
            patternL_T_rows[j].push_back(i); // add row i to column j of L^T's pattern
            j = tree.parent(j); // move up the elimination tree to the parent of j
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