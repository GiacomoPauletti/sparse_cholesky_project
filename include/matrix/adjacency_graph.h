#ifndef ADJACENCY_GRAPH_H
#define ADJACENCY_GRAPH_H

#include <vector>
#include <stdint.h>
#include "sparse_matrix.h"

// wraps a CSRMatrix and exposes it as a graph
// where matrix non-zero entries are interpreted as edges 

// the elimination tree construction works by processing the graph of A
// column by column. For each column j we need to find rows i > j such
// that A(i, j)!=0 which are the neighbours of j in the lower triangular
// part of the graph. The method adj(j) gives us that, assuming A stores
// the lower triangle.
class AdjacencyGraph {
    private:
        CSRMatrix* matrix;

    public:
        AdjacencyGraph(CSRMatrix* matrix);
        // returns the list of neighbouring vertices of a given node 
        // i.e. the column indices of non-zero entries on row vertex
        std::vector<uint32_t> adj(int vertex);  
};

#endif // ADJACENCY_GRAPH_H
