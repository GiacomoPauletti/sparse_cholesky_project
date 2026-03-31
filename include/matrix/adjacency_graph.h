#ifndef ADJACENCY_GRAPH_H
#define ADJACENCY_GRAPH_H

#include <vector>
#include <stdint.h>
#include "sparse_matrix.h"

class AdjacencyGraph {
    private:
        CSRMatrix* matrix;

    public:
        AdjacencyGraph(CSRMatrix* matrix);
        std::vector<uint32_t> adj(int vertex);  // adjacent vertices (aka neighbors) of "vertex"
};

#endif // ADJACENCY_GRAPH_H
