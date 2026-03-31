#include "adjacency_graph.h"

AdjacencyGraph::AdjacencyGraph(CSRMatrix* matrix) {
    this->matrix = matrix;
}

std::vector<uint32_t> AdjacencyGraph::adj(int vertex) {
    assert(vertex<matrix->rows);
    std::vector<uint32_t> adj_set(0);
    for (int i = matrix->row_start[vertex]; i < matrix->row_start[vertex+1]; i++) {
        adj_set.push_back(matrix->col[i]);
    }
    return adj_set;
}