#include <vector>
#include <algorithm>
#include <queue>
#include <numeric>
#include "cholesky.h"
#include "adjacency_graph.h"

/* Below this size, a subgraph is left in its current relative order rather
 * than being split further -- separating tiny pieces costs more than it
 * saves. */
static const size_t ND_MIN_SUBGRAPH = 8;

/* -----------------------------------------------------------------------
 * build_full_adjacency: AdjacencyGraph::adj(i) only returns the entries
 * actually stored in row i, which for a symmetric matrix stored as lower
 * triangle means "neighbours with a smaller index". To walk the graph in
 * both directions (needed for BFS) we build an explicit undirected
 * adjacency list once, combining each row with its mirrored entries --
 * the same forward/backward bookkeeping idea used earlier for patternL_T.
 * ----------------------------------------------------------------------- */
static std::vector<std::vector<uint32_t>> build_full_adjacency(CSRMatrix* A) {
    uint32_t n = (uint32_t)A->rows;
    std::vector<std::vector<uint32_t>> adj(n);
    AdjacencyGraph G(A);

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j : G.adj(i)) {
            if (j == i) continue;
            adj[i].push_back(j);
            adj[j].push_back(i);
        }
    }
    return adj;
}

/* -----------------------------------------------------------------------
 * bfs_levels: BFS restricted to `subset`, starting from `start`.
 * level[v] = -1 for vertices outside subset or unreached (disconnected).
 * Returns the furthest vertex reached (for pseudo-peripheral refinement).
 * ----------------------------------------------------------------------- */
static uint32_t bfs_levels(const std::vector<std::vector<uint32_t>>& adj,
                            const std::vector<char>& inSubset,
                            uint32_t start,
                            std::vector<int>& level,
                            int& maxLevel) {
    std::fill(level.begin(), level.end(), -1);
    std::queue<uint32_t> q;
    level[start] = 0;
    q.push(start);
    uint32_t last = start;
    maxLevel = 0;

    while (!q.empty()) {
        uint32_t u = q.front(); q.pop();
        last = u;
        maxLevel = std::max(maxLevel, level[u]);
        for (uint32_t w : adj[u]) {
            if (!inSubset[w] || level[w] != -1) continue;
            level[w] = level[u] + 1;
            q.push(w);
        }
    }
    return last;
}

/* -----------------------------------------------------------------------
 * connected_components: splits `subset` into its connected components
 * (restricted to edges within `subset`). Components with no edges between
 * them need no separator at all -- they can be ordered fully independently
 * (Theorem 4.7's zero-interaction guarantee, applied one level up: it's
 * cheaper and better than computing any separator between them).
 * ----------------------------------------------------------------------- */
static std::vector<std::vector<uint32_t>> connected_components(
    const std::vector<std::vector<uint32_t>>& adj,
    const std::vector<uint32_t>& subset)
{
    uint32_t n = (uint32_t)adj.size();
    std::vector<char> inSubset(n, 0);
    for (uint32_t v : subset) inSubset[v] = 1;

    std::vector<char> visited(n, 0);
    std::vector<std::vector<uint32_t>> components;

    for (uint32_t start : subset) {
        if (visited[start]) continue;
        std::vector<uint32_t> comp;
        std::queue<uint32_t> q;
        visited[start] = 1;
        q.push(start);
        while (!q.empty()) {
            uint32_t u = q.front(); q.pop();
            comp.push_back(u);
            for (uint32_t w : adj[u]) {
                if (!inSubset[w] || visited[w]) continue;
                visited[w] = 1;
                q.push(w);
            }
        }
        components.push_back(std::move(comp));
    }
    return components;
}

/* -----------------------------------------------------------------------
 * nested_dissection: recursively orders `subset`, appending the resulting
 * global vertex order into `out`. Separator vertices of each split are
 * appended LAST within that split -- eliminating them last is exactly what
 * keeps the two sides from being linked together in the factor.
 * ----------------------------------------------------------------------- */
static void nested_dissection(const std::vector<std::vector<uint32_t>>& adj,
                               std::vector<uint32_t> subset,
                               std::vector<uint32_t>& out)
{
    if (subset.empty()) return;

    /* Multiple components: no separator needed between them at all. */
    auto components = connected_components(adj, subset);
    if (components.size() > 1) {
        for (auto& comp : components)
            nested_dissection(adj, std::move(comp), out);
        return;
    }

    if (subset.size() <= ND_MIN_SUBGRAPH) {
        for (uint32_t v : subset) out.push_back(v);
        return;
    }

    uint32_t n = (uint32_t)adj.size();
    std::vector<char> inSubset(n, 0);
    for (uint32_t v : subset) inSubset[v] = 1;

    /* One round of pseudo-peripheral refinement: BFS from an arbitrary
     * vertex, then BFS again from the vertex it reached furthest -- a
     * cheap way to get a level structure with more, thinner levels
     * (better separator candidates) than a single arbitrary BFS. */
    std::vector<int> level(n);
    int maxLevel;
    uint32_t far1 = bfs_levels(adj, inSubset, subset[0], level, maxLevel);
    bfs_levels(adj, inSubset, far1, level, maxLevel);

    /* Group subset vertices by level, and find the best separator level:
     * minimize its size, subject to keeping the two sides reasonably
     * balanced (each at least 20% of the subset). */
    std::vector<std::vector<uint32_t>> byLevel(maxLevel + 1);
    for (uint32_t v : subset) byLevel[level[v]].push_back(v);

    std::vector<size_t> prefix(maxLevel + 2, 0);
    for (int l = 0; l <= maxLevel; l++)
        prefix[l + 1] = prefix[l] + byLevel[l].size();

    size_t total = subset.size();
    int bestLevel = -1;
    size_t bestSize = total + 1;
    for (int l = 0; l <= maxLevel; l++) {
        size_t before = prefix[l];
        size_t after  = total - prefix[l + 1];
        if (before < total / 5 || after < total / 5) continue; /* too unbalanced */
        if (byLevel[l].size() < bestSize) {
            bestSize = byLevel[l].size();
            bestLevel = l;
        }
    }
    if (bestLevel < 0) {
        /* No balanced level found (can happen on odd/degenerate graphs) --
         * fall back to the single smallest level, wherever it is. */
        for (int l = 0; l <= maxLevel; l++)
            if (byLevel[l].size() < bestSize) { bestSize = byLevel[l].size(); bestLevel = l; }
    }

    std::vector<uint32_t> left, right, sep;
    for (uint32_t v : subset) {
        if (level[v] < bestLevel)       left.push_back(v);
        else if (level[v] == bestLevel) sep.push_back(v);
        else                             right.push_back(v);
    }

    /* Guard against a degenerate split (e.g. separator = everything). */
    if (left.empty() || right.empty()) {
        for (uint32_t v : subset) out.push_back(v);
        return;
    }

    nested_dissection(adj, std::move(left), out);
    nested_dissection(adj, std::move(right), out);
    for (uint32_t v : sep) out.push_back(v);
}

/* ========================================================================
 * SparseCholeskyOrdering
 * ===================================================================== */

SparseCholeskyOrdering::SparseCholeskyOrdering() {}

SparseCholeskyOrdering::SparseCholeskyOrdering(CSRMatrix* A) : A(A) {}

void SparseCholeskyOrdering::setMatrix(CSRMatrix* A) {
    this->A = A;
    computed = false;
}

void SparseCholeskyOrdering::order() {
    if (!A) return; /* no matrix set (e.g. default-constructed, unused) */

    uint32_t n = (uint32_t)A->rows;
    auto adj = build_full_adjacency(A);

    std::vector<uint32_t> all(n);
    std::iota(all.begin(), all.end(), 0);

    std::vector<uint32_t> newOrder;
    newOrder.reserve(n);
    nested_dissection(adj, std::move(all), newOrder);

    /* newOrder[k] = which original vertex sits at new position k.
     * perm[old] = new position -- what buildTree()/buildPatterns() need
     * if applied to a relabeled matrix. */
    perm.assign(n, 0);
    for (uint32_t k = 0; k < n; k++)
        perm[newOrder[k]] = k;

    computed = true;
}

const std::vector<uint32_t>& SparseCholeskyOrdering::permutation() const {
    return perm;
}

void SparseCholeskyOrdering::applyToPattern(const CSRPattern& P, CSRPattern* out) const {
    uint32_t n = (uint32_t)P.rows;

    /* Build full undirected edge list in the ORIGINAL numbering first
     * (P typically stores only the lower triangle). */
    std::vector<std::vector<uint32_t>> fullAdj(n);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t k = P.row_start[i]; k < P.row_start[i + 1]; k++) {
            uint32_t j = P.col[k];
            if (j == i) continue;
            fullAdj[i].push_back(j);
            fullAdj[j].push_back(i);
        }
    }

    out->symmetric = P.symmetric;
    out->rows = out->cols = n;
    out->row_start.resize(n + 1);
    out->col.resize(0);

    /* new2old[k] = which original vertex sits at new position k */
    std::vector<uint32_t> new2old(n);
    for (uint32_t v = 0; v < n; v++) new2old[perm[v]] = v;

    for (uint32_t newI = 0; newI < n; newI++) {
        out->row_start[newI] = out->col.size;
        uint32_t oldI = new2old[newI];

        std::vector<uint32_t> row;
        for (uint32_t oldJ : fullAdj[oldI]) {
            uint32_t newJ = perm[oldJ];
            if (newJ < newI) row.push_back(newJ); /* keep lower triangle only */
        }
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        for (uint32_t c : row) out->col.push_back(c);
        out->col.push_back(newI); /* diagonal */
    }
    out->row_start[n] = out->col.size;
    out->nnz = out->col.size;
}