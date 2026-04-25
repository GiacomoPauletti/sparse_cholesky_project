/**
 * suite_sparse.cpp
 *
 * Reads a list of "GROUP/NAME" matrix identifiers from a config file,
 * downloads each one from sparse.tamu.edu, builds a CSRMatrix, runs the
 * SparseCholeskySolver, and writes results to a CSV file.
 *
 * Usage:
 *   ./suite_sparse [matrices.cfg] [results.csv]
 *
 * Defaults:
 *   config  : matrices.cfg   (next to the binary, or path given as argv[1])
 *   output  : results.csv    (next to the binary, or path given as argv[2])
 *
 * Config file format:
 *   - One "GROUP/NAME" entry per line.
 *   - Lines whose first non-whitespace character is '#' are comments.
 *   - Blank lines are ignored.
 *
 * CSV columns:
 *   matrix, n, nnz, spd, factorize_ms, solve_ms, residual, status
 *
 *   status values: PASS | FAIL_RESIDUAL | SKIP_NOT_SPD | SKIP_NOT_SQUARE |
 *                  SKIP_NOT_SYMMETRIC | ERR_DOWNLOAD | ERR_PARSE | ERR_SOLVER
 */

#include <archive.h>
#include <archive_entry.h>
#include <curl/curl.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cholesky.h"
#include "sparse_matrix.h"

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double elapsed_ms(Clock::time_point t0)
{
    return Ms(Clock::now() - t0).count();
}

static size_t write_to_buffer(void *ptr, size_t size, size_t nmemb,
                               void *userdata)
{
    auto *buf = static_cast<std::vector<char> *>(userdata);
    const char *bytes = static_cast<const char *>(ptr);
    buf->insert(buf->end(), bytes, bytes + size * nmemb);
    return size * nmemb;
}

enum class DownloadStatus { OK, CURL_FAIL, HTTP_FAIL };

static DownloadStatus download(const std::string &url, std::vector<char> &buf)
{
    curl_global_init(CURL_GLOBAL_ALL);
    CURL *handle = curl_easy_init();
    if (!handle) {
        curl_global_cleanup();
        return DownloadStatus::CURL_FAIL;
    }

    curl_easy_setopt(handle, CURLOPT_URL,            url.c_str());
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION,  write_to_buffer);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA,      &buf);
    curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(handle, CURLOPT_TIMEOUT,        120L); /* 2 min max */

    CURLcode rc = curl_easy_perform(handle);

    long http_code = 0;
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(handle);
    curl_global_cleanup();

    if (rc != CURLE_OK)  return DownloadStatus::CURL_FAIL;
    if (http_code != 200) return DownloadStatus::HTTP_FAIL;
    return DownloadStatus::OK;
}

static bool extract_mtx(const std::vector<char> &tar_buf,
                         std::string             &mtx_contents)
{
    struct archive *a = archive_read_new();
    archive_read_support_filter_gzip(a);
    archive_read_support_format_tar(a);

    if (archive_read_open_memory(a, tar_buf.data(), tar_buf.size()) !=
        ARCHIVE_OK) {
        archive_read_free(a);
        return false;
    }

    struct archive_entry *entry;
    while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
        const char *name = archive_entry_pathname(entry);
        const char *ext  = strrchr(name, '.');
        if (!ext || strcmp(ext, ".mtx") != 0) {
            archive_read_data_skip(a);
            continue;
        }
        char chunk[8192];
        la_ssize_t n;
        while ((n = archive_read_data(a, chunk, sizeof(chunk))) > 0)
            mtx_contents.append(chunk, static_cast<size_t>(n));
        break;
    }
    archive_read_free(a);
    return !mtx_contents.empty();
}

struct MtxEntry { int row, col; double val; };

struct ParseResult {
    bool ok          = false;
    bool is_symmetric = false;
    int  rows        = 0;
    int  cols        = 0;
    std::vector<MtxEntry> entries;
};

static ParseResult parse_mtx(const std::string &mtx)
{
    ParseResult R;
    std::istringstream stream(mtx);
    std::string line;

    if (!std::getline(stream, line)) return R;

    /* Lower-case header for keyword search */
    std::string hdr = line;
    for (auto &c : hdr) c = static_cast<char>(tolower(c));

    if (hdr.find("matrixmarket") == std::string::npos) return R;
    R.is_symmetric = (hdr.find("symmetric") != std::string::npos);

    /* Skip comment lines */
    while (std::getline(stream, line))
        if (!line.empty() && line[0] != '%') break;

    int nnz_declared = 0;
    {
        std::istringstream ss(line);
        if (!(ss >> R.rows >> R.cols >> nnz_declared)) return R;
    }

    R.entries.reserve(static_cast<size_t>(nnz_declared));
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream ss(line);
        MtxEntry e;
        if (!(ss >> e.row >> e.col >> e.val)) continue;
        R.entries.push_back(e);
    }

    R.ok = true;
    return R;
}

struct BuildResult {
    CSRPattern P;
    CSRMatrix  A;
};

/* Returns a heap-allocated BuildResult.
 * Caller is responsible for delete.
 * Returning by value is impossible because TArray has a deleted copy constructor
 * and TArray inside CSRPattern/CSRMatrix also deletes the implicit move. */
static BuildResult *build_csr(int n, const std::vector<MtxEntry> &entries)
{
    struct E { uint32_t row, col; double val; };

    std::vector<E> lower;
    lower.reserve(entries.size());

    for (const auto &e : entries) {
        uint32_t r = static_cast<uint32_t>(e.row - 1);
        uint32_t c = static_cast<uint32_t>(e.col - 1);
        /* Keep only the lower triangle (row >= col) */
        if (r >= c)
            lower.push_back({r, c, e.val});
        else
            lower.push_back({c, r, e.val});  /* mirror upper → lower */
    }

    /* Sort: by row, then off-diagonal before diagonal, then by col */
    std::sort(lower.begin(), lower.end(), [](const E &a, const E &b) {
        if (a.row != b.row) return a.row < b.row;
        bool ad = (a.row == a.col), bd = (b.row == b.col);
        if (ad != bd) return bd; /* diagonal goes last */
        return a.col < b.col;
    });

    size_t total_nnz = lower.size();

    BuildResult *BR = new BuildResult();
    CSRPattern  &P  = BR->P;
    CSRMatrix   &A  = BR->A;

    P.symmetric = true;
    P.rows = P.cols = static_cast<size_t>(n);
    P.nnz  = total_nnz;
    P.row_start.resize(static_cast<size_t>(n) + 1);
    P.col.resize(total_nnz);

    A.symmetric = true;
    A.rows = A.cols = static_cast<size_t>(n);
    A.nnz  = total_nnz;
    A.data.resize(total_nnz);

    size_t k = 0;
    for (int i = 0; i < n; ++i) {
        P.row_start[i] = static_cast<uint32_t>(k);
        for (; k < total_nnz && lower[k].row == static_cast<uint32_t>(i); ++k) {
            P.col[k]  = lower[k].col;
            A.data[k] = lower[k].val;
        }
    }
    P.row_start[n] = static_cast<uint32_t>(k);

    A.row_start = P.row_start.data;
    A.col       = P.col.data;

    return BR;
}

/* =========================================================================
 * Residual  ||Ax - b|| / ||b||  (full symmetric matvec)
 * ====================================================================== */
static double residual(const CSRMatrix           &A,
                        const std::vector<double> &x,
                        const std::vector<double> &b)
{
    size_t n = A.rows;
    std::vector<double> Ax(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        for (uint32_t k = A.row_start[i]; k < A.row_start[i + 1]; ++k) {
            uint32_t j = A.col[k];
            Ax[i] += A.data[k] * x[j];
            if (j != i)
                Ax[j] += A.data[k] * x[i];
        }
    }

    double r2 = 0.0, b2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double r = Ax[i] - b[i];
        r2 += r * r;
        b2 += b[i] * b[i];
    }
    return (b2 > 0.0) ? std::sqrt(r2 / b2) : std::sqrt(r2);
}

/* =========================================================================
 * SPD check: all diagonal elements > 0, and matrix is diagonally dominant
 * (heuristic — a full eigenvalue test is too expensive here).
 * The real guard is the ASSERT inside factorize() which will throw if a
 * pivot goes non-positive.
 * ====================================================================== */
static bool heuristic_spd(const CSRMatrix *A)
{
    for (size_t i = 0; i < A->rows; ++i) {
        /* Diagonal is the last entry in each row (by our sort order) */
        uint32_t diag_idx = A->row_start[i + 1] - 1;
        if (A->col[diag_idx] != static_cast<uint32_t>(i)) return false; /* no diag */
        if (A->data[diag_idx] <= 0.0)                     return false;
    }
    return true;
}

static std::vector<std::string> read_config(const std::string &path)
{
    std::vector<std::string> result;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open config file: %s\n", path.c_str());
        return result;
    }
    std::string line;
    while (std::getline(f, line)) {
        /* Strip leading whitespace */
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        if (line[0] == '#') continue; /* comment */
        /* Strip trailing whitespace */
        size_t end = line.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) line = line.substr(0, end + 1);
        if (!line.empty()) result.push_back(line);
    }
    return result;
}

struct TestResult {
    std::string matrix;
    int    n            = 0;
    int    nnz          = 0;
    bool   spd          = false;
    double factorize_ms = 0.0;
    double solve_ms     = 0.0;
    double rel_residual = 0.0;
    std::string status; /* PASS | FAIL_* | SKIP_* | ERR_* */
};


static void write_csv(const std::string             &path,
                       const std::vector<TestResult> &results)
{
    FILE *f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Cannot write CSV to: %s\n", path.c_str());
        return;
    }
    fprintf(f, "matrix,n,nnz,spd,factorize_ms,solve_ms,residual,status\n");
    for (const auto &r : results) {
        fprintf(f, "%s,%d,%d,%s,%.4f,%.4f,%.6e,%s\n",
                r.matrix.c_str(),
                r.n,
                r.nnz,
                r.spd ? "true" : "false",
                r.factorize_ms,
                r.solve_ms,
                r.rel_residual,
                r.status.c_str());
    }
    fclose(f);
}

static void print_summary(const std::vector<TestResult> &results)
{
    int pass = 0, fail = 0, skip = 0, err = 0;
    for (const auto &r : results) {
        if      (r.status == "PASS")              ++pass;
        else if (r.status.substr(0,4) == "SKIP")  ++skip;
        else if (r.status.substr(0,3) == "ERR")   ++err;
        else                                       ++fail;
    }
    printf("\n========== SUMMARY ==========\n");
    printf("  PASS : %d\n", pass);
    printf("  FAIL : %d\n", fail);
    printf("  SKIP : %d\n", skip);
    printf("  ERR  : %d\n", err);
    printf("  TOTAL: %d\n", (int)results.size());
    printf("==============================\n\n");

    /* Print failures/errors explicitly */
    for (const auto &r : results) {
        if (r.status != "PASS" && r.status.substr(0,4) != "SKIP") {
            printf("  [%s] %s  (residual=%.3e)\n",
                   r.status.c_str(), r.matrix.c_str(), r.rel_residual);
        }
    }
}

static TestResult test_matrix(const std::string &group_name)
{
    TestResult R;
    R.matrix = group_name;

    /* Split "GROUP/NAME" */
    auto slash = group_name.find('/');
    if (slash == std::string::npos) {
        R.status = "ERR_BAD_NAME";
        return R;
    }
    std::string group = group_name.substr(0, slash);
    std::string name  = group_name.substr(slash + 1);

    printf("[%s] Downloading ...\n", group_name.c_str());

    std::string url =
        "https://sparse.tamu.edu/MM/" + group + "/" + name + ".tar.gz";

    std::vector<char> tar_buf;
    DownloadStatus ds = download(url, tar_buf);
    if (ds == DownloadStatus::CURL_FAIL) {
        printf("[%s] ERR_DOWNLOAD (curl)\n", group_name.c_str());
        R.status = "ERR_DOWNLOAD";
        return R;
    }
    if (ds == DownloadStatus::HTTP_FAIL) {
        printf("[%s] ERR_DOWNLOAD (HTTP != 200)\n", group_name.c_str());
        R.status = "ERR_DOWNLOAD";
        return R;
    }

    std::string mtx_contents;
    if (!extract_mtx(tar_buf, mtx_contents)) {
        printf("[%s] ERR_PARSE (no .mtx in archive)\n", group_name.c_str());
        R.status = "ERR_PARSE";
        return R;
    }

    ParseResult PR = parse_mtx(mtx_contents);
    if (!PR.ok) {
        printf("[%s] ERR_PARSE (bad .mtx header)\n", group_name.c_str());
        R.status = "ERR_PARSE";
        return R;
    }

    if (PR.rows != PR.cols) {
        printf("[%s] SKIP_NOT_SQUARE (%dx%d)\n",
               group_name.c_str(), PR.rows, PR.cols);
        R.n      = PR.rows;
        R.status = "SKIP_NOT_SQUARE";
        return R;
    }

    if (!PR.is_symmetric) {
        printf("[%s] SKIP_NOT_SYMMETRIC\n", group_name.c_str());
        R.n      = PR.rows;
        R.status = "SKIP_NOT_SYMMETRIC";
        return R;
    }

    BuildResult *BR = build_csr(PR.rows, PR.entries);

    R.n   = static_cast<int>(BR->A.rows);
    R.nnz = static_cast<int>(BR->A.nnz);

    if (!heuristic_spd(&BR->A)) {
        printf("[%s] SKIP_NOT_SPD (heuristic check failed)\n",
               group_name.c_str());
        R.status = "SKIP_NOT_SPD";
        delete BR;
        return R;
    }
    R.spd = true;

    printf("[%s] n=%d  nnz=%d  Factorizing ...\n",
           group_name.c_str(), R.n, R.nnz);

    /* ---------- Factorize ---------- */
    std::vector<double> x(static_cast<size_t>(R.n), 0.0);
    std::vector<double> b(static_cast<size_t>(R.n), 1.0);

    try {
        SparseCholeskySolver solver(&BR->A);

        auto t0 = Clock::now();
        solver.initialize(&BR->A);
        R.factorize_ms = elapsed_ms(t0);

        printf("[%s] Factorize: %.2f ms  Solving ...\n",
               group_name.c_str(), R.factorize_ms);

        /* ---------- Solve ---------- */
        auto t1 = Clock::now();
        solver.solve(x.data(), b.data());
        R.solve_ms = elapsed_ms(t1);

    } catch (...) {
        printf("[%s] ERR_SOLVER (exception during factorize/solve)\n",
               group_name.c_str());
        R.status = "ERR_SOLVER";
        delete BR;
        return R;
    }

    R.rel_residual = residual(BR->A, x, b);
    delete BR;

    const double TOL = 1e-6;
    if (R.rel_residual <= TOL) {
        printf("[%s] PASS  residual=%.3e  solve=%.2f ms\n",
               group_name.c_str(), R.rel_residual, R.solve_ms);
        R.status = "PASS";
    } else {
        printf("[%s] FAIL_RESIDUAL  residual=%.3e  (tol=%.0e)\n",
               group_name.c_str(), R.rel_residual, TOL);
        R.status = "FAIL_RESIDUAL";
    }

    return R;
}

int main(int argc, char **argv)
{
    const char *cfg_path = (argc > 1) ? argv[1] : "matrices.cfg";
    const char *csv_path = (argc > 2) ? argv[2] : "results.csv";

    std::vector<std::string> matrices = read_config(cfg_path);
    if (matrices.empty()) {
        fprintf(stderr, "No matrices found in config file: %s\n", cfg_path);
        return 1;
    }

    printf("Testing %zu matrices from '%s'\n\n", matrices.size(), cfg_path);

    std::vector<TestResult> results;
    results.reserve(matrices.size());

    for (const auto &m : matrices)
        results.push_back(test_matrix(m));

    print_summary(results);
    write_csv(csv_path, results);
    printf("Results written to: %s\n", csv_path);

    return 0;
}