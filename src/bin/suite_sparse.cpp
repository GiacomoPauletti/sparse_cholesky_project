#include <archive.h>
#include <archive_entry.h>
#include <curl/curl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "cholesky.h"
#include "sparse_matrix.h"

/* -------------------------------------------------------------------------
 * libcurl write callback
 * ---------------------------------------------------------------------- */
static size_t write_to_buffer(void *ptr, size_t size, size_t nmemb, void *userdata)
{
    auto *buf = (std::vector<char> *)userdata;
    const char *bytes = (const char *)ptr;
    buf->insert(buf->end(), bytes, bytes + size * nmemb);
    return size * nmemb;
}

/* -------------------------------------------------------------------------
 * Download a URL into a memory buffer.
 * ---------------------------------------------------------------------- */
static bool download(const std::string &url, std::vector<char> &buf)
{
    printf("Downloading %s ...\n", url.c_str());

    curl_global_init(CURL_GLOBAL_ALL);
    CURL *handle = curl_easy_init();
    if (!handle) {
        fprintf(stderr, "curl_easy_init() failed\n");
        return false;
    }

    curl_easy_setopt(handle, CURLOPT_URL,            url.c_str());
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION,  write_to_buffer);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA,      &buf);
    curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);

    CURLcode rc = curl_easy_perform(handle);

    long http_code = 0;
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(handle);
    curl_global_cleanup();

    if (rc != CURLE_OK) {
        fprintf(stderr, "curl error: %s\n", curl_easy_strerror(rc));
        return false;
    }
    if (http_code != 200) {
        fprintf(stderr, "HTTP error: %ld\n", http_code);
        return false;
    }

    printf("Downloaded %zu bytes\n", buf.size());
    return true;
}

/* -------------------------------------------------------------------------
 * Extract the first .mtx file from the tarball into a string.
 * ---------------------------------------------------------------------- */
static bool extract_mtx(const std::vector<char> &tar_buf, std::string &mtx_contents)
{
    struct archive *a = archive_read_new();
    archive_read_support_filter_gzip(a);
    archive_read_support_format_tar(a);

    if (archive_read_open_memory(a, tar_buf.data(), tar_buf.size()) != ARCHIVE_OK) {
        fprintf(stderr, "archive_read_open_memory failed: %s\n",
                archive_error_string(a));
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
        char chunk[4096];
        la_ssize_t n;
        while ((n = archive_read_data(a, chunk, sizeof(chunk))) > 0)
            mtx_contents.append(chunk, (size_t)n);
        break;
    }

    archive_read_free(a);
    return !mtx_contents.empty();
}

/* -------------------------------------------------------------------------
 * A single matrix entry (1-indexed, as in the file).
 * ---------------------------------------------------------------------- */
struct Entry {
    int    row;
    int    col;
    double val;
};

/* -------------------------------------------------------------------------
 * Parse the full .mtx file.
 * ---------------------------------------------------------------------- */
static bool parse_mtx(const std::string  &mtx,
                       bool               &is_symmetric,
                       int                &rows,
                       int                &cols,
                       int                &nnz,
                       std::vector<Entry> &entries)
{
    std::istringstream stream(mtx);
    std::string line;

    if (!std::getline(stream, line)) return false;
    std::string lline = line;
    for (auto &c : lline) c = (char)tolower(c);
    if (lline.find("matrixmarket") == std::string::npos) return false;
    is_symmetric = (lline.find("symmetric") != std::string::npos);

    while (std::getline(stream, line))
        if (!line.empty() && line[0] != '%') break;

    std::istringstream size_stream(line);
    if (!(size_stream >> rows >> cols >> nnz)) return false;

    entries.reserve((size_t)nnz);
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream ls(line);
        Entry e;
        if (!(ls >> e.row >> e.col >> e.val)) continue;
        entries.push_back(e);
    }

    return true;
}

/* -------------------------------------------------------------------------
 * Build CSRPattern and CSRMatrix from entries.
 * ---------------------------------------------------------------------- */
static void build_csr(int                       n,
                       const std::vector<Entry> &entries,
                       CSRPattern               &P,
                       CSRMatrix                &A)
{
    struct E { uint32_t row, col; double val; };
    std::vector<E> lower;
    lower.reserve(entries.size());

    for (auto &e : entries) {
        uint32_t r = (uint32_t)(e.row - 1);
        uint32_t c = (uint32_t)(e.col - 1);
        if (r >= c) lower.push_back({r, c, e.val});
        else        lower.push_back({c, r, e.val});
    }

    std::sort(lower.begin(), lower.end(), [](const E &a, const E &b) {
        if (a.row != b.row) return a.row < b.row;
        bool a_diag = (a.row == a.col);
        bool b_diag = (b.row == b.col);
        if (a_diag != b_diag) return b_diag;
        return a.col < b.col;
    });

    size_t total_nnz = lower.size();

    P.symmetric = true;
    P.rows = P.cols = (size_t)n;
    P.nnz  = total_nnz;
    P.row_start.resize((size_t)n + 1);
    P.col.resize(total_nnz);

    A.symmetric = true;
    A.rows = A.cols = (size_t)n;
    A.nnz  = total_nnz;
    A.data.resize(total_nnz);

    size_t k = 0;
    for (int i = 0; i < n; i++) {
        P.row_start[i] = (uint32_t)k;
        for (; k < total_nnz && lower[k].row == (uint32_t)i; k++) {
            P.col[k]  = lower[k].col;
            A.data[k] = lower[k].val;
        }
    }
    P.row_start[n] = (uint32_t)k;

    A.row_start = P.row_start.data;
    A.col       = P.col.data;
}

/* -------------------------------------------------------------------------
 * Compute ||Ax - b|| / ||b|| (full symmetric matvec).
 * ---------------------------------------------------------------------- */
static double residual(const CSRMatrix          &A,
                        const std::vector<double> &x,
                        const std::vector<double> &b)
{
    size_t n = A.rows;
    std::vector<double> Ax(n, 0.0);

    for (size_t i = 0; i < n; i++) {
        for (uint32_t k = A.row_start[i]; k < A.row_start[i + 1]; k++) {
            uint32_t j = A.col[k];
            Ax[i] += A.data[k] * x[j];
            if (j != i)
                Ax[j] += A.data[k] * x[i]; /* symmetric contribution */
        }
    }

    double r2 = 0.0, b2 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double r = Ax[i] - b[i];
        r2 += r * r;
        b2 += b[i] * b[i];
    }
    return std::sqrt(r2 / b2);
}

int main()
{
    const std::string group = "HB";
    const std::string name  = "nos4";
    const std::string url   = "https://sparse.tamu.edu/MM/" + group + "/" + name + ".tar.gz";

    std::vector<char> tar_buf;
    if (!download(url, tar_buf)) return 1;

    std::string mtx_contents;
    if (!extract_mtx(tar_buf, mtx_contents)) {
        fprintf(stderr, "Failed to extract .mtx\n");
        return 1;
    }

    bool               is_symmetric;
    int                rows, cols, nnz;
    std::vector<Entry> entries;
    if (!parse_mtx(mtx_contents, is_symmetric, rows, cols, nnz, entries))
        return 1;

    printf("\nFormat : %s  n=%d  nnz=%d\n",
           is_symmetric ? "symmetric" : "general", rows, nnz);

    CSRPattern P;
    CSRMatrix  A;
    build_csr(rows, entries, P, A);


    /* b = [1, 1, ..., 1] */
    std::vector<double> b((size_t)rows, 1.0);
    std::vector<double> x((size_t)rows, 0.0);

    printf("Factorizing...\n");
    SparseCholeskySolver solver(&A);
    solver.initialize(&A);

    printf("Solving...\n");
    solver.solve(x.data(), b.data());

    double rel = residual(A, x, b);
    printf("\n||Ax - b|| / ||b|| = %.6e\n", rel);
    printf("%s\n", rel < 1e-6 ? "PASS" : "FAIL");

    return 0;
}