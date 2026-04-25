/**
 * suite_sparse_analysis.cpp
 *
 * For each SPD matrix in a config file, performs two analyses and writes
 * two separate CSV files:
 *
 *  1. fill_in.csv  — Fill-in analysis
 *       Columns: matrix, n, nnz_A, nnz_L_worstcase, nnz_L_symbolic,
 *                nnz_L_actual, fillin_ratio_vs_worst, fillin_ratio_vs_A,
 *                symbolic_ms, factorize_ms
 *
 *       Definitions:
 *         nnz_A            : nonzeros in lower triangle of A (stored)
 *         nnz_L_worstcase  : n*(n+1)/2  (dense lower triangle)
 *         nnz_L_symbolic   : nnz predicted by buildPatterns() — exact for
 *                            Cholesky (no over-estimate)
 *         nnz_L_actual     : nnz in the numeric factor L (== symbolic for
 *                            Cholesky; included to verify)
 *         fillin_ratio_vs_worst : nnz_L_symbolic / nnz_L_worstcase
 *                            (how much the ordering saves vs dense)
 *         fillin_ratio_vs_A    : nnz_L_symbolic / nnz_A
 *                            (fill-in factor relative to original sparsity)
 *
 *  2. cg_vs_cholesky.csv  — Iterative vs direct solver comparison
 *       Columns: matrix, n, nnz_A, chol_factorize_ms, chol_solve_ms,
 *                chol_total_ms, chol_residual,
 *                cg_solve_ms, cg_iterations, cg_residual,
 *                speedup_chol_vs_cg   (chol_total_ms / cg_solve_ms)
 *
 *       CG is run unpreconditioned (no preconditioner in the repo) with
 *       tol=1e-6 and max_iter=10*n.  Both solvers use b=[1,1,...,1].
 *
 * Usage:
 *   ./suite_sparse_analysis [matrices.cfg] [fill_in.csv] [cg_vs_cholesky.csv]
 *
 * Defaults:
 *   matrices.cfg, fill_in.csv, cg_vs_cholesky.csv
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
#include "conjugate_gradient.h"
#include "sparse_matrix.h"

/* =========================================================================
 * Timing
 * ====================================================================== */
using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double elapsed_ms(Clock::time_point t0)
{
    return Ms(Clock::now() - t0).count();
}

/* =========================================================================
 * HTTP download
 * ====================================================================== */
static size_t write_to_buffer(void *ptr, size_t size, size_t nmemb,
                               void *userdata)
{
    auto *buf = static_cast<std::vector<char> *>(userdata);
    const char *bytes = static_cast<const char *>(ptr);
    buf->insert(buf->end(), bytes, bytes + size * nmemb);
    return size * nmemb;
}

enum class DLStatus { OK, CURL_FAIL, HTTP_FAIL };

static DLStatus download(const std::string &url, std::vector<char> &buf)
{
    curl_global_init(CURL_GLOBAL_ALL);
    CURL *h = curl_easy_init();
    if (!h) { curl_global_cleanup(); return DLStatus::CURL_FAIL; }

    curl_easy_setopt(h, CURLOPT_URL,            url.c_str());
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION,  write_to_buffer);
    curl_easy_setopt(h, CURLOPT_WRITEDATA,      &buf);
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(h, CURLOPT_TIMEOUT,        120L);

    CURLcode rc = curl_easy_perform(h);
    long http_code = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(h);
    curl_global_cleanup();

    if (rc != CURLE_OK)   return DLStatus::CURL_FAIL;
    if (http_code != 200) return DLStatus::HTTP_FAIL;
    return DLStatus::OK;
}

/* =========================================================================
 * .tar.gz → first .mtx
 * ====================================================================== */
static bool extract_mtx(const std::vector<char> &tar_buf,
                         std::string             &out)
{
    struct archive *a = archive_read_new();
    archive_read_support_filter_gzip(a);
    archive_read_support_format_tar(a);
    if (archive_read_open_memory(a, tar_buf.data(), tar_buf.size()) != ARCHIVE_OK) {
        archive_read_free(a); return false;
    }
    struct archive_entry *entry;
    while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
        const char *name = archive_entry_pathname(entry);
        const char *ext  = strrchr(name, '.');
        if (!ext || strcmp(ext, ".mtx") != 0) { archive_read_data_skip(a); continue; }
        char chunk[8192]; la_ssize_t n;
        while ((n = archive_read_data(a, chunk, sizeof(chunk))) > 0)
            out.append(chunk, static_cast<size_t>(n));
        break;
    }
    archive_read_free(a);
    return !out.empty();
}

/* =========================================================================
 * Matrix Market parser
 * ====================================================================== */
struct MtxEntry { int row, col; double val; };

struct ParseResult {
    bool ok           = false;
    bool is_symmetric = false;
    int  rows         = 0;
    int  cols         = 0;
    std::vector<MtxEntry> entries;
};

static ParseResult parse_mtx(const std::string &mtx)
{
    ParseResult R;
    std::istringstream stream(mtx);
    std::string line;
    if (!std::getline(stream, line)) return R;
    std::string hdr = line;
    for (auto &c : hdr) c = static_cast<char>(tolower(c));
    if (hdr.find("matrixmarket") == std::string::npos) return R;
    R.is_symmetric = (hdr.find("symmetric") != std::string::npos);
    while (std::getline(stream, line))
        if (!line.empty() && line[0] != '%') break;
    int nnz_decl = 0;
    { std::istringstream ss(line); if (!(ss >> R.rows >> R.cols >> nnz_decl)) return R; }
    R.entries.reserve(static_cast<size_t>(nnz_decl));
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream ss(line); MtxEntry e;
        if (!(ss >> e.row >> e.col >> e.val)) continue;
        R.entries.push_back(e);
    }
    R.ok = true;
    return R;
}

/* =========================================================================
 * Build CSRPattern + CSRMatrix  (heap-allocated to avoid TArray copy issues)
 * ====================================================================== */
struct BuildResult { CSRPattern P; CSRMatrix A; };

static BuildResult *build_csr(int n, const std::vector<MtxEntry> &entries)
{
    struct E { uint32_t row, col; double val; };
    std::vector<E> lower;
    lower.reserve(entries.size());
    for (const auto &e : entries) {
        uint32_t r = static_cast<uint32_t>(e.row - 1);
        uint32_t c = static_cast<uint32_t>(e.col - 1);
        if (r >= c) lower.push_back({r, c, e.val});
        else        lower.push_back({c, r, e.val});
    }
    std::sort(lower.begin(), lower.end(), [](const E &a, const E &b) {
        if (a.row != b.row) return a.row < b.row;
        bool ad = (a.row == a.col), bd = (b.row == b.col);
        if (ad != bd) return bd;
        return a.col < b.col;
    });
    size_t total_nnz = lower.size();
    BuildResult *BR  = new BuildResult();
    CSRPattern  &P   = BR->P;
    CSRMatrix   &A   = BR->A;
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
            P.col[k] = lower[k].col; A.data[k] = lower[k].val;
        }
    }
    P.row_start[n] = static_cast<uint32_t>(k);
    A.row_start = P.row_start.data;
    A.col       = P.col.data;
    return BR;
}

/* =========================================================================
 * Heuristic SPD check (positive diagonal)
 * ====================================================================== */
static bool heuristic_spd(const CSRMatrix *A)
{
    for (size_t i = 0; i < A->rows; ++i) {
        uint32_t diag_idx = A->row_start[i + 1] - 1;
        if (A->col[diag_idx] != static_cast<uint32_t>(i)) return false;
        if (A->data[diag_idx] <= 0.0)                     return false;
    }
    return true;
}

/* =========================================================================
 * Residual  ||Ax - b|| / ||b||  (full symmetric matvec)
 * ====================================================================== */
static double compute_residual(const CSRMatrix           &A,
                                const std::vector<double> &x,
                                const std::vector<double> &b)
{
    size_t n = A.rows;
    std::vector<double> Ax(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (uint32_t k = A.row_start[i]; k < A.row_start[i + 1]; ++k) {
            uint32_t j = A.col[k];
            Ax[i] += A.data[k] * x[j];
            if (j != i) Ax[j] += A.data[k] * x[i];
        }
    }
    double r2 = 0.0, b2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double r = Ax[i] - b[i]; r2 += r * r; b2 += b[i] * b[i];
    }
    return (b2 > 0.0) ? std::sqrt(r2 / b2) : std::sqrt(r2);
}

/* =========================================================================
 * Config reader
 * ====================================================================== */
static std::vector<std::string> read_config(const std::string &path)
{
    std::vector<std::string> result;
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open config: %s\n", path.c_str()); return result; }
    std::string line;
    while (std::getline(f, line)) {
        size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos) continue;
        line = line.substr(s);
        if (line[0] == '#') continue;
        size_t e = line.find_last_not_of(" \t\r\n");
        if (e != std::string::npos) line = line.substr(0, e + 1);
        if (!line.empty()) result.push_back(line);
    }
    return result;
}

/* =========================================================================
 * Result records
 * ====================================================================== */
struct FillResult {
    std::string matrix;
    int    n                    = 0;
    long   nnz_A                = 0;
    long   nnz_L_worstcase      = 0;  /* n*(n+1)/2  */
    long   nnz_L_symbolic       = 0;  /* from buildPatterns() */
    long   nnz_L_actual         = 0;  /* from factorize()     */
    double fillin_ratio_vs_worst= 0.0;/* nnz_L_sym / worst    */
    double fillin_ratio_vs_A    = 0.0;/* nnz_L_sym / nnz_A    */
    double symbolic_ms          = 0.0;
    double factorize_ms         = 0.0;
    std::string status;
};

struct CGResult {
    std::string matrix;
    int    n                  = 0;
    long   nnz_A              = 0;
    double chol_factorize_ms  = 0.0;
    double chol_solve_ms      = 0.0;
    double chol_total_ms      = 0.0;
    double chol_residual      = 0.0;
    double cg_solve_ms        = 0.0;
    int    cg_iterations      = 0;
    double cg_residual        = 0.0;
    double speedup_chol_vs_cg = 0.0; /* chol_total / cg_solve */
    std::string status;
};

/* =========================================================================
 * Write CSVs
 * ====================================================================== */
static void write_fill_csv(const std::string              &path,
                            const std::vector<FillResult>  &results)
{
    FILE *f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); return; }
    fprintf(f, "matrix,n,nnz_A,nnz_L_worstcase,nnz_L_symbolic,nnz_L_actual,"
               "fillin_ratio_vs_worst,fillin_ratio_vs_A,symbolic_ms,factorize_ms,status\n");
    for (const auto &r : results) {
        fprintf(f, "%s,%d,%ld,%ld,%ld,%ld,%.6f,%.6f,%.4f,%.4f,%s\n",
                r.matrix.c_str(), r.n,
                r.nnz_A, r.nnz_L_worstcase, r.nnz_L_symbolic, r.nnz_L_actual,
                r.fillin_ratio_vs_worst, r.fillin_ratio_vs_A,
                r.symbolic_ms, r.factorize_ms,
                r.status.c_str());
    }
    fclose(f);
    printf("Fill-in CSV written to: %s\n", path.c_str());
}

static void write_cg_csv(const std::string             &path,
                          const std::vector<CGResult>   &results)
{
    FILE *f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); return; }
    fprintf(f, "matrix,n,nnz_A,chol_factorize_ms,chol_solve_ms,chol_total_ms,"
               "chol_residual,cg_solve_ms,cg_iterations,cg_residual,speedup_chol_vs_cg,status\n");
    for (const auto &r : results) {
        fprintf(f, "%s,%d,%ld,%.4f,%.4f,%.4f,%.6e,%.4f,%d,%.6e,%.4f,%s\n",
                r.matrix.c_str(), r.n, r.nnz_A,
                r.chol_factorize_ms, r.chol_solve_ms, r.chol_total_ms,
                r.chol_residual,
                r.cg_solve_ms, r.cg_iterations, r.cg_residual,
                r.speedup_chol_vs_cg,
                r.status.c_str());
    }
    fclose(f);
    printf("CG vs Cholesky CSV written to: %s\n", path.c_str());
}

/* =========================================================================
 * Core per-matrix analysis
 * ====================================================================== */
struct AnalysisResult { FillResult fill; CGResult cg; };

static AnalysisResult analyse_matrix(const std::string &group_name)
{
    AnalysisResult AR;
    AR.fill.matrix = AR.cg.matrix = group_name;

    /* ---- Download ---- */
    auto slash = group_name.find('/');
    if (slash == std::string::npos) {
        AR.fill.status = AR.cg.status = "ERR_BAD_NAME";
        return AR;
    }
    std::string group = group_name.substr(0, slash);
    std::string name  = group_name.substr(slash + 1);
    std::string url   = "https://sparse.tamu.edu/MM/" + group + "/" + name + ".tar.gz";

    printf("[%s] Downloading ...\n", group_name.c_str());
    std::vector<char> tar_buf;
    auto dl = download(url, tar_buf);
    if (dl != DLStatus::OK) {
        AR.fill.status = AR.cg.status = "ERR_DOWNLOAD";
        return AR;
    }

    /* ---- Extract & parse ---- */
    std::string mtx_contents;
    if (!extract_mtx(tar_buf, mtx_contents)) {
        AR.fill.status = AR.cg.status = "ERR_PARSE";
        return AR;
    }
    ParseResult PR = parse_mtx(mtx_contents);
    if (!PR.ok || PR.rows != PR.cols || !PR.is_symmetric) {
        AR.fill.status = AR.cg.status = "SKIP";
        return AR;
    }

    /* ---- Build CSR ---- */
    BuildResult *BR = build_csr(PR.rows, PR.entries);
    int  n   = static_cast<int>(BR->A.rows);
    long nnzA = static_cast<long>(BR->A.nnz);

    AR.fill.n    = AR.cg.n    = n;
    AR.fill.nnz_A = AR.cg.nnz_A = nnzA;

    if (!heuristic_spd(&BR->A)) {
        AR.fill.status = AR.cg.status = "SKIP_NOT_SPD";
        delete BR; return AR;
    }

    printf("[%s] n=%d  nnz_A=%ld\n", group_name.c_str(), n, nnzA);

    /* ==================================================================
     * ANALYSIS 1 : FILL-IN
     * ================================================================== */
    {
        /* worst case: dense lower triangle */
        long worst = (static_cast<long>(n) * (n + 1)) / 2;
        AR.fill.nnz_L_worstcase = worst;

        /* Symbolic phase: get predicted pattern */
        SparseCholeskySymbolic symbolic(&BR->A);
        CSRPattern *patL   = new CSRPattern();
        CSRPattern *patL_T = new CSRPattern();

        auto t_sym0 = Clock::now();
        symbolic.buildTree();
        symbolic.buildPatterns(patL, patL_T);
        AR.fill.symbolic_ms = elapsed_ms(t_sym0);

        long nnz_sym = static_cast<long>(patL->nnz);
        AR.fill.nnz_L_symbolic = nnz_sym;

        /* Numeric factorization */
        SparseCholeskyFactorization factorization(&BR->A);
        factorization.setPatternL(patL);

        CSRMatrix *factor = nullptr;
        bool factor_ok = true;
        auto t_fac0 = Clock::now();
        try {
            factor = factorization.factorize();
        } catch (...) {
            factor_ok = false;
        }
        AR.fill.factorize_ms = elapsed_ms(t_fac0);

        if (!factor_ok || !factor) {
            AR.fill.status = "ERR_SOLVER";
        } else {
            AR.fill.nnz_L_actual        = static_cast<long>(factor->nnz);
            AR.fill.fillin_ratio_vs_worst = static_cast<double>(nnz_sym) / worst;
            AR.fill.fillin_ratio_vs_A    = static_cast<double>(nnz_sym) / nnzA;
            AR.fill.status               = "OK";

            printf("[%s] Fill-in: worst=%ld  sym=%ld  actual=%ld"
                   "  ratio_vs_worst=%.4f  ratio_vs_A=%.2f\n",
                   group_name.c_str(), worst, nnz_sym,
                   AR.fill.nnz_L_actual,
                   AR.fill.fillin_ratio_vs_worst,
                   AR.fill.fillin_ratio_vs_A);
        }

        /* Cleanup factor; patL/patL_T owned by symbolic, don't double-free */
        delete factor;
        delete patL;
        delete patL_T;
    }

    /* ==================================================================
     * ANALYSIS 2 : CG vs CHOLESKY
     * ================================================================== */
    {
        const double TOL      = 1e-6;
        const int    MAX_ITER = 10 * n;

        std::vector<double> b(static_cast<size_t>(n), 1.0);

        /* ---- Cholesky ---- */
        {
            std::vector<double> x(static_cast<size_t>(n), 0.0);
            bool ok = true;

            SparseCholeskySolver solver(&BR->A);
            auto t0 = Clock::now();
            try { solver.initialize(&BR->A); }
            catch (...) { ok = false; }
            AR.cg.chol_factorize_ms = elapsed_ms(t0);

            if (ok) {
                auto t1 = Clock::now();
                try { solver.solve(x.data(), b.data()); }
                catch (...) { ok = false; }
                AR.cg.chol_solve_ms = elapsed_ms(t1);
            }

            AR.cg.chol_total_ms = AR.cg.chol_factorize_ms + AR.cg.chol_solve_ms;

            if (ok) {
                AR.cg.chol_residual = compute_residual(BR->A, x, b);
                printf("[%s] Cholesky: factorize=%.2f ms  solve=%.2f ms"
                       "  residual=%.3e\n",
                       group_name.c_str(),
                       AR.cg.chol_factorize_ms, AR.cg.chol_solve_ms,
                       AR.cg.chol_residual);
            } else {
                AR.cg.status = "ERR_CHOL";
            }
        }

        /* ---- CG (unpreconditioned) ---- */
        if (AR.cg.status.empty()) {
            std::vector<double> x_cg(static_cast<size_t>(n), 0.0);
            std::vector<double> r_cg(static_cast<size_t>(n));
            std::vector<double> p_cg(static_cast<size_t>(n));
            std::vector<double> Ap_cg(static_cast<size_t>(n));

            /* Hard caps:
             *   - at most 5000 iterations regardless of n
             *   - at most 3x Cholesky-total wall-clock time (min 500 ms)
             * Status column reports CG_TIMEOUT or CG_MAXITER if a limit was
             * hit without reaching tol, so the comparison is still meaningful. */
            const int    CG_MAX_ITER   = std::min(5000, 10 * n);
            const double CG_TIMEOUT_MS = 3.0 * AR.cg.chol_total_ms + 500.0;

            /* Init: x=0 => r=b, p=b */
            for (int i = 0; i < n; ++i) r_cg[i] = b[i];
            for (int i = 0; i < n; ++i) p_cg[i] = b[i];

            double b2 = 0.0;
            for (int i = 0; i < n; ++i) b2 += b[i] * b[i];
            double r2 = b2; /* r=b initially */
            double rel_error = 1.0;

            bool timed_out   = false;
            bool iter_capped = false;
            int  iters       = 0;
            auto t_cg        = Clock::now();

            while (iters < CG_MAX_ITER && rel_error > TOL) {
                r2 = cg_iterate_once(BR->A,
                                     x_cg.data(), r_cg.data(),
                                     p_cg.data(), Ap_cg.data(), r2);
                rel_error = (b2 > 0.0) ? std::sqrt(r2 / b2) : std::sqrt(r2);
                ++iters;
                /* Check wall-clock every 50 iters to keep overhead low */
                if ((iters % 50) == 0 && elapsed_ms(t_cg) >= CG_TIMEOUT_MS) {
                    timed_out = true;
                    break;
                }
            }
            if (!timed_out && iters == CG_MAX_ITER && rel_error > TOL)
                iter_capped = true;

            AR.cg.cg_solve_ms   = elapsed_ms(t_cg);
            AR.cg.cg_iterations = iters;
            AR.cg.cg_residual   = compute_residual(BR->A, x_cg, b);
            AR.cg.speedup_chol_vs_cg =
                (AR.cg.cg_solve_ms > 0.0)
                    ? AR.cg.chol_total_ms / AR.cg.cg_solve_ms
                    : 0.0;

            if      (timed_out)   AR.cg.status = "CG_TIMEOUT";
            else if (iter_capped) AR.cg.status = "CG_MAXITER";
            else                  AR.cg.status = "OK";

            printf("[%s] CG: solve=%.2f ms  iters=%d  residual=%.3e"
                   "  speedup(chol_total/cg)=%.3f  [%s]\n",
                   group_name.c_str(),
                   AR.cg.cg_solve_ms, AR.cg.cg_iterations,
                   AR.cg.cg_residual, AR.cg.speedup_chol_vs_cg,
                   AR.cg.status.c_str());
        }
    }

    delete BR;
    return AR;
}

/* =========================================================================
 * main
 * ====================================================================== */
int main(int argc, char **argv)
{
    const char *cfg_path      = (argc > 1) ? argv[1] : "matrices.cfg";
    const char *fill_csv_path = (argc > 2) ? argv[2] : "fill_in.csv";
    const char *cg_csv_path   = (argc > 3) ? argv[3] : "cg_vs_cholesky.csv";

    std::vector<std::string> matrices = read_config(cfg_path);
    if (matrices.empty()) {
        fprintf(stderr, "No matrices in config: %s\n", cfg_path);
        return 1;
    }

    printf("Analysing %zu matrices from '%s'\n\n", matrices.size(), cfg_path);

    std::vector<FillResult> fill_results;
    std::vector<CGResult>   cg_results;
    fill_results.reserve(matrices.size());
    cg_results.reserve(matrices.size());

    for (const auto &m : matrices) {
        AnalysisResult AR = analyse_matrix(m);
        fill_results.push_back(AR.fill);
        cg_results.push_back(AR.cg);
        printf("\n");
    }

    /* ---- Summary ---- */
    printf("========== SUMMARY ==========\n");
    int ok_fill = 0, ok_cg = 0, skipped = 0, err = 0;
    for (size_t i = 0; i < fill_results.size(); ++i) {
        if (fill_results[i].status == "OK") ++ok_fill;
        if (cg_results[i].status   == "OK") ++ok_cg;
        if (fill_results[i].status.substr(0,4) == "SKIP") ++skipped;
        if (fill_results[i].status.substr(0,3) == "ERR")  ++err;
    }
    printf("  Fill-in OK      : %d\n", ok_fill);
    printf("  CG vs Chol OK   : %d\n", ok_cg);
    printf("  Skipped         : %d\n", skipped);
    printf("  Errors          : %d\n", err);
    printf("  Total matrices  : %zu\n", matrices.size());
    printf("=============================\n\n");

    write_fill_csv(fill_csv_path, fill_results);
    write_cg_csv(cg_csv_path, cg_results);

    return 0;
}