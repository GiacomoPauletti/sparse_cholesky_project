#pragma once

#include <stddef.h>

#include "array.h"
#include "linear_solver.h"
#include "sparse_matrix.h"

/* Conjugate gradient wrapped as a LinearSolver, so that it can be swapped at
 * runtime with SparseCholeskySolver.
 *
 * The algorithm itself lives in conjugate_gradient.h and is not duplicated
 * here : this class only owns the three work vectors (r, p and Ap) that the
 * free function expects the caller to provide, together with the stopping
 * criterion, and it resizes them when the matrix changes.
 *
 * A is required to be symmetric positive definite. Note that CSRMatrix::mvp
 * handles the symmetric (lower triangular) storage, so a matrix assembled with
 * symmetric = true works as is.
 */
struct CGSolver : public LinearSolver {
	/* The stopping criterion is fixed at construction : stop as soon as
	 * |b - A x| / |b| <= tol, and in any case after iter_max iterations.
	 * It belongs to the constructor rather than to the LinearSolver
	 * interface, since a direct method has no such notion. */
	/* profiler is optional : nullptr (the default) disables the
	 * instrumentation of this solver. */
	CGSolver(double tol = 1e-6, int iter_max = 500,
		 Profiler *profiler = nullptr);
	CGSolver(CSRMatrix *A, double tol = 1e-6, int iter_max = 500,
		 Profiler *profiler = nullptr);

	/* Cheap : only records A and sizes the work vectors. Contrary to the
	 * Cholesky case there is nothing to recompute when the values of A
	 * change, but calling it again is harmless. */
	void initialize(CSRMatrix *A) override;

	/* x is used as the initial iterate (warm start) and holds the solution
	 * on return. */
	void solve(double *__restrict x, const double *__restrict b) override;

	size_t iterations() const override { return last_iter; }

	const double tol;
	const int iter_max;

	/* Diagnostics of the last call to solve(), handy to display in a GUI. */
	size_t last_iter = 0;
	double last_error = 0;

    private:
	CSRMatrix *A = nullptr;
	TArray<double> r;  /* residual  b - A * x */
	TArray<double> p;  /* direction of search */
	TArray<double> Ap; /* product A * p       */
};
