#include <assert.h>

#include "conjugate_gradient2.h"

#include "conjugate_gradient.h"

CGSolver::CGSolver(double tol, int iter_max, Profiler *profiler)
    : LinearSolver(profiler), tol(tol), iter_max(iter_max)
{
}

CGSolver::CGSolver(CSRMatrix *A, double tol, int iter_max, Profiler *profiler)
    : LinearSolver(profiler), tol(tol), iter_max(iter_max)
{
	initialize(A);
}

void CGSolver::initialize(CSRMatrix *A)
{
	/* Only sizes the work vectors : reported for symmetry with the
	 * Cholesky setup, which it should dwarf. */
	ProfileSection section(profiler, "cg initialize");
	ProfileStep step(profiler, "allocate work vectors");

	assert(A);
	assert(A->rows == A->cols);

	this->A = A;
	r.resize(A->rows);
	p.resize(A->rows);
	Ap.resize(A->rows);
}

void CGSolver::solve(double *__restrict x, const double *__restrict b)
{
	/* The iteration loop is the only meaningful measurement of an
	 * iterative method. Divide the total by iterations() to get the cost
	 * of one mvp + a few blas level 1 passes. */
	ProfileSection section(profiler, "cg solve");
	ProfileStep step(profiler, "iterations");

	assert(A);

	/* inited = false : the free function forms r = b - A * x and p = r from
	 * the incoming x, which is exactly the warm start we want. */
	last_iter = conjugate_gradient_solve(*A, b, x, r.data, p.data, Ap.data,
					     &last_error, tol, iter_max,
					     /* inited = */ false);
}
