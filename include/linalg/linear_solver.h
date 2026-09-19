#pragma once

#include <stddef.h>

#include "profiler.h"
#include "sparse_matrix.h"

/* Common interface for the solvers of a SPD linear system A * x = b.
 *
 * Two implementations are provided :
 *   - SparseCholeskySolver (see cholesky.h) : direct method. The expensive
 *     work happens once, in initialize(), which factorizes A = L * L^T ; every
 *     subsequent solve() is then only a pair of triangular substitutions.
 *   - CGSolver (see conjugate_gradient2.h)  : iterative method. initialize()
 *     is essentially free, but each solve() loops until the relative residual
 *     drops below a tolerance.
 *
 * The contract is the same for both :
 *   - initialize(A) must be called before the first solve(), and again
 *     whenever the *values* of A change (the solvers keep a pointer to A, they
 *     do not own it, so A must outlive the solver).
 *   - solve(x, b) writes the solution in x. On entry x may hold an initial
 *     guess : CGSolver uses it as a starting iterate (a warm start from the
 *     previous time step typically saves a good fraction of the iterations),
 *     SparseCholeskySolver simply overwrites it.
 */
struct LinearSolver {
	/* The profiler is optional : a null pointer (the default) means this
	 * solver is not instrumented, and every profiling call it makes then
	 * costs a single null test. The solver does not own it, so it must
	 * outlive the solver. */
	explicit LinearSolver(Profiler *profiler = nullptr)
	    : profiler(profiler)
	{
	}

	virtual void initialize(CSRMatrix *A) = 0;
	virtual void solve(double *__restrict x, const double *__restrict b) = 0;

	/* Number of iterations the last solve() took. A direct method solves in
	 * one shot, hence the default ; iterative methods override it. */
	virtual size_t iterations() const { return 1; }

	virtual ~LinearSolver();

	Profiler *profiler = nullptr;
};
