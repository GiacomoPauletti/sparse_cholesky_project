#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "navier_stokes.h"

#include "P1.h"
#include "tiny_blas.h"

#if USE_FEM_MATRIX
NavierStokesSolver::NavierStokesSolver(const Mesh &m, Profiler *profiler)
#else
NavierStokesSolver::NavierStokesSolver(const Mesh &m, Profiler *profiler,
				       LinearSolverKind backend, double tol,
				       size_t iter_max)
#endif
	: m(m)
	, N(m.vertex_count())
	, omega(N)
	, Momega(N)
	, psi(N)
	, r(N)
	, p(N)
	, Ap(N)
{
	this->profiler = profiler;
#if !USE_FEM_MATRIX
	this->backend = backend;
	this->tol = tol;
	this->iter_max = iter_max;
#endif
#if USE_FEM_MATRIX
	build_P1_mass_matrix(m, M);
	build_P1_stiffness_matrix(m, S);
#else
	build_P1_CSRPattern(m, P);
	build_P1_mass_matrix(m, P, M);
	build_P1_stiffness_matrix(m, P, S);
#endif
	vol = M.sum();
	inited = false;
	t = 0;
}

void NavierStokesSolver::set_zero_mean(double *V)
{
	/* Your implementation goes here */
}

void NavierStokesSolver::compute_transport(double *T)
{
	memset(T, 0, N * sizeof(double));

	/* Your implementation goes here */
}

size_t NavierStokesSolver::compute_stream_function()
{
	size_t iter = 0;

	/* Your implementation goes here */

	return iter;
}

void NavierStokesSolver::time_step(double dt, double nu)
{
	compute_stream_function();

	/**********************************************************************
	 * Solve the system :
	 *
	 *  (M + \nu * dt * S)omega(t+dt) = M * omega(t) + dt * T(Omega,Psi)(t)
	 *
	 *********************************************************************/

	/* Your implementation goes here */

	set_zero_mean(omega.data);

	t += dt;
}
