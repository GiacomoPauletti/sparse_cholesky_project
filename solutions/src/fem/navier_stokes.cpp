#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "navier_stokes.h"

#include "P1.h"
#include "tiny_blas.h"

NavierStokesSolver::NavierStokesSolver(const Mesh &m, Profiler *profiler,
				       LinearSolverKind backend, double tol,
				       size_t iter_max,
				       CholeskyOrderingKind ordering)
    : m(m), N(m.vertex_count()), omega(N), Momega(N), psi(N), r(N), p(N), Ap(N)
{
	/* Assigned here rather than in the initializer list : these members are
	 * declared after the arrays, and build_stream_solver() below reads all
	 * of them. */
	this->profiler = profiler;
	this->backend = backend;
	this->tol = tol;
	this->iter_max = iter_max;
	this->cholesky_ordering = ordering;

	{
		ProfileSection section(profiler, "assembly");
		{
			ProfileStep step(profiler, "P1 CSR pattern");
			build_P1_CSRPattern(m, P);
		}
		{
			ProfileStep step(profiler, "mass matrix M");
			build_P1_mass_matrix(m, P, M);
		}
		{
			ProfileStep step(profiler, "stiffness matrix S");
			build_P1_stiffness_matrix(m, P, S);
		}
		{
			ProfileStep step(profiler, "total volume (M.sum)");
			vol = M.sum();
		}
	}
	inited = false;
	t = 0;
	build_stream_solver();
}

NavierStokesSolver::~NavierStokesSolver()
{
	delete stream_solver;
	delete vort_solver;
}

LinearSolver *NavierStokesSolver::make_solver(CSRMatrix *A)
{
	LinearSolver *solver;

	if (backend == SOLVER_CG) {
		/* tol and iter_max are read here, when the solver is built. If you
		 * change them afterwards, call set_backend() or let the next
		 * update_vort_solver() rebuild to make them take effect. */
		solver = new CGSolver(tol, (int)iter_max, profiler);
	} else {
		solver = new SparseCholeskySolver(A, cholesky_ordering,
						  profiler);
	}
	/* Cheap for CG, this is where the factorization happens for Cholesky. */
	solver->initialize(A);

	return solver;
}

void NavierStokesSolver::set_backend(LinearSolverKind kind)
{
	if (kind == backend) {
		return;
	}
	backend = kind;

	delete stream_solver;
	stream_solver = nullptr;
	delete vort_solver;
	vort_solver = nullptr;
	/* Forces update_vort_solver() to reassemble K at the next time step. */
	cached_coeff = -1.0;

	build_stream_solver();
}

void NavierStokesSolver::build_stream_solver()
{
	/* The stiffness matrix is the pure-Neumann Laplacian, which is singular
	 * (the constant vector is in its kernel). A Cholesky factorization needs an
	 * SPD matrix, so we copy S and pin DoF 0 : its row/column is decoupled and
	 * its diagonal set to 1, which fixes psi[0] = 0 when the rhs entry is 0.
	 * Since the stream function only enters compute_transport() through
	 * differences, this constant shift is harmless. Pinning is not required
	 * by CG (which copes with the consistent singular system), but sharing
	 * the same operator between the two backends keeps them comparable, and
	 * CG simply drives psi[0] to 0 as well. */
	ProfileSection section(profiler, "stream solver setup");
	ProfileStep pin(profiler, "copy and pin S");

	Spin.symmetric = true;
	Spin.rows = S.rows;
	Spin.cols = S.cols;
	Spin.nnz = S.nnz;
	Spin.row_start = S.row_start; /* shares the P1 pattern P */
	Spin.col = S.col;
	Spin.data.resize(S.nnz);
	for (size_t k = 0; k < S.nnz; ++k) {
		Spin.data[k] = S.data[k];
	}
	/* Lower triangular storage : row 0 holds only its diagonal, and the
	 * coupling to DoF 0 in the other rows are the entries with col == 0. */
	for (size_t i = 0; i < N; ++i) {
		for (uint32_t k = Spin.row_start[i]; k < Spin.row_start[i + 1]; ++k) {
			if (Spin.col[k] == 0) {
				Spin.data[k] = (i == 0) ? 1.0 : 0.0;
			}
		}
	}
	pin.stop();

	stream_solver = make_solver(&Spin);
}

void NavierStokesSolver::update_vort_solver(double coeff)
{
	/* Reassemble K = M + coeff * S only when coeff = nu*dt changes (e.g. the
	 * user moved a slider) ; otherwise reuse the solver as is. Assembling K
	 * costs one pass over nnz, which is negligible next to a factorization,
	 * and it also lets CG do a single mvp per iteration instead of applying
	 * M and S separately. */
	if (vort_solver && coeff == cached_coeff) {
		return;
	}
	ProfileSection section(profiler, "vorticity solver setup");
	ProfileStep assemble(profiler, "assemble K = M + nu*dt*S");

	K.symmetric = true;
	K.rows = M.rows;
	K.cols = M.cols;
	K.nnz = M.nnz;
	K.row_start = M.row_start; /* M and S share the P1 pattern P */
	K.col = M.col;
	K.data.resize(M.nnz);
	for (size_t k = 0; k < M.nnz; ++k) {
		K.data[k] = M.data[k] + coeff * S.data[k];
	}
	assemble.stop();

	delete vort_solver;
	vort_solver = make_solver(&K);
	cached_coeff = coeff;
}

void NavierStokesSolver::set_zero_mean(double *V)
{
	ProfileStep step(profiler, "set zero mean");

	M.mvp(V, Ap.data);
	double s = blas_sum_in_place(Ap.data, N);
	for (size_t i = 0; i < N; ++i) {
		V[i] -= s / vol;
	}
}

void NavierStokesSolver::compute_transport(double *T)
{
	ProfileStep step(profiler, "compute transport");

	memset(T, 0, N * sizeof(double));

	for (size_t t = 0; t < m.triangle_count(); t++) {
		uint32_t a = m.indices[3 * t + 0];
		uint32_t b = m.indices[3 * t + 1];
		uint32_t c = m.indices[3 * t + 2];
		assert(a < N && b < N && c < N);
		double sum = omega[a] + omega[b] + omega[c];
		T[a] += sum * (psi[b] - psi[c]);
		T[b] += sum * (psi[c] - psi[a]);
		T[c] += sum * (psi[a] - psi[b]);
	}
	for (size_t v = 0; v < N; v++) {
		T[v] *= 1.0 / 6;
	}
}

size_t NavierStokesSolver::compute_stream_function()
{
	ProfileSection section(profiler, "stream function");

	double *R = r.data;
	double *Om = omega.data;
	double *MOm = Momega.data;
	double *Psi = psi.data;

	/* Solve Spin * Psi = M * omega, starting from the previous Psi (which CG
	 * exploits as a warm start and Cholesky ignores). */
	{
		ProfileStep step(profiler, "rhs : M * omega");
		M.mvp(Om, MOm);

		/* Copy the rhs into scratch R and pin DoF 0 (psi[0] = 0) to
		 * match the pinned operator. We keep MOm intact since
		 * time_step() reuses it. */
		blas_copy(MOm, R, N);
		R[0] = 0.0;
	}

	stream_solver->solve(Psi, R);

	return stream_solver->iterations();
}

void NavierStokesSolver::time_step(double dt, double nu)
{
	ProfileSection section(profiler, "time step");

	double *P = p.data;
	double *Om = omega.data;
	double *MOm = Momega.data;

	/* Computes Psi and, as a side effect, MOm = M * omega(t). */
	stream_iter = compute_stream_function();

	/**********************************************************************
	 * Solve the system :
	 *
	 *  (M + \nu * dt * S)omega(t+dt) = M * omega(t) + dt * T(Omega,Psi)(t)
	 *
	 *********************************************************************/

	{
		ProfileSection vorticity(profiler, "vorticity");

		/* Form rhs, saved in P */
		compute_transport(P);
		{
			ProfileStep step(profiler, "rhs : M*omega + dt*T");
			blas_axpby(1, MOm, dt, P, N);
		}

		/* Solve with the selected backend, starting from omega(t).
		 * Rebuilding the solver only happens when nu*dt changed, so
		 * the setup lines usually show far fewer calls than the solve
		 * ones. */
		update_vort_solver(nu * dt);
		vort_solver->solve(Om, P);
		vort_iter = vort_solver->iterations();

		set_zero_mean(omega.data);
	}

	t += dt;
}
