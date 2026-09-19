/* Convergence of the conjugate gradient towards the sparse Cholesky solution.
 *
 * Both backends solve exactly the same two SPD systems of a single Navier
 * Stokes time step :
 *
 *      Spin * psi     = M * omega(t)            (stream function)
 *      K    * omega   = M * omega(t) + dt * T   with K = M + nu*dt*S
 *
 * Cholesky is a direct method, so its answer is taken as the reference. CG is
 * then run from the very same initial state for a range of tolerances ; the
 * distance to the reference must decrease like the tolerance until it hits the
 * roundoff floor. Results are written as a CSV for plotting.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <filesystem>

#include "mesh.h"
#include "mesh_bounds.h"
#include "navier_stokes.h"
#include "profiler.h"
#include "sphere.h"

/* Same defaults as the test_NS viewer. */
static const double DT = 0.002;
static const double NU = 1e-1;
static const int SUBDIV = 16;
static const int ITER_MAX = 20000;

typedef std::chrono::high_resolution_clock Clock;

static double ms_since(const Clock::time_point &t0)
{
	std::chrono::duration<double, std::milli> d = Clock::now() - t0;
	return d.count();
}

/* Mirrors rescale_and_recenter_mesh() of test_navier_stokes.cpp, so that the
 * geometry (and hence the conditioning) matches what the viewer shows. */
static void rescale_and_recenter_mesh(Mesh &mesh)
{
	Aabb bbox = compute_mesh_bounds(mesh);
	Vec3 center = (bbox.min + bbox.max) * 0.5f;
	Vec3 extent = (bbox.max - bbox.min);
	float size = max(extent);
	if (size == 0) {
		size = 1;
	}
	for (size_t i = 0; i < mesh.vertex_count(); ++i) {
		mesh.positions[i] -= center;
		mesh.positions[i] /= (size / 2);
	}
}

/* The default initial vorticity of test_NS :
 * 100 * z * exp(-50*z^2) * (1 + 0.5 * cos(20 * theta)) */
static void reset_solver(NavierStokesSolver &s)
{
	for (size_t i = 0; i < s.N; ++i) {
		double x = s.m.positions[i].x;
		double y = s.m.positions[i].y;
		double z = s.m.positions[i].z;
		double theta = atan2(sqrt(x * x + y * y), z);
		s.omega[i] =
		    100 * z * exp(-50 * z * z) * (1 + 0.5 * cos(20 * theta));
	}
	s.set_zero_mean(s.omega.data);
	memset(s.psi.data, 0, s.N * sizeof(double));
	s.t = 0;
}

/* Relative l2 distance between two vectors, normalized by the reference. */
static double rel_l2(const TArray<double> &ref, const TArray<double> &v,
		     size_t N)
{
	double num = 0;
	double den = 0;
	for (size_t i = 0; i < N; ++i) {
		double d = ref[i] - v[i];
		num += d * d;
		den += ref[i] * ref[i];
	}
	return (den > 0) ? sqrt(num / den) : sqrt(num);
}

int main(int argc, char **argv)
{
	int subdiv = (argc > 1) ? atoi(argv[1]) : SUBDIV;
	const char *csv_path =
	    (argc > 2) ? argv[2] : "plots/cg_vs_cholesky.csv";
	const char *perf_path = (argc > 3) ? argv[3] : "performance.txt";

	Profiler profiler("CG vs Cholesky benchmark");

	Mesh mesh;
	profiler.beginSection("mesh");
	profiler.startStep("generation (sphere)");
	if (load_sphere(mesh, subdiv)) {
		fprintf(stderr, "Could not build sphere mesh.\n");
		return EXIT_FAILURE;
	}
	profiler.endStep();
	profiler.startStep("rescale and recenter");
	rescale_and_recenter_mesh(mesh);
	profiler.endStep();
	profiler.endSection();

	/**********************************************************************
	 * Reference : one time step with the direct solver.
	 *********************************************************************/
	profiler.beginSection("cholesky reference");
	Clock::time_point t0 = Clock::now();
	NavierStokesSolver ref(mesh, &profiler); /* factorizes Spin */
	double chol_setup_ms = ms_since(t0);

	reset_solver(ref);
	t0 = Clock::now();
	ref.time_step(DT, NU); /* factorizes K, then two substitutions */
	double chol_step_ms = ms_since(t0);
	profiler.endSection();

	printf("Sphere subdivision : %d\n", subdiv);
	printf("DoF                : %zu\n", ref.N);
	printf("nu                 : %g\n", NU);
	printf("dt                 : %g\n", DT);
	printf("Cholesky setup     : %.3f ms\n", chol_setup_ms);
	printf("Cholesky step      : %.3f ms\n\n", chol_step_ms);

	std::filesystem::path out(csv_path);
	if (out.has_parent_path()) {
		std::filesystem::create_directories(out.parent_path());
	}
	FILE *f = fopen(csv_path, "w");
	if (!f) {
		fprintf(stderr, "Could not open %s for writing.\n", csv_path);
		return EXIT_FAILURE;
	}
	fprintf(f, "tol,iter_psi,iter_omega,converged,err_omega,err_psi,"
		   "cg_setup_ms,cg_step_ms,n_dof,nu,dt,"
		   "chol_setup_ms,chol_step_ms\n");

	printf("%-10s %8s %8s %12s %12s %10s\n", "tol", "it_psi", "it_om",
	       "err_omega", "err_psi", "step_ms");

	/**********************************************************************
	 * Scan : one time step with CG, from the same initial state.
	 *********************************************************************/
	profiler.beginSection("cg scan");
	for (int e = 1; e <= 14; ++e) {
		double tol = pow(10.0, -e);

		/* One subsection per tolerance : the CG solve line then shows
		 * how the cost grows as the tolerance tightens, next to the
		 * (tolerance independent) Cholesky reference above. */
		char label[64];
		snprintf(label, sizeof(label), "tol = 1e-%d", e);
		ProfileSection scan(&profiler, label);

		t0 = Clock::now();
		NavierStokesSolver cg(mesh, &profiler);
		/* tol and iter_max are latched when the CG solvers are built,
		 * which set_backend() does, so they must be set beforehand. */
		cg.tol = tol;
		cg.iter_max = ITER_MAX;
		cg.set_backend(SOLVER_CG);
		double cg_setup_ms = ms_since(t0);

		reset_solver(cg);
		t0 = Clock::now();
		cg.time_step(DT, NU);
		double cg_step_ms = ms_since(t0);

		double err_omega = rel_l2(ref.omega, cg.omega, ref.N);
		double err_psi = rel_l2(ref.psi, cg.psi, ref.N);
		/* iter_max reached means the tolerance was not attained. */
		int converged = (cg.stream_iter < (size_t)ITER_MAX &&
				 cg.vort_iter < (size_t)ITER_MAX);

		fprintf(f,
			"%.1e,%zu,%zu,%d,%.17g,%.17g,%.6f,%.6f,%zu,%g,%g,%.6f,"
			"%.6f\n",
			tol, cg.stream_iter, cg.vort_iter, converged, err_omega,
			err_psi, cg_setup_ms, cg_step_ms, ref.N, NU, DT,
			chol_setup_ms, chol_step_ms);

		printf("%-10.1e %8zu %8zu %12.3e %12.3e %10.3f\n", tol,
		       cg.stream_iter, cg.vort_iter, err_omega, err_psi,
		       cg_step_ms);
	}

	profiler.endSection();

	fclose(f);
	printf("\nWrote %s\n", csv_path);

	profiler.dump(perf_path);
	printf("Wrote %s\n", perf_path);

	return EXIT_SUCCESS;
}
