/* Headless profiling of the Navier Stokes pipeline.
 *
 * Runs the exact same computation as the test_NS viewer, minus the viewer :
 * a sphere mesh is generated, the P1 matrices are assembled, a linear solver
 * is built and a number of time steps are performed. Everything is measured by
 * a Profiler, whose report is written to performance.txt.
 *
 * Keeping this out of test_NS matters : in the viewer the solver time is
 * buried under the frame rate limiter, the GPU uploads and the idle frames,
 * which makes the numbers useless.
 *
 * Syntax :
 *      profile_NS [n] [cholesky|cg] [tol] [steps]
 *
 *      n      : subdivision level of the sphere         (default 16)
 *      solver : cholesky (direct) or cg (iterative)     (default cholesky)
 *      tol    : relative residual target, CG only       (default 1e-8)
 *      steps  : number of time steps to profile         (default 100)
 *
 * tol is accepted and ignored by the Cholesky backend, which is direct.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mesh.h"
#include "mesh_bounds.h"
#include "navier_stokes.h"
#include "profiler.h"
#include "sphere.h"

/* Same defaults as the test_NS viewer. */
static const double DT = 0.002;
static const double NU = 1e-1;

static const int DEFAULT_SUBDIV = 16;
static const double DEFAULT_TOL = 1e-8;
static const int DEFAULT_STEPS = 100;
static const int ITER_MAX = 20000;

static const char *PERF_PATH = "performance.txt";

static void syntax(const char *prg_name)
{
	printf("Syntax : %s [n] [cholesky|cg] [tol] [steps]\n", prg_name);
	printf("         n      : sphere subdivision level     (default %d)\n",
	       DEFAULT_SUBDIV);
	printf("         solver : cholesky or cg              (default "
	       "cholesky)\n");
	printf("         tol    : CG relative residual target (default %g)\n",
	       DEFAULT_TOL);
	printf("         steps  : number of time steps        (default %d)\n",
	       DEFAULT_STEPS);
}

/* Mirrors rescale_and_recenter_mesh() of test_navier_stokes.cpp, so that the
 * geometry (and hence the conditioning of the two systems) is the same as what
 * the viewer shows. */
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

int main(int argc, char **argv)
{
	int subdiv = DEFAULT_SUBDIV;
	LinearSolverKind backend = SOLVER_CHOLESKY;
	double tol = DEFAULT_TOL;
	int steps = DEFAULT_STEPS;

	if (argc > 1) {
		if (strcmp(argv[1], "-h") == 0 ||
		    strcmp(argv[1], "--help") == 0) {
			syntax(argv[0]);
			return EXIT_SUCCESS;
		}
		subdiv = atoi(argv[1]);
		if (subdiv <= 0) {
			printf("Invalid subdivision level '%s'.\n", argv[1]);
			syntax(argv[0]);
			return EXIT_FAILURE;
		}
	}
	if (argc > 2) {
		if (strcmp(argv[2], "cholesky") == 0) {
			backend = SOLVER_CHOLESKY;
		} else if (strcmp(argv[2], "cg") == 0) {
			backend = SOLVER_CG;
		} else {
			printf("Unknown solver '%s'.\n", argv[2]);
			syntax(argv[0]);
			return EXIT_FAILURE;
		}
	}
	if (argc > 3) {
		tol = atof(argv[3]);
		if (tol <= 0) {
			printf("Invalid tolerance '%s'.\n", argv[3]);
			syntax(argv[0]);
			return EXIT_FAILURE;
		}
	}
	if (argc > 4) {
		steps = atoi(argv[4]);
		if (steps < 0) {
			printf("Invalid step count '%s'.\n", argv[4]);
			syntax(argv[0]);
			return EXIT_FAILURE;
		}
	}

	const char *backend_name =
	    (backend == SOLVER_CG) ? "conjugate gradient" : "cholesky";

	char title[256];
	snprintf(title, sizeof(title),
		 "Navier Stokes : sphere %d, %s, tol %g, %d step(s)", subdiv,
		 backend_name, tol, steps);
	Profiler profiler(title);

	/**********************************************************************
	 * Mesh.
	 *********************************************************************/
	Mesh mesh;
	profiler.beginSection("mesh");
	profiler.startStep("generation (sphere)");
	int err = load_sphere(mesh, subdiv);
	profiler.endStep();
	if (err) {
		fprintf(stderr, "Could not build sphere mesh.\n");
		return EXIT_FAILURE;
	}
	profiler.startStep("rescale and recenter");
	rescale_and_recenter_mesh(mesh);
	profiler.endStep();
	profiler.endSection();

	/**********************************************************************
	 * Assembly and solver setup, both profiled from inside the solver.
	 *********************************************************************/
	profiler.beginSection("solver construction");
	NavierStokesSolver solver(mesh, &profiler, backend, tol, ITER_MAX);
	profiler.endSection();

	profiler.beginSection("initial condition");
	reset_solver(solver);
	profiler.endSection();

	printf("Sphere subdivision : %d\n", subdiv);
	printf("DoF                : %zu\n", solver.N);
	printf("Triangles          : %zu\n", mesh.triangle_count());
	printf("Solver             : %s\n", backend_name);
	if (backend == SOLVER_CG) {
		printf("Tolerance          : %g\n", tol);
		printf("Max iterations     : %d\n", ITER_MAX);
	}
	printf("nu                 : %g\n", NU);
	printf("dt                 : %g\n", DT);
	printf("Time steps         : %d\n\n", steps);

	/**********************************************************************
	 * Time loop. nu and dt are constant, so the vorticity solver is built
	 * at the first step only : its setup shows up with a single call, while
	 * the solve lines show `steps` of them.
	 *********************************************************************/
	profiler.beginSection("time loop");
	for (int i = 0; i < steps; ++i) {
		solver.time_step(DT, NU);
	}
	profiler.endSection();

	if (backend == SOLVER_CG) {
		printf("CG iterations of the last step : %zu (psi), %zu "
		       "(omega)\n\n",
		       solver.stream_iter, solver.vort_iter);
	}

	profiler.report(stdout);
	if (!profiler.dump(PERF_PATH)) {
		return EXIT_FAILURE;
	}
	printf("\nWrote %s\n", PERF_PATH);

	return EXIT_SUCCESS;
}
