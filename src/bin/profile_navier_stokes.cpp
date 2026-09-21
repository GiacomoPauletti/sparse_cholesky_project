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
 *      profile_NS [n] [cholesky|cg] [tol] [steps] [out] [natural|nested]
 *                 [uplooking|multifrontal|parmultifrontal]
 *
 *      n      : subdivision level of the sphere         (default 16)
 *      solver : cholesky (direct) or cg (iterative)     (default cholesky)
 *      tol    : relative residual target, CG only       (default 1e-8)
 *      steps  : number of time steps to profile         (default 100)
 *      out    : report file                             (default performance.txt)
 *      order  : vertex ordering, Cholesky only          (default natural)
 *      fact   : factorization algorithm, Cholesky only  (default uplooking)
 *               parmultifrontal uses OMP_NUM_THREADS threads
 *
 * tol is accepted and ignored by the Cholesky backend, which is direct, and
 * order and fact likewise by CG, which never factorizes. order and fact are
 * independent : either factorization runs in either ordering, and the two
 * produce the same factor, so comparing them is a pure timing comparison.
 * The out argument exists for parameter sweeps : several runs sharing a
 * working directory would otherwise overwrite each other's report.
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

#ifdef _OPENMP
#include <omp.h>
#endif

/* Same defaults as the test_NS viewer. */
static const double DT = 0.002;
static const double NU = 1e-1;

static const int DEFAULT_SUBDIV = 16;
static const double DEFAULT_TOL = 1e-8;
static const int DEFAULT_STEPS = 100;
static const int ITER_MAX = 20000;

static const char *DEFAULT_PERF_PATH = "performance.txt";

static void syntax(const char *prg_name)
{
	printf("Syntax : %s [n] [cholesky|cg] [tol] [steps] [out] "
	       "[natural|nested] [uplooking|multifrontal|parmultifrontal]\n", prg_name);
	printf("         n      : sphere subdivision level     (default %d)\n",
	       DEFAULT_SUBDIV);
	printf("         solver : cholesky or cg              (default "
	       "cholesky)\n");
	printf("         tol    : CG relative residual target (default %g)\n",
	       DEFAULT_TOL);
	printf("         steps  : number of time steps        (default %d)\n",
	       DEFAULT_STEPS);
	printf("         out    : report file                 (default %s)\n",
	       DEFAULT_PERF_PATH);
	printf("         order  : natural or nested           (default "
	       "natural)\n");
	printf("         fact   : uplooking, multifrontal or parmultifrontal "
	       "(default uplooking)\n");
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
	const char *perf_path = DEFAULT_PERF_PATH;
	CholeskyOrderingKind ordering = CHOLESKY_ORDERING_NATURAL;
	CholeskyFactorizationKind factorization =
	    CHOLESKY_FACTORIZATION_UPLOOKING;

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
	if (argc > 5) {
		perf_path = argv[5];
	}
	if (argc > 6) {
		if (strcmp(argv[6], "natural") == 0) {
			ordering = CHOLESKY_ORDERING_NATURAL;
		} else if (strcmp(argv[6], "nested") == 0) {
			ordering = CHOLESKY_ORDERING_NESTED;
		} else {
			printf("Unknown ordering '%s'.\n", argv[6]);
			syntax(argv[0]);
			return EXIT_FAILURE;
		}
	}
	if (argc > 7) {
		if (strcmp(argv[7], "uplooking") == 0) {
			factorization = CHOLESKY_FACTORIZATION_UPLOOKING;
		} else if (strcmp(argv[7], "multifrontal") == 0) {
			factorization = CHOLESKY_FACTORIZATION_MULTIFRONTAL;
		} else if (strcmp(argv[7], "parmultifrontal") == 0) {
			factorization = CHOLESKY_FACTORIZATION_PAR_MULTIFRONTAL;
		} else {
			printf("Unknown factorization '%s'.\n", argv[7]);
			syntax(argv[0]);
			return EXIT_FAILURE;
		}
	}

	const char *backend_name =
	    (backend == SOLVER_CG) ? "conjugate gradient" : "cholesky";
	const char *ordering_name =
	    (ordering == CHOLESKY_ORDERING_NESTED) ? "nested" : "natural";
	const char *factorization_name =
	    (factorization == CHOLESKY_FACTORIZATION_PAR_MULTIFRONTAL)
		? "parallel multifrontal"
	    : (factorization == CHOLESKY_FACTORIZATION_MULTIFRONTAL)
		? "multifrontal"
		: "up-looking";

	char title[256];
	if (backend == SOLVER_CG) {
		snprintf(title, sizeof(title),
			 "Navier Stokes : sphere %d, %s, tol %g, %d step(s)",
			 subdiv, backend_name, tol, steps);
	} else {
		snprintf(title, sizeof(title),
			 "Navier Stokes : sphere %d, %s (%s, %s ordering), "
			 "%d step(s)",
			 subdiv, backend_name, factorization_name,
			 ordering_name, steps);
	}
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

	/* Recorded in the report so that a sweep of report files can be
	 * compared without having to guess the problem size from the file
	 * name. Triangles = 12 * n^2 and DoF = 6 * n^2 + 2 for a sphere
	 * obtained by subdividing each face of a cube into n x n quads, but
	 * the measured values are written rather than the formula. */
	profiler.setInfo("sphere subdivision", (size_t)subdiv);
	profiler.setInfo("elements (triangles)", mesh.triangle_count());
	profiler.setInfo("vertices (DoF)", mesh.vertex_count());
	profiler.setInfo("solver", backend_name);
	if (backend == SOLVER_CHOLESKY) {
		profiler.setInfo("ordering", ordering_name);
		profiler.setInfo("factorization", factorization_name);
	}
#ifdef _OPENMP
	/* Only the parallel factorization uses them, but recorded for every run. */
	profiler.setInfo("threads", (size_t)omp_get_max_threads());
#endif
	if (backend == SOLVER_CG) {
		profiler.setInfo("tolerance", tol);
		profiler.setInfo("max iterations", (size_t)ITER_MAX);
	}
	profiler.setInfo("nu", NU);
	profiler.setInfo("dt", DT);
	profiler.setInfo("time steps", (size_t)steps);

	/**********************************************************************
	 * Assembly and solver setup, both profiled from inside the solver.
	 *********************************************************************/
	profiler.beginSection("solver construction");
	NavierStokesSolver solver(mesh, &profiler, backend, tol, ITER_MAX,
				  ordering, factorization);
	profiler.endSection();

	profiler.beginSection("initial condition");
	reset_solver(solver);
	profiler.endSection();

	printf("Sphere subdivision : %d\n", subdiv);
	printf("DoF                : %zu\n", solver.N);
	printf("Triangles          : %zu\n", mesh.triangle_count());
	printf("Solver             : %s\n", backend_name);
	if (backend == SOLVER_CHOLESKY) {
		printf("Ordering           : %s\n", ordering_name);
		printf("Factorization      : %s\n", factorization_name);
	}
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
	size_t stream_iters = 0;
	size_t vort_iters = 0;

	profiler.beginSection("time loop");
	for (int i = 0; i < steps; ++i) {
		solver.time_step(DT, NU);
		stream_iters += solver.stream_iter;
		vort_iters += solver.vort_iter;
	}
	profiler.endSection();

	if (backend == SOLVER_CG && steps > 0) {
		/* Averaged over the run : the first step starts from psi = 0
		 * and needs noticeably more iterations than the later ones,
		 * which warm start from the previous solution. */
		double avg_stream = (double)stream_iters / steps;
		double avg_vort = (double)vort_iters / steps;
		profiler.setInfo("cg iterations per psi solve", avg_stream);
		profiler.setInfo("cg iterations per omega solve", avg_vort);
		printf("CG iterations per step : %.1f (psi), %.1f (omega)\n\n",
		       avg_stream, avg_vort);
	}

	/* Fill in : the number of nonzeros the factor carries compared to the
	 * operator it came from. This is the memory side of the direct method,
	 * which a timing alone does not show. */
	if (backend == SOLVER_CHOLESKY) {
		SparseCholeskySolver *chol =
		    dynamic_cast<SparseCholeskySolver *>(solver.stream_solver);
		if (chol && chol->getFactor()) {
			profiler.setInfo("nnz(A)", (size_t)solver.Spin.nnz);
			profiler.setInfo("nnz(L)",
					 (size_t)chol->getFactor()->nnz);
			printf("nnz(A) : %zu, nnz(L) : %zu (fill in x%.2f)\n\n",
			       (size_t)solver.Spin.nnz,
			       (size_t)chol->getFactor()->nnz,
			       (double)chol->getFactor()->nnz /
				   (double)solver.Spin.nnz);
		}
	}

	profiler.report(stdout);
	if (!profiler.dump(perf_path)) {
		return EXIT_FAILURE;
	}
	printf("\nWrote %s\n", perf_path);

	return EXIT_SUCCESS;
}
