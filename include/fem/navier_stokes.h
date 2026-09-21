#include "array.h"
#define USE_FEM_MATRIX false
#if USE_FEM_MATRIX
	#include "fem_matrix.h"
#else
	#include "sparse_matrix.h"
	#include "cholesky.h"
	#include "conjugate_gradient2.h"
	#include "linear_solver.h"
#endif
#include "mesh.h"
#include "profiler.h"

#if !USE_FEM_MATRIX
/* Which linear solver backend the two SPD systems are solved with. */
enum LinearSolverKind {
	SOLVER_CHOLESKY, /* direct  : sparse Cholesky factorization */
	SOLVER_CG        /* iterative : conjugate gradient */
};
#endif

struct NavierStokesSolver {
	/* profiler is optional : nullptr (the default) leaves the solver, and
	 * the linear solvers it builds, uninstrumented. It is not owned and
	 * must outlive the solver. */
#if USE_FEM_MATRIX
	NavierStokesSolver(const Mesh &m, Profiler *profiler = nullptr);
#else
	/* backend, tol and iter_max are constructor arguments because the
	 * constructor already builds the stream solver : setting the members
	 * afterwards would need a set_backend() rebuild, i.e. a wasted
	 * factorization when profiling the CG backend. */
	NavierStokesSolver(const Mesh &m, Profiler *profiler = nullptr,
			   LinearSolverKind backend = SOLVER_CHOLESKY,
			   double tol = 1e-6, size_t iter_max = 500,
			   CholeskyOrderingKind ordering =
			       CHOLESKY_ORDERING_NATURAL,
			   CholeskyFactorizationKind factorization =
			       CHOLESKY_FACTORIZATION_UPLOOKING);
#endif
	~NavierStokesSolver();
	const Mesh &m;
	size_t N;   // DoF
	double vol; // Surface(m), used for insuring zero mean to omega and psi

	TArray<double> omega;
	TArray<double> Momega;
	TArray<double> psi;
#if USE_FEM_MATRIX
	FEMatrix S; // Stiffness matrix
	FEMatrix M; // Mass matrix
#else
	CSRPattern P; // Pattern arrays
	CSRMatrix S;  // Stiffness matrix
	CSRMatrix M;  // Mass matrix
#endif
	TArray<double> r;  // scratch (rhs of the stream function solve)
	TArray<double> p;  // scratch (rhs of the vorticity solve)
	TArray<double> Ap; // scratch (used by set_zero_mean)

#if !USE_FEM_MATRIX
	/* The two SPD systems to solve at each time step, assembled once and
	 * handed to a LinearSolver (Cholesky or CG, see set_backend).
	 * The stiffness matrix is the pure-Neumann Laplacian, hence singular, so
	 * Spin pins DoF 0 (psi is defined up to a constant) to make it SPD. K is
	 * the vorticity system matrix M + nu*dt*S, rebuilt whenever nu*dt
	 * changes. */
	CSRMatrix Spin;
	CSRMatrix K;
	LinearSolver *stream_solver = nullptr;
	LinearSolver *vort_solver = nullptr;
	double cached_coeff = -1.0; // nu*dt the current vort_solver was built for

	LinearSolverKind backend = SOLVER_CHOLESKY;
	/* Which permutation the Cholesky backend factorizes in. Ignored by CG.
	 * Like backend, changing it needs the solvers rebuilt. */
	CholeskyOrderingKind cholesky_ordering = CHOLESKY_ORDERING_NATURAL;
	/* Which numerical factorization the Cholesky backend runs. Ignored by
	 * CG, and like the ordering it only takes effect on a rebuild. */
	CholeskyFactorizationKind cholesky_factorization =
	    CHOLESKY_FACTORIZATION_UPLOOKING;
	/* Switches backend and discards the current solvers. The vorticity one
	 * is rebuilt lazily at the next time_step(). */
	void set_backend(LinearSolverKind kind);

	/* Iteration counts of the last time_step(), both 1 for Cholesky. */
	size_t stream_iter = 0;
	size_t vort_iter = 0;

	LinearSolver *make_solver(CSRMatrix *A);
	void build_stream_solver();
	void update_vort_solver(double coeff);
#endif

	bool inited; // Initialization computes first residue and error

	size_t iter_max = 500;
	double tol = 1e-6;

	/* Handed down to every LinearSolver built by make_solver(). */
	Profiler *profiler = nullptr;

	double t;

	void set_zero_mean(double *V);
	size_t compute_stream_function();
	void compute_transport(double *T);
	void time_step(double dt, double nu);
};
