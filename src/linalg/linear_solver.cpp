#include "linear_solver.h"

/* Out of line destructor : this anchors the vtable of LinearSolver in a single
 * translation unit instead of emitting it in every file that includes the
 * header. */
LinearSolver::~LinearSolver() = default;
