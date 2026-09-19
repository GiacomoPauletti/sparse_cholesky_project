#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

/******************************************************************************
 * Profiler : a hierarchical, aggregating wall clock profiler.
 *
 * The profiler records a tree of named timers. There are two kinds of nodes :
 *
 *   - sections, opened with beginSection(name) / closed with endSection().
 *     A section is a *container* : a logical phase of the program, which is
 *     expected to hold several finer measurements (and possibly subsections).
 *     Typical sections : "mesh", "assembly", "time step", "cholesky init".
 *
 *   - steps, opened with startStep(name) / closed with endStep().
 *     A step is the *leaf* unit of measurement, the thing you actually want a
 *     number for : "ordering", "symbolic", "factorization", "forward
 *     substitution", ...
 *
 * Both are pushed on the same stack, so they nest freely : endSection() and
 * endStep() always close the most recently opened node (the name is therefore
 * only needed when opening). Mismatching the kinds (closing a section with
 * endStep()) is reported at dump time rather than crashing, so that an
 * instrumented run never dies because of its instrumentation.
 *
 * Nodes are *aggregated* : reopening the same name under the same parent adds
 * to the same node. A step entered once per time step over 500 time steps
 * shows up as a single line with calls = 500, plus total / average / min /
 * max. This is what makes the profiler usable inside the main loop.
 *
 * The whole run is enclosed in an implicit root node whose timer starts at
 * construction and stops at dump(), so every percentage is relative to the
 * observed wall clock time.
 *
 * Usage :
 *
 *      Profiler prof("Navier Stokes");
 *      prof.beginSection("mesh");
 *          prof.startStep("generation");
 *          load_sphere(mesh, subdiv);
 *          prof.endStep();
 *      prof.endSection();
 *      ...
 *      prof.dump("performance.txt");
 *
 * or, exception/early-return safe and nullable (see ProfileSection /
 * ProfileStep below) :
 *
 *      void f(Profiler *prof) {
 *              ProfileSection s(prof, "mesh");
 *              ProfileStep    t(prof, "generation");
 *              ...
 *      }
 *
 * A null Profiler * means "do not profile" : every helper below degenerates
 * into a no-op, which is how the instrumented classes stay usable outside of a
 * profiling run.
 *
 * Not thread safe : one Profiler per thread.
 *****************************************************************************/

class Profiler {
    public:
	enum Kind { SECTION, STEP };

	explicit Profiler(const char *title = "Profile");

	/* Opens / closes a container node. */
	void beginSection(const char *name);
	void endSection();

	/* Opens / closes a measurement node. */
	void startStep(const char *name);
	void endStep();

	/* Closes whatever is still open (a section and a step have the same
	 * closing semantics, only the kind check differs). */
	void endAll();

	/* Drops every recorded timing and restarts the root timer. Handy to
	 * discard a warmup phase. */
	void reset();

	/* Writes the report. Returns false if the file could not be opened.
	 * dump() does not stop the profiler : timers still open are reported
	 * as such and measurement may continue afterwards. */
	bool dump(const char *path = "performance.txt") const;

	/* Same report, on an already opened stream (stdout for instance). */
	void report(void *stream) const;

	/* Elapsed wall clock since construction (or since the last reset). */
	double elapsedMs() const;

	const std::string &title() const { return name; }

    private:
	struct Node {
		std::string label;
		Kind kind = SECTION;
		int parent = -1;
		std::vector<int> children; /* in order of first appearance */

		size_t calls = 0;
		double total_ms = 0;
		double min_ms = 0;
		double max_ms = 0;

		/* Set while the node is open, for reentrancy diagnostics. */
		int open_depth = 0;
	};

	struct Frame {
		int node;
		int64_t start_ns;
	};

	int openNode(const char *name, Kind kind);
	void closeNode(Kind expected);
	int childByName(int parent, const char *name, Kind kind) const;
	void printNode(void *stream, int node, int depth, double root_ms) const;

	/* total_ms of the node minus the total_ms of its children, i.e. the
	 * time spent in the node itself and in whatever it does not measure. */
	double selfMs(int node) const;

	std::string name;
	std::vector<Node> nodes; /* nodes[0] is the implicit root */
	std::vector<Frame> stack;
	int64_t root_start_ns = 0;
	size_t mismatches = 0; /* endStep() on a section, and vice versa */
	size_t underflows = 0; /* end*() with nothing open */
};

/* RAII wrappers. Both accept a null Profiler *, in which case they do nothing,
 * so that instrumented code needs no "if (profiler)" guard. */

struct ProfileSection {
	explicit ProfileSection(Profiler *p, const char *name)
	    : prof(p)
	{
		if (prof) {
			prof->beginSection(name);
		}
	}
	~ProfileSection()
	{
		if (prof) {
			prof->endSection();
		}
	}
	ProfileSection(const ProfileSection &) = delete;
	ProfileSection &operator=(const ProfileSection &) = delete;

    private:
	Profiler *prof;
};

struct ProfileStep {
	explicit ProfileStep(Profiler *p, const char *name)
	    : prof(p)
	{
		if (prof) {
			prof->startStep(name);
		}
	}
	~ProfileStep()
	{
		stop();
	}
	/* Closes early, e.g. to measure only the first half of a scope. */
	void stop()
	{
		if (prof) {
			prof->endStep();
			prof = nullptr;
		}
	}
	ProfileStep(const ProfileStep &) = delete;
	ProfileStep &operator=(const ProfileStep &) = delete;

    private:
	Profiler *prof;
};
