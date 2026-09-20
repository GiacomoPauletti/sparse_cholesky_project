#include <stdio.h>
#include <string.h>
#include <time.h>

#include "profiler.h"

/* Monotonic clock, in nanoseconds. CLOCK_MONOTONIC is immune to wall clock
 * adjustments, contrary to the gettimeofday() used by common/chrono.h. */
static int64_t now_ns()
{
	timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (int64_t)ts.tv_sec * 1000000000 + (int64_t)ts.tv_nsec;
}

Profiler::Profiler(const char *title)
    : name(title ? title : "Profile")
{
	reset();
}

void Profiler::setInfo(const char *key, const char *value)
{
	std::string k = key ? key : "?";
	for (size_t i = 0; i < info.size(); ++i) {
		if (info[i].first == k) {
			info[i].second = value ? value : "";
			return;
		}
	}
	info.push_back(std::make_pair(k, std::string(value ? value : "")));
}

void Profiler::setInfo(const char *key, double value)
{
	char buf[64];
	snprintf(buf, sizeof(buf), "%g", value);
	setInfo(key, buf);
}

void Profiler::setInfo(const char *key, size_t value)
{
	char buf[64];
	snprintf(buf, sizeof(buf), "%zu", value);
	setInfo(key, buf);
}

void Profiler::reset()
{
	nodes.clear();
	stack.clear();
	mismatches = 0;
	underflows = 0;

	Node root;
	root.label = name;
	root.kind = SECTION;
	root.parent = -1;
	nodes.push_back(root);

	root_start_ns = now_ns();
}

double Profiler::elapsedMs() const
{
	return (double)(now_ns() - root_start_ns) * 1e-6;
}

int Profiler::childByName(int parent, const char *name, Kind kind) const
{
	const Node &p = nodes[parent];
	for (size_t i = 0; i < p.children.size(); ++i) {
		const Node &c = nodes[p.children[i]];
		if (c.kind == kind && c.label == name) {
			return p.children[i];
		}
	}
	return -1;
}

int Profiler::openNode(const char *label, Kind kind)
{
	int parent = stack.empty() ? 0 : stack.back().node;

	int idx = childByName(parent, label ? label : "?", kind);
	if (idx < 0) {
		Node n;
		n.label = label ? label : "?";
		n.kind = kind;
		n.parent = parent;
		idx = (int)nodes.size();
		nodes.push_back(n);
		/* push_back may have reallocated, so re-index the parent. */
		nodes[parent].children.push_back(idx);
	}
	nodes[idx].open_depth++;

	Frame f;
	f.node = idx;
	f.start_ns = now_ns();
	stack.push_back(f);

	return idx;
}

void Profiler::closeNode(Kind expected)
{
	if (stack.empty()) {
		underflows++;
		return;
	}
	Frame f = stack.back();
	stack.pop_back();

	double ms = (double)(now_ns() - f.start_ns) * 1e-6;

	Node &n = nodes[f.node];
	if (n.kind != expected) {
		mismatches++;
	}
	n.open_depth--;
	/* A recursively reentered node would count the inner elapsed time
	 * twice in total_ms ; only the outermost activation is accumulated. */
	if (n.open_depth > 0) {
		return;
	}
	if (n.calls == 0 || ms < n.min_ms) {
		n.min_ms = ms;
	}
	if (n.calls == 0 || ms > n.max_ms) {
		n.max_ms = ms;
	}
	n.calls++;
	n.total_ms += ms;
}

void Profiler::beginSection(const char *label)
{
	openNode(label, SECTION);
}

void Profiler::endSection()
{
	closeNode(SECTION);
}

void Profiler::startStep(const char *label)
{
	openNode(label, STEP);
}

void Profiler::endStep()
{
	closeNode(STEP);
}

void Profiler::endAll()
{
	while (!stack.empty()) {
		closeNode(nodes[stack.back().node].kind);
	}
}

double Profiler::selfMs(int node) const
{
	const Node &n = nodes[node];
	double children = 0;
	for (size_t i = 0; i < n.children.size(); ++i) {
		children += nodes[n.children[i]].total_ms;
	}
	double self = n.total_ms - children;
	return (self > 0) ? self : 0;
}

/* Column widths of the report. NAME_W is the width of the (indented) name
 * column, chosen so that a line fits in 100 characters. */
#define NAME_W 46

void Profiler::printNode(void *stream, int node, int depth, double root_ms) const
{
	FILE *f = (FILE *)stream;
	const Node &n = nodes[node];

	/* Sections are suffixed with a slash, like directories, so that the
	 * containers stand out from the leaf measurements. */
	char label[256];
	snprintf(label, sizeof(label), "%*s%s%s", 2 * depth, "",
		 n.label.c_str(), n.kind == SECTION ? "/" : "");

	double pct = (root_ms > 0) ? 100.0 * n.total_ms / root_ms : 0;
	double avg = n.calls ? n.total_ms / (double)n.calls : 0;

	fprintf(f, "%-*.*s %8zu %12.3f %12.3f %10.3f %10.3f %10.3f %7.2f\n",
		NAME_W, NAME_W, label, n.calls, n.total_ms, selfMs(node), avg,
		n.min_ms, n.max_ms, pct);

	for (size_t i = 0; i < n.children.size(); ++i) {
		printNode(stream, n.children[i], depth + 1, root_ms);
	}
}

void Profiler::report(void *stream) const
{
	FILE *f = (FILE *)stream;
	double root_ms = elapsedMs();

	fprintf(f, "==========================================================="
		   "=====================================\n");
	fprintf(f, " %s\n", name.c_str());
	fprintf(f, " wall clock : %.3f ms\n", root_ms);
	for (size_t i = 0; i < info.size(); ++i) {
		fprintf(f, " %s : %s\n", info[i].first.c_str(),
			info[i].second.c_str());
	}
	fprintf(f, "==========================================================="
		   "=====================================\n");
	fprintf(f, " total : cumulated time of all the calls to a node.\n");
	fprintf(f, " self  : total minus the total of its children, i.e. time "
		   "not accounted for below.\n");
	fprintf(f, " %%     : total, as a fraction of the wall clock above.\n");
	fprintf(f, " A trailing / marks a section (container), the other lines "
		   "are steps (measurements).\n");
	fprintf(f, "-----------------------------------------------------------"
		   "-------------------------------------\n");
	fprintf(f, "%-*.*s %8s %12s %12s %10s %10s %10s %7s\n", NAME_W, NAME_W,
		"name", "calls", "total(ms)", "self(ms)", "avg(ms)", "min(ms)",
		"max(ms)", "%");
	fprintf(f, "-----------------------------------------------------------"
		   "-------------------------------------\n");

	/* The root's own total is the wall clock : it is never closed. */
	{
		char label[256];
		snprintf(label, sizeof(label), "%s/", name.c_str());
		double self = root_ms;
		for (size_t i = 0; i < nodes[0].children.size(); ++i) {
			self -= nodes[nodes[0].children[i]].total_ms;
		}
		fprintf(f,
			"%-*.*s %8d %12.3f %12.3f %10.3f %10s %10s %7.2f\n",
			NAME_W, NAME_W, label, 1, root_ms,
			(self > 0) ? self : 0, root_ms, "-", "-", 100.0);
	}
	for (size_t i = 0; i < nodes[0].children.size(); ++i) {
		printNode(stream, nodes[0].children[i], 1, root_ms);
	}

	fprintf(f, "-----------------------------------------------------------"
		   "-------------------------------------\n");

	if (!stack.empty() || mismatches || underflows) {
		fprintf(f, "\n Instrumentation warnings :\n");
		if (!stack.empty()) {
			fprintf(f, "   %zu node(s) still open at dump time :",
				stack.size());
			for (size_t i = 0; i < stack.size(); ++i) {
				fprintf(f, " %s",
					nodes[stack[i].node].label.c_str());
			}
			fprintf(f, "\n   (their time is not accounted for)\n");
		}
		if (mismatches) {
			fprintf(f,
				"   %zu section/step mismatch(es) : a section "
				"was closed with endStep() or vice versa.\n",
				mismatches);
		}
		if (underflows) {
			fprintf(f,
				"   %zu extra end() call(s) with nothing "
				"open.\n",
				underflows);
		}
	}
}

bool Profiler::dump(const char *path) const
{
	FILE *f = fopen(path ? path : "performance.txt", "w");
	if (!f) {
		fprintf(stderr, "Profiler : could not open %s for writing.\n",
			path ? path : "performance.txt");
		return false;
	}
	report(f);
	fclose(f);
	return true;
}
