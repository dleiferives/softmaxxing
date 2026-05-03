#pragma once
#include "program.hpp"

// Fills live[0..prog.n_nodes-1] via backward DFS from prog.output_node.
// live[i] = true iff node i contributes to the output value.
// Returns the count of live nodes.
int compute_live(const Program& prog, bool live[Program::MAX_NODES]);
