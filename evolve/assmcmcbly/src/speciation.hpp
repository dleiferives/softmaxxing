#pragma once
#include "program.hpp"
#include "problem.hpp"
#include <vector>
#include <limits>

// Run prog at the N_BEH sample points in problem.behavior_samples and store outputs.
// Used by the novelty cache.
void compute_fingerprint(const Program& p, float out[N_BEH], const ProblemDef& problem);
