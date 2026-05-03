#pragma once
#include "program.hpp"
#include "problem.hpp"
#include <vector>
#include <limits>

struct Species {
    Program  rep;
    double   best_fit   = std::numeric_limits<double>::max();
    int      stagnation = 0;
    std::vector<int> members;
};

// Run prog at the N_BEH sample points in problem.behavior_samples and store outputs.
// Used by the novelty cache — not for speciation.
void   compute_fingerprint(const Program& p, float out[N_BEH], const ProblemDef& problem);

// NEAT-style structural distance: aligns instructions by innovation number and counts
// excess, disjoint, and matching-gene differences.  Lower = more related.
double neat_distance(const Program& a, const Program& b);
