#pragma once
#include "program.hpp"
#include <vector>
#include <limits>

static constexpr int N_BEH = 5;  // number of behavioral sample points

struct Species {
    Program  rep;
    float    rep_fp[N_BEH] = {};  // precomputed output fingerprint of rep
    double   best_fit   = std::numeric_limits<double>::max();
    int      stagnation = 0;
    std::vector<int> members;
};

// Run prog at 5 log-spaced x values and store outputs in out[].
void   compute_fingerprint(const Program& p, float out[N_BEH]);

// Mean absolute difference between two fingerprints.
double behavioral_distance(const float fa[N_BEH], const float fb[N_BEH]);
