#include "speciation.hpp"
#include "execute.hpp"
#include <cmath>
#include <algorithm>

void compute_fingerprint(const Program& p, float out[N_BEH], const ProblemDef& problem) {
    float out_val[1];
    for (int i = 0; i < N_BEH; i++) {
        const float* xs = &problem.behavior_samples[i * problem.n_inputs];
        execute(p, xs, problem.n_inputs, out_val, 1);
        out[i] = std::isfinite(out_val[0]) ? std::clamp(out_val[0], -1e6f, 1e6f) : 1e6f;
    }
}

