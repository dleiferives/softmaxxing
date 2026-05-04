#include "problem.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

std::pair<float,float> InputSpec::range_at(int stage) const {
    if (!use_curriculum || curriculum.empty())
        return range;
    int idx = std::min(stage, int(curriculum.size()) - 1);
    return curriculum[idx];
}

void finalize_problem(ProblemDef& p) {
    if (int(p.inputs.size()) < p.n_inputs)
        p.inputs.resize(p.n_inputs);

    if (p.behavior_samples.empty()) {
        p.behavior_samples.resize(N_BEH * p.n_inputs);
        for (int j = 0; j < p.n_inputs; j++) {
            auto [lo, hi] = p.inputs[j].range;
            for (int i = 0; i < N_BEH; i++) {
                float t = float(i) / float(N_BEH - 1);
                p.behavior_samples[i * p.n_inputs + j] = lo * std::pow(hi / lo, t);
            }
        }
    }
}

int problem_n_stages(const ProblemDef& p) {
    int n = 1;
    for (auto& inp : p.inputs)
        if (inp.use_curriculum && !inp.curriculum.empty())
            n = std::max(n, int(inp.curriculum.size()));
    return n;
}

std::vector<float> make_test_inputs(const ProblemDef& p, int stage, int n_cases) {
    std::vector<float> v(n_cases * p.n_inputs);
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int j = 0; j < p.n_inputs; ++j) {
        auto [lo, hi] = p.inputs[j].range_at(stage);
        std::uniform_real_distribution<float> dist(lo, hi);

        for (int i = 0; i < n_cases; ++i) {
            v[i * p.n_inputs + j] = dist(gen);
        }
    }
    return v;
}
