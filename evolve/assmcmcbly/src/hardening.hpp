#pragma once
#include "program.hpp"
#include "problem.hpp"
#include <vector>

struct Hardness {
    float scores[Program::MAX_NODES] = {};

    void recompute(const Program& prog, double base_fitness,
                   const ProblemDef& problem,
                   const std::vector<float>& test_inputs);

    float weight(int i) const {
        return 1.0f / (1.0f + scores[i]);
    }

    void reset(int from, int to) {
        for (int i = from; i < to; i++) scores[i] = 0.0f;
    }
};
