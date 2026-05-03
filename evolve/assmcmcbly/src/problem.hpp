#pragma once
#include <functional>
#include <utility>
#include <vector>

// N_BEH lives here so speciation.hpp can include problem.hpp without circularity.
static constexpr int N_BEH = 5;

struct InputSpec {
    std::pair<float,float> range = {1e-4f, 1e4f};
    bool use_curriculum = false;
    // Curriculum stages: range widens from curriculum[0] → curriculum.back().
    // Ignored when use_curriculum == false.
    std::vector<std::pair<float,float>> curriculum;

    // Returns the active range at a given stage index.
    std::pair<float,float> range_at(int stage) const;
};

struct ProblemDef {
    int n_inputs  = 1;
    int n_outputs = 1;

    // Oracle: maps n_inputs floats → n_outputs floats (the ground-truth answer).
    std::function<void(const float* in, float* out)> oracle;

    // Per-input configuration. Must have n_inputs entries after finalize_problem().
    std::vector<InputSpec> inputs;

    // Fitness threshold to advance all inputs to the next curriculum stage.
    double curriculum_advance_thresh = 0.01;

    // Behavior sample inputs for speciation fingerprinting.
    // Flat layout: N_BEH samples × n_inputs values each.
    // Auto-populated (log-spaced from each input's range) if empty at finalize time.
    std::vector<float> behavior_samples;
};

// Called once before use: fills missing inputs[] entries and auto-fills behavior_samples.
void finalize_problem(ProblemDef& p);

// Max curriculum stage count across all inputs (minimum 1 if none use curriculum).
int problem_n_stages(const ProblemDef& p);

// Generate n_cases test inputs for the given stage.
// Returns flat vector: case0_in0, case0_in1, …, caseN_in(n_inputs-1).
// Each input dimension is log-spaced over its range_at(stage).
std::vector<float> make_test_inputs(const ProblemDef& p, int stage, int n_cases);
