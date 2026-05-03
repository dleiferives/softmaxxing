#pragma once
#include "program.hpp"
#include <vector>

static constexpr int N_CASES = 100;

extern std::vector<float> TEST_INPUTS;

// Rebuild TEST_INPUTS as N_CASES log-spaced points in [x_lo, x_hi].
void set_curriculum_range(float x_lo, float x_hi);

// Evaluates the program on all test inputs. Fills case_err[N_CASES] with
// per-case squared relative error. Returns aggregate MSRE + penalties.
double fitness_and_cases(const Program& prog, float case_err[N_CASES]);

// Convenience wrapper: aggregate fitness only.
double fitness(const Program& prog);
