#pragma once
#include "program.hpp"
#include <vector>

static constexpr int N_CASES = 100;

extern const std::vector<float> TEST_INPUTS;

// Evaluates the program on all test inputs. Fills case_err[N_CASES] with
// per-case squared relative error. Returns aggregate MSRE + penalties.
double fitness_and_cases(const Program& prog, float case_err[N_CASES]);

// Convenience wrapper: aggregate fitness only.
double fitness(const Program& prog);
