#pragma once
#include "program.hpp"
#include "problem.hpp"
#include <vector>

static constexpr int N_CASES = 100;

// Evaluates prog on test_inputs. Fills case_err[N_CASES] with per-case squared
// relative error. Returns aggregate MSRE + penalties.
double fitness_and_cases(const Program& prog,
                         float case_err[N_CASES],
                         const ProblemDef& problem,
                         const std::vector<float>& test_inputs);

// Convenience wrapper: aggregate fitness only.
double fitness(const Program& prog,
               const ProblemDef& problem,
               const std::vector<float>& test_inputs);
