#pragma once
#include "program.hpp"
#include "problem.hpp"
#include <vector>
#include <chrono>

static constexpr int N_CASES = 200;

// Evaluates prog on test_inputs. Fills case_err[N_CASES] with per-case squared
// relative error. Returns aggregate MSRE + penalties.
// penalize_length=false omits the instruction-count penalty (use during novelty search).
double fitness_and_cases(const Program& prog,
                         float case_err[N_CASES],
                         const ProblemDef& problem,
                         const std::vector<float>& test_inputs,
                         bool penalize_length = true);

inline std::chrono::steady_clock::duration fitness_duration(0);
inline long fitness_calls;

// Convenience wrapper: aggregate fitness only.
double fitness(const Program& prog,
               const ProblemDef& problem,
               const std::vector<float>& test_inputs);
