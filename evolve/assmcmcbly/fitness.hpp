#pragma once
#include "program.hpp"
#include <vector>

extern const std::vector<float> TEST_INPUTS;

// Returns mean squared relative error, penalised by the fraction of dead
// (DAG-unreachable) instructions: fit * (1 + 0.1 * dead/total).
double fitness(const Program& prog);
