#pragma once
#include "program.hpp"
#include "hardening.hpp"
#include "problem.hpp"
#include <random>
#include <vector>

Program mutate               (const Program& src, const Hardness& hardness, std::mt19937& rng,
                              const ProblemDef& problem, const std::vector<float>& test_inputs);
// NEAT-style crossover: aligns on innovation numbers, best for within-island (shared lineage).
Program crossover            (const Program& a, double fa, const Program& b, double fb, std::mt19937& rng);
// Positional crossover: generates N_CANDS candidates using hardness-biased cut points
// (prefer cuts right after/before high-hardness instructions), scores each by total
// inherited live hardness, returns best. Falls back to uniform cuts for diversity.
Program crossover_positional (const Program& a, const Hardness& ha,
                               const Program& b, const Hardness& hb,
                               std::mt19937& rng);
