#pragma once
#include "program.hpp"
#include "hardening.hpp"
#include "problem.hpp"
#include <random>
#include <vector>

Program mutate   (const Program& src, const Hardness& hardness, std::mt19937& rng,
                  const ProblemDef& problem, const std::vector<float>& test_inputs);
Program crossover(const Program& a, double fa, const Program& b, double fb, std::mt19937& rng);
