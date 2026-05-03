#pragma once
#include "program.hpp"
#include "hardening.hpp"
#include <random>

// hardness is used to bias instruction selection: hard (load-bearing)
// instructions are less likely to be mutated or removed.
Program mutate   (const Program& src, const Hardness& hardness, std::mt19937& rng);
Program crossover(const Program& a, double fa, const Program& b, double fb, std::mt19937& rng);
