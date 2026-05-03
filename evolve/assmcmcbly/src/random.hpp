#pragma once
#include "program.hpp"
#include <random>

// Generate a random function node whose src indices are all < n_available.
Node random_node(int n_available, std::mt19937& rng);

// Append one random function node to prog.  Returns false if prog is full.
bool add_random_node(Program& prog, std::mt19937& rng);

// Build a small random program: n_inputs input terminals + 1–8 random function nodes.
Program random_program(int n_inputs, std::mt19937& rng);
