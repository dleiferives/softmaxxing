#pragma once
#include "program.hpp"
#include <random>

Instr  random_instr(std::mt19937& rng);
bool   append_random_chromosome(Program& prog, std::mt19937& rng);
Program random_program(std::mt19937& rng);
