#pragma once
#include "program.hpp"
#include <string>

// Emit a self-contained C++ file to {run_dir}/{gen}.cpp containing:
//   float jit_eval(float x)  — straight-line C translation of the program
//   float vm_eval(float x)   — register-machine interpreter (always correct)
// Both functions are dependency-free; compile with any C99/C++ compiler.
void emit_solution(const Program& prog, int gen, double fit,
                   const std::string& run_dir);
