#pragma once
#include "program.hpp"
#include <cstdint>

// Compute which instructions transitively contribute to the value in r0 at
// the end of execution.  live[i] = true if instruction i is in the DAG.
// Returns the count of live instructions.
int compute_dag(const Program& prog, bool live[Program::MAX_INSTRS]);
