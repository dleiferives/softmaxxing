#pragma once
#include "program.hpp"

// Place inputs[0..n_in-1] into regs[0..n_in-1], execute, read outputs[0..n_out-1]
// from regs[0..n_out-1].  For n_in=1, n_out=1 this is identical to the old interface.
void execute(const Program& prog,
             const float* inputs, int n_in,
             float* outputs, int n_out);
