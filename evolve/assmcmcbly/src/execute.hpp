#pragma once
#include "program.hpp"

// Evaluate the DAG.  nodes[0..n_in-1] are loaded from inputs[].
// outputs[0..n_out-1] are read as floats from the designated output node.
// For n_in=1, n_out=1 this is the standard single-input/single-output interface.
void execute(const Program& prog,
             const float* inputs, int n_in,
             float* outputs, int n_out);
