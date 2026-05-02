#pragma once
#include "program.hpp"

// Per-instruction hardness scores stored alongside a Program.
// hardness[i] = how much fitness worsens when instruction i is ablated.
// High hardness → instruction is load-bearing → less likely to be mutated.
struct Hardness {
    float scores[Program::MAX_INSTRS] = {};

    // Recompute hardness for prog given its current fitness baseline.
    // Replaces each instruction one at a time with a harmless no-op
    // (MOV r15 r15 → writes to a scratch register, not r0) and measures delta.
    void recompute(const Program& prog, double base_fitness);

    // Mutation probability weight for instruction i: lower = harder to mutate.
    float weight(int i) const {
        return 1.0f / (1.0f + scores[i]);
    }

    void reset(int from, int to) {
        for (int i = from; i < to; i++) scores[i] = 0.0f;
    }
};
