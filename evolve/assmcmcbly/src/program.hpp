#pragma once
#include "types.hpp"

// ── Program ───────────────────────────────────────────────────────────────────
//
// Flat layout: instrs[0..num_instrs-1] run in order.

struct Program {
    static constexpr int     NUM_REGS   = 16;
    // Sentinel in src2: use ins.lit.i as integer immediate instead of a register.
    // Only valid for integer/bitwise/shift ops — not float ops.
    static constexpr uint8_t IMM_SRC    = 0xFF;
    static constexpr int     MAX_INSTRS = 512;

    Instr    instrs[MAX_INSTRS] = {};
    uint16_t num_instrs         = 0;
};
