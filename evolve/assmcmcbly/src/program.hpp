#pragma once
#include "types.hpp"

// ── Program ───────────────────────────────────────────────────────────────────
//
// Flat layout: instrs[0..num_instrs-1] run in order.
// Chromosome boundaries live in chrom_lens[0..num_chroms-1].
// Execution ignores chromosomes — they exist only for genetic ops.

struct Program {
    static constexpr int NUM_REGS        = 16;
    static constexpr int MAX_CHROMOSOMES = 64;
    static constexpr int MAX_CHROM_LEN   = 8;
    static constexpr int MAX_INSTRS      = MAX_CHROMOSOMES * MAX_CHROM_LEN; // 512

    Instr    instrs[MAX_INSTRS]          = {};
    uint8_t  chrom_lens[MAX_CHROMOSOMES] = {};
    uint8_t  num_chroms                  = 0;
    uint16_t num_instrs                  = 0;

    // O(num_chroms) — only called by genetic ops, not execution
    int chrom_start(int ci) const {
        int s = 0;
        for (int j = 0; j < ci; j++) s += chrom_lens[j];
        return s;
    }
};
