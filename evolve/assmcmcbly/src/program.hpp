#pragma once
#include "types.hpp"

// ── Node ──────────────────────────────────────────────────────────────────────
//
// One operation in the expression DAG.  The node's "output value" is identified
// by its index in Program::nodes[].  DAG acyclicity is guaranteed by construction:
// src1 < node_index, src2 < node_index (always).
//
// Layout convention in Program:
//   nodes[0..n_inputs-1]   — implicit input terminals; evaluator fills them
//                            from the input array; op field is ignored.
//   nodes[n_inputs..n_nodes-1] — function nodes; op is meaningful.
//
// LOADI / LOADF nodes are constant terminals: src1/src2 unused, lit holds value.
// Unary ops (BNOT LNOT INEG FNEG ITF FTI MOV) only consume src1.
// IMM_SRC in src2: use lit.i as a 32-bit integer immediate (same as before).

struct Node {
    Op       op    = Op::MOV;
    uint16_t src1  = 0;
    uint16_t src2  = 0;
    union { int32_t i; float f; } lit = {};
    uint32_t innov = 0;  // structural identity; preserved across field mutations
};

// ── Program ───────────────────────────────────────────────────────────────────

struct Program {
    static constexpr uint16_t MAX_NODES = 512;
    static constexpr uint16_t IMM_SRC   = 0xFFFF;

    Node     nodes[MAX_NODES] = {};
    uint16_t n_nodes          = 0;   // total allocated nodes (terminals + function nodes)
    uint16_t n_inputs         = 0;   // first n_inputs nodes are implicit input terminals
    uint16_t output_node      = 0;   // index of the node whose value is the program output
};
