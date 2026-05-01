#pragma once
#include <cstdint>

// ── Register ──────────────────────────────────────────────────────────────────

enum class RegType : uint8_t { INT, FLOAT };

union RegVal {
    int32_t  i;
    uint32_t u;
    float    f;
};

struct Reg {
    RegType type = RegType::FLOAT;
    RegVal  val  = {};
};

// ── Operations ────────────────────────────────────────────────────────────────

enum class Op : uint8_t {
    IADD, ISUB, IMUL,
    FADD, FSUB, FMUL,
    BAND, BOR, BXOR,
    LSHL, LSHR,           // logical shift  (zero-fill)
    ASHL, ASHR,           // arithmetic shift (sign-extend right)
    ILT,  IEQ,            // signed int compare  → int 0/1
    ULT,  UEQ,            // unsigned int compare → int 0/1
    FLT,  FEQ,            // float compare        → int 0/1
    BNOT, LNOT, INEG, FNEG,
    ITF, FTI,             // reinterpret bits, flip type tag
    LOADI, LOADF,
    MOV,
    COUNT
};

// ── Instruction ───────────────────────────────────────────────────────────────

struct Instr {
    Op      op   = Op::MOV;
    uint8_t dst  = 0;
    uint8_t src1 = 0;
    uint8_t src2 = 0;
    union { int32_t i; float f; } lit = {};
};
