#include "execute.hpp"
#include "dag.hpp"

float execute(const Program& prog, float input) {
    bool live[Program::MAX_INSTRS];
    compute_dag(prog, live);

    Reg regs[Program::NUM_REGS] = {};
    regs[0].type  = RegType::FLOAT;
    regs[0].val.f = input;

    const Instr* ins = prog.instrs;
    const Instr* end = ins + prog.num_instrs;

    for (int idx = 0; ins != end; ++ins, ++idx) {
        if (!live[idx]) continue;
        const int di    = ins->dst  % Program::NUM_REGS;
        const RegVal& a = regs[ins->src1 % Program::NUM_REGS].val;
        const RegVal& b = regs[ins->src2 % Program::NUM_REGS].val;
        Reg& d          = regs[di];

        switch (ins->op) {
        case Op::IADD:  d.type=RegType::INT;   d.val.i = a.i + b.i;           break;
        case Op::ISUB:  d.type=RegType::INT;   d.val.i = a.i - b.i;           break;
        case Op::IMUL:  d.type=RegType::INT;   d.val.i = a.i * b.i;           break;
        case Op::FADD:  d.type=RegType::FLOAT; d.val.f = a.f + b.f;           break;
        case Op::FSUB:  d.type=RegType::FLOAT; d.val.f = a.f - b.f;           break;
        case Op::FMUL:  d.type=RegType::FLOAT; d.val.f = a.f * b.f;           break;
        case Op::BAND:  d.type=RegType::INT;   d.val.u = a.u & b.u;           break;
        case Op::BOR:   d.type=RegType::INT;   d.val.u = a.u | b.u;           break;
        case Op::BXOR:  d.type=RegType::INT;   d.val.u = a.u ^ b.u;           break;
        case Op::LSHL:  d.type=RegType::INT;   d.val.u = a.u << (b.u & 31u);  break;
        case Op::LSHR:  d.type=RegType::INT;   d.val.u = a.u >> (b.u & 31u);  break;
        case Op::ASHL:  d.type=RegType::INT;   d.val.i = a.i << (b.u & 31u);  break;
        case Op::ASHR:  d.type=RegType::INT;   d.val.i = a.i >> (b.u & 31u);  break;
        case Op::ILT:   d.type=RegType::INT;   d.val.i = a.i <  b.i ? 1 : 0; break;
        case Op::IEQ:   d.type=RegType::INT;   d.val.i = a.i == b.i ? 1 : 0; break;
        case Op::ULT:   d.type=RegType::INT;   d.val.i = a.u <  b.u ? 1 : 0; break;
        case Op::UEQ:   d.type=RegType::INT;   d.val.i = a.u == b.u ? 1 : 0; break;
        case Op::FLT:   d.type=RegType::INT;   d.val.i = a.f <  b.f ? 1 : 0; break;
        case Op::FEQ:   d.type=RegType::INT;   d.val.i = a.f == b.f ? 1 : 0; break;
        case Op::BNOT:  d.type=RegType::INT;   d.val.u = ~a.u;                break;
        case Op::LNOT:  d.type=RegType::INT;   d.val.i = a.i == 0 ? 1 : 0;   break;
        case Op::INEG:  d.type=RegType::INT;   d.val.i = -a.i;                break;
        case Op::FNEG:  d.type=RegType::FLOAT; d.val.f = -a.f;                break;
        case Op::ITF:   d.type=RegType::FLOAT; d.val.u = a.u;                 break;
        case Op::FTI:   d.type=RegType::INT;   d.val.u = a.u;                 break;
        case Op::LOADI: d.type=RegType::INT;   d.val.i = ins->lit.i;          break;
        case Op::LOADF: d.type=RegType::FLOAT; d.val.f = ins->lit.f;          break;
        case Op::MOV:   d = regs[ins->src1 % Program::NUM_REGS];              break;
        default: break;
        }
    }
    return regs[0].val.f;
}
