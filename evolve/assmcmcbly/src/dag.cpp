#include "dag.hpp"
#include <cstring>

int compute_dag(const Program& prog, bool live[Program::MAX_INSTRS]) {
    const int n = prog.num_instrs;
    memset(live, 0, n * sizeof(bool));

    // needed[r] = true if the current value of register r is needed by a
    // downstream instruction or is the output.
    bool needed[Program::NUM_REGS] = {};
    needed[0] = true;  // r0 is the output

    // Walk backwards: an instruction is live if its dst is currently needed.
    // Once we mark it live, its src registers become needed.
    for (int i = n - 1; i >= 0; i--) {
        const Instr& ins = prog.instrs[i];
        int dst = ins.dst % Program::NUM_REGS;

        if (!needed[dst]) continue;

        live[i] = true;

        // LOADI/LOADF/BNOT/LNOT/INEG/FNEG/ITF/FTI/MOV only use src1 (or neither)
        switch (ins.op) {
        case Op::LOADI:
        case Op::LOADF:
            // no register sources
            break;
        case Op::BNOT: case Op::LNOT: case Op::INEG: case Op::FNEG:
        case Op::ITF:  case Op::FTI:  case Op::MOV:
            needed[ins.src1 % Program::NUM_REGS] = true;
            break;
        default:
            // binary op: uses src1 and src2
            needed[ins.src1 % Program::NUM_REGS] = true;
            needed[ins.src2 % Program::NUM_REGS] = true;
            break;
        }

        // dst is satisfied by this instruction; clear the need so that an
        // earlier write to the same register is not spuriously marked live.
        // But only if this is the last write to dst before the end — we
        // handle this by clearing and re-setting as we scan backwards.
        needed[dst] = false;
    }

    int count = 0;
    for (int i = 0; i < n; i++) count += live[i];
    return count;
}
