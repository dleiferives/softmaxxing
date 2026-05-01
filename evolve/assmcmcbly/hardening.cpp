#include "hardening.hpp"
#include "fitness.hpp"
#include <algorithm>

void Hardness::recompute(const Program& prog, double base_fitness) {
    // Scratch no-op: MOV r15 r15 — writes to r15 which nothing reads
    static const Instr NOOP = [](){
        Instr i; i.op = Op::MOV;
        i.dst = 15; i.src1 = 15; i.src2 = 0;
        return i;
    }();

    for (int i = 0; i < prog.num_instrs; i++) {
        Program tmp = prog;
        tmp.instrs[i] = NOOP;
        double ablated = fitness(tmp);
        // hardness = how much worse things got without this instruction
        scores[i] = float(std::max(0.0, ablated - base_fitness));
    }
    // zero out any stale entries beyond current program length
    for (int i = prog.num_instrs; i < Program::MAX_INSTRS; i++)
        scores[i] = 0.0f;
}
