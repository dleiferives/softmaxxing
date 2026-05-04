#include "random.hpp"
#include "innovation.hpp"

Instr random_instr(std::mt19937& rng) {
    static std::uniform_int_distribution<int>      op_d (0, int(Op::COUNT) - 1);
    static std::uniform_int_distribution<int>      reg_d(0, Program::NUM_REGS - 1);
    static std::uniform_int_distribution<uint32_t> any32;

    Instr ins;
    ins.op    = Op(op_d(rng));
    ins.dst   = uint8_t(reg_d(rng));
    ins.src1  = uint8_t(reg_d(rng));
    ins.src2  = uint8_t(reg_d(rng));
    ins.lit.i = int32_t(any32(rng));
    ins.innov = next_innovation();
    return ins;
}

Program random_program(std::mt19937& rng) {
    Program prog = {};
    std::uniform_int_distribution<int> n_d(1, 8);
    int n = n_d(rng);
    for (int i = 0; i < n && prog.num_instrs < Program::MAX_INSTRS; i++)
        prog.instrs[prog.num_instrs++] = random_instr(rng);
    return prog;
}
