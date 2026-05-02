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

bool append_random_chromosome(Program& prog, std::mt19937& rng) {
    if (prog.num_chroms >= Program::MAX_CHROMOSOMES) return false;
    std::uniform_int_distribution<int> len_d(1, Program::MAX_CHROM_LEN);
    int len = len_d(rng);
    if (prog.num_instrs + len > Program::MAX_INSTRS) return false;

    int ci = prog.num_chroms++;
    prog.chrom_lens[ci] = uint8_t(len);
    for (int i = 0; i < len; i++)
        prog.instrs[prog.num_instrs++] = random_instr(rng);
    return true;
}

Program random_program(std::mt19937& rng) {
    Program prog = {};
    std::uniform_int_distribution<int> nc_d(1, 8);
    int nc = nc_d(rng);
    for (int i = 0; i < nc; i++)
        append_random_chromosome(prog, rng);
    return prog;
}
