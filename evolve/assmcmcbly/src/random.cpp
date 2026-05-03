#include "random.hpp"
#include "innovation.hpp"

static bool op_is_unary(Op op) {
    return op == Op::BNOT || op == Op::LNOT || op == Op::INEG ||
           op == Op::FNEG || op == Op::ITF  || op == Op::FTI  ||
           op == Op::MOV;
}

static bool op_is_terminal(Op op) {
    return op == Op::LOADI || op == Op::LOADF;
}

static bool op_supports_imm(Op op) {
    switch (op) {
    case Op::IADD: case Op::ISUB: case Op::IMUL:
    case Op::BAND: case Op::BOR:  case Op::BXOR:
    case Op::LSHL: case Op::LSHR: case Op::ASHL: case Op::ASHR:
    case Op::ILT:  case Op::IEQ:  case Op::ULT:  case Op::UEQ:
        return true;
    default:
        return false;
    }
}

Node random_node(int n_available, std::mt19937& rng) {
    std::uniform_int_distribution<int>      op_d(0, int(Op::COUNT) - 1);
    std::uniform_int_distribution<uint32_t> any32;

    Node nd;
    nd.op    = Op(op_d(rng));
    nd.lit.i = int32_t(any32(rng));
    nd.innov = next_innovation();

    int n_src = (n_available > 0) ? n_available : 1;
    std::uniform_int_distribution<int> src_d(0, n_src - 1);

    nd.src1 = uint16_t(src_d(rng));

    if (op_is_terminal(nd.op) || op_is_unary(nd.op)) {
        nd.src2 = 0;
    } else if (op_supports_imm(nd.op) && (rng() & 3) == 0) {
        nd.src2 = Program::IMM_SRC;
    } else {
        nd.src2 = uint16_t(src_d(rng));
    }

    return nd;
}

bool add_random_node(Program& prog, std::mt19937& rng) {
    if (prog.n_nodes >= Program::MAX_NODES) return false;
    int idx = prog.n_nodes;
    int n_avail = idx > 0 ? idx : 1;
    prog.nodes[idx] = random_node(n_avail, rng);
    prog.n_nodes++;
    return true;
}

Program random_program(int n_inputs, std::mt19937& rng) {
    std::uniform_int_distribution<int> nc_d(1, 8);
    int n_func = nc_d(rng);

    Program prog;
    prog.n_inputs = uint16_t(n_inputs);
    prog.n_nodes  = uint16_t(n_inputs);

    for (int i = 0; i < n_func; i++)
        add_random_node(prog, rng);

    prog.output_node = uint16_t(prog.n_nodes - 1);
    return prog;
}
