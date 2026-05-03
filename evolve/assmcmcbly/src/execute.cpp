#include "execute.hpp"
#include "dag.hpp"

void execute(const Program& prog,
             const float* inputs, int n_in,
             float* outputs, int n_out) {
    const int n = prog.n_nodes;
    if (n == 0) {
        for (int k = 0; k < n_out; k++) outputs[k] = 0.0f;
        return;
    }

    bool live[Program::MAX_NODES];
    compute_live(prog, live);

    // One value slot per node; zero-initialised so inactive nodes read as 0.
    RegVal val[Program::MAX_NODES] = {};

    // Fill input terminals from the input array.
    int n_fill = n_in < prog.n_inputs ? n_in : prog.n_inputs;
    for (int i = 0; i < n_fill; i++)
        val[i].f = inputs[i];

    // Evaluate function nodes in index order.  Because src1/src2 < node_index
    // this is a valid topological order.
    for (int i = prog.n_inputs; i < n; i++) {
        if (!live[i]) continue;

        const Node& nd = prog.nodes[i];
        const RegVal& a = val[nd.src1];
        RegVal imm_b;
        if (nd.src2 == Program::IMM_SRC) imm_b.i = nd.lit.i;
        const RegVal& b = (nd.src2 == Program::IMM_SRC) ? imm_b : val[nd.src2];
        RegVal& d = val[i];

        switch (nd.op) {
        case Op::IADD:  d.i = a.i + b.i;                      break;
        case Op::ISUB:  d.i = a.i - b.i;                      break;
        case Op::IMUL:  d.i = a.i * b.i;                      break;
        case Op::FADD:  d.f = a.f + b.f;                      break;
        case Op::FSUB:  d.f = a.f - b.f;                      break;
        case Op::FMUL:  d.f = a.f * b.f;                      break;
        case Op::BAND:  d.u = a.u & b.u;                      break;
        case Op::BOR:   d.u = a.u | b.u;                      break;
        case Op::BXOR:  d.u = a.u ^ b.u;                      break;
        case Op::LSHL:  d.u = a.u << (b.u & 31u);             break;
        case Op::LSHR:  d.u = a.u >> (b.u & 31u);             break;
        case Op::ASHL:  d.i = a.i << (int)(b.u & 31u);        break;
        case Op::ASHR:  d.i = a.i >> (int)(b.u & 31u);        break;
        case Op::ILT:   d.i = a.i <  b.i ? 1 : 0;            break;
        case Op::IEQ:   d.i = a.i == b.i ? 1 : 0;            break;
        case Op::ULT:   d.i = a.u <  b.u ? 1 : 0;            break;
        case Op::UEQ:   d.i = a.u == b.u ? 1 : 0;            break;
        case Op::FLT:   d.i = a.f <  b.f ? 1 : 0;            break;
        case Op::FEQ:   d.i = a.f == b.f ? 1 : 0;            break;
        case Op::BNOT:  d.u = ~a.u;                           break;
        case Op::LNOT:  d.i = a.i == 0 ? 1 : 0;              break;
        case Op::INEG:  d.i = -a.i;                           break;
        case Op::FNEG:  d.f = -a.f;                           break;
        case Op::ITF:   d.u = a.u;                            break;
        case Op::FTI:   d.u = a.u;                            break;
        case Op::LOADI: d.i = nd.lit.i;                       break;
        case Op::LOADF: d.f = nd.lit.f;                       break;
        case Op::MOV:   d   = a;                              break;
        default: break;
        }
    }

    for (int k = 0; k < n_out; k++)
        outputs[k] = val[prog.output_node].f;
}
