#include "dag.hpp"
#include <cstring>

int compute_live(const Program& prog, bool live[Program::MAX_NODES]) {
    const int n = prog.n_nodes;
    memset(live, 0, n * sizeof(bool));
    if (n == 0 || prog.output_node >= n) return 0;

    uint16_t stack[Program::MAX_NODES];
    int top = 0;
    stack[top++] = prog.output_node;

    while (top > 0) {
        uint16_t idx = stack[--top];
        if (live[idx]) continue;
        live[idx] = true;

        // Input terminals have no upstream deps
        if (idx < prog.n_inputs) continue;

        const Node& nd = prog.nodes[idx];

        // Constant terminals have no upstream deps
        if (nd.op == Op::LOADI || nd.op == Op::LOADF) continue;

        // src1 is always used for function nodes
        if (!live[nd.src1]) stack[top++] = nd.src1;

        // src2 used for binary ops with a register operand
        bool is_unary = (nd.op == Op::BNOT || nd.op == Op::LNOT ||
                         nd.op == Op::INEG || nd.op == Op::FNEG ||
                         nd.op == Op::ITF  || nd.op == Op::FTI  ||
                         nd.op == Op::MOV);
        if (!is_unary && nd.src2 != Program::IMM_SRC && !live[nd.src2])
            stack[top++] = nd.src2;
    }

    int count = 0;
    for (int i = 0; i < n; i++) count += live[i];
    return count;
}
