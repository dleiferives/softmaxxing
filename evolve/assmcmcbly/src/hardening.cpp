#include "hardening.hpp"
#include "fitness.hpp"
#include <algorithm>

void Hardness::recompute(const Program& prog, double base_fitness,
                          const ProblemDef& problem,
                          const std::vector<float>& test_inputs) {
    // Ablate each function node by setting it to LOADF 0.0 and measuring
    // how much fitness degrades.  Higher delta → node is more critical.
    for (int i = prog.n_inputs; i < prog.n_nodes; i++) {
        Program tmp = prog;
        tmp.nodes[i].op    = Op::LOADF;
        tmp.nodes[i].lit.f = 0.0f;
        double ablated = fitness(tmp, problem, test_inputs);
        scores[i] = float(std::max(0.0, ablated - base_fitness));
    }
    // Input terminals and unused slots get zero hardness.
    for (int i = 0; i < prog.n_inputs; i++)          scores[i] = 0.0f;
    for (int i = prog.n_nodes; i < Program::MAX_NODES; i++) scores[i] = 0.0f;
}
