#include "fitness.hpp"
#include "dag.hpp"
#include <cmath>
#include <vector>

#ifdef USE_JIT
#  include "jit.hpp"
#else
#  include "execute.hpp"
#endif

static std::vector<float> make_test_inputs() {
    std::vector<float> v(100);
    for (int i = 0; i < 100; i++)
        v[i] = std::pow(10.0f, -2.0f + 4.0f * float(i) / 99.0f);
    return v;
}

const std::vector<float> TEST_INPUTS = make_test_inputs();

double fitness(const Program& prog) {
#ifdef USE_JIT
    JitProgram jit = jit_compile(prog);
#endif
    double err = 0.0;
    for (float x : TEST_INPUTS) {
#ifdef USE_JIT
        float got = jit.fn(x);
#else
        float got = execute(prog, x);
#endif
        float target = 1.0f / std::sqrt(x);
        if (!std::isfinite(got)) { err += 1e6; continue; }
        double re = (double(got) - double(target)) / double(target);
        err += re * re;
    }
    double msre = err / TEST_INPUTS.size();

    // Penalise dead instructions: programs bloated with unreachable code score worse.
    bool live[Program::MAX_INSTRS];
    int live_count = compute_dag(prog, live);
    int dead_count = prog.num_instrs - live_count;
    if (prog.num_instrs > 0) {
        double dead_ratio = double(dead_count) / double(prog.num_instrs);
        msre = msre * (1.0 + 0.1 * dead_ratio);
    }

    return msre;
}
