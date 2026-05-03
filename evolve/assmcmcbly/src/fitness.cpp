#include "fitness.hpp"
#include "dag.hpp"
#include <cmath>
#include <limits>
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
    float out_min =  std::numeric_limits<float>::infinity();
    float out_max = -std::numeric_limits<float>::infinity();

    for (float x : TEST_INPUTS) {
#ifdef USE_JIT
        float got = jit.fn(x);
#else
        float got = execute(prog, x);
#endif
        float target = 1.0f / std::sqrt(x);
        if (!std::isfinite(got)) { err += 1e6; continue; }

        if (got < out_min) out_min = got;
        if (got > out_max) out_max = got;

        double re = (double(got) - double(target)) / double(target);
        err += re * re;
    }
    double msre = err / TEST_INPUTS.size();

    // Penalise constant-output programs that ignore x.
    // 1/sqrt(x) spans [0.1, 10] over the test range; any program that produces
    // a non-trivially x-dependent output will have range >> 0.01.
    if (std::isfinite(out_min) && out_max - out_min < 0.01f) {
        msre += 10.0;
    }

    return msre;
}
