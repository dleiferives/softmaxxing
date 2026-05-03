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

static std::vector<float> make_test_inputs(float lo, float hi) {
    std::vector<float> v(N_CASES);
    for (int i = 0; i < N_CASES; i++)
        v[i] = lo * std::pow(hi / lo, float(i) / (N_CASES - 1));
    return v;
}

std::vector<float> TEST_INPUTS = make_test_inputs(0.25f, 4.0f);  // curriculum stage 0

void set_curriculum_range(float x_lo, float x_hi) {
    TEST_INPUTS = make_test_inputs(x_lo, x_hi);
}

double fitness_and_cases(const Program& prog, float case_err[N_CASES]) {
#ifdef USE_JIT
    JitProgram jit = jit_compile(prog);
#endif
    double err    = 0.0;
    float out_min =  std::numeric_limits<float>::infinity();
    float out_max = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < N_CASES; i++) {
        float x = TEST_INPUTS[i];
#ifdef USE_JIT
        float got = jit.fn(x);
#else
        float got = execute(prog, x);
#endif
        float target = 1.0f / std::sqrt(x);
        if (!std::isfinite(got)) {
            case_err[i] = 1e6f;
            err += 1e6;
            continue;
        }

        if (got < out_min) out_min = got;
        if (got > out_max) out_max = got;

        float re     = (got - target) / target;
        case_err[i]  = re * re;
        err         += case_err[i];
    }

    double msre = err / N_CASES;

    // Penalise constant-output programs that ignore x.
    if (std::isfinite(out_min) && out_max - out_min < 0.01f)
        msre += 10.0;

    return msre;
}

double fitness(const Program& prog) {
    float case_err[N_CASES];
    return fitness_and_cases(prog, case_err);
}
