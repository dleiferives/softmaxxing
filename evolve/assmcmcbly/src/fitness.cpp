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

double fitness_and_cases(const Program& prog,
                         float case_err[N_CASES],
                         const ProblemDef& problem,
                         const std::vector<float>& test_inputs,
                         bool penalize_length) {
#ifdef USE_JIT
    JitProgram jit = jit_compile(prog);
#endif
    double err    = 0.0;
    float out_min =  std::numeric_limits<float>::infinity();
    float out_max = -std::numeric_limits<float>::infinity();

    const int ni = problem.n_inputs;

    for (int i = 0; i < N_CASES; i++) {
        const float* xs = &test_inputs[i * ni];

#ifdef USE_JIT
        float got = jit.fn(xs[0]);
#else
        float out_buf[1] = {};
        execute(prog, xs, ni, out_buf, problem.n_outputs);
        float got = out_buf[0];
#endif

        if (!std::isfinite(got)) {
            case_err[i] = 1e6f;
            err += 1e6;
            continue;
        }

        if (got < out_min) out_min = got;
        if (got > out_max) out_max = got;

        float target_buf[1] = {};
        problem.oracle(xs, target_buf);
        float target = target_buf[0];

        float re     = (got - target) / target;
        case_err[i]  = re * re;
        err         += case_err[i];
    }

    double msre = err / N_CASES;

    if (std::isfinite(out_min) && out_max - out_min < 0.01f)
        msre += 10.0;

    if (penalize_length) {
        bool live[Program::MAX_NODES];
        int nlive = compute_live(prog, live);
        msre += msre * 0.01 * nlive;
    }
    return msre;
}

double fitness(const Program& prog,
               const ProblemDef& problem,
               const std::vector<float>& test_inputs) {
    float case_err[N_CASES];
    return fitness_and_cases(prog, case_err, problem, test_inputs);
}
