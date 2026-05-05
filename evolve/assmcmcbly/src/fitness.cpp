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
	auto start = std::chrono::steady_clock::now();
#ifdef USE_JIT
    JitProgram jit = jit_compile(prog);
#endif
    float err    = 0.0;
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
	// try mre
        case_err[i]  = re * re;
        //case_err[i]  = fabs(re); //* re;
        err         += case_err[i];
    }
    double tmp_err = err;

    double msre = tmp_err / N_CASES;

    if (std::isfinite(out_min) && out_max - out_min < 0.01f)
        msre += 10.0;
	
    msre += (msre * prog.num_instrs) * 0.001;
        auto end = std::chrono::steady_clock::now();
        fitness_duration += end-start;
        fitness_calls++;
    return msre;
}

double fitness(const Program& prog,
               const ProblemDef& problem,
               const std::vector<float>& test_inputs) {
    float case_err[N_CASES];
    return fitness_and_cases(prog, case_err, problem, test_inputs);
}
