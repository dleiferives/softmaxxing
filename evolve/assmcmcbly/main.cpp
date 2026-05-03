#include "src/population.hpp"
#include "src/execute.hpp"
#include "src/print.hpp"
#include "src/problem.hpp"
#include <iostream>
#include <random>
#include <cmath>

int main() {
    // --- Define the problem ---
    ProblemDef problem;
    problem.n_inputs  = 1;
    problem.n_outputs = 1;
    problem.oracle    = [](const float* in, float* out) {
        out[0] = 1.0f / std::sqrt(in[0]);
    };

    InputSpec x0;
    x0.range          = {1e-4f, 1e4f};
    x0.use_curriculum = false;
    x0.curriculum     = {
        { 0.25f,   4.0f    },
        { 0.0625f, 16.0f   },
        { 0.01f,   100.0f  },
        { 0.001f,  1000.0f },
        { 1e-4f,   1e4f    },
    };
    problem.inputs.push_back(x0);
    problem.curriculum_advance_thresh = 0.01;
    finalize_problem(problem);  // auto-fills behavior_samples

    // --- Run ---
    std::mt19937 rng(std::random_device{}());

    std::cout << "Initialising population (size=" << Population::SIZE << ")...\n";

    static Population pop;
    pop.init(rng, problem);

    for (int gen = 1; ; gen++) {
        pop.step(rng);

        if (gen % 100 == 0) {
            const auto& b = pop.best();
            std::cout << "gen=" << gen
                      << "  fit=" << b.fit
                      << "  chroms=" << int(b.prog.num_chroms)
                      << "  instrs=" << b.prog.num_instrs
                      << "  species=" << pop.species.size()
                      << "  stage=" << pop.curriculum_stage
                      << "  gstag=" << pop.global_stagnation
                      << (pop.hot_burst_remaining > 0 ? "  HOT" : "")
                      << "\n";
        }

        if (gen % 10000 == 0) {
            std::cout << "\n--- gen " << gen << " best ---\n";
            print_program(pop.best().prog);
            std::cout << "Sample outputs:\n";
            float in_buf[1], out_buf[1], tgt_buf[1];
            for (float x : {0.25f, 1.0f, 4.0f, 9.0f, 16.0f, 100.0f}) {
                in_buf[0] = x;
                execute(pop.best().prog, in_buf, 1, out_buf, 1);
                problem.oracle(in_buf, tgt_buf);
                float err = std::abs(out_buf[0] - tgt_buf[0]) / tgt_buf[0] * 100.0f;
                std::cout << "  x=" << x
                          << "  got=" << out_buf[0]
                          << "  target=" << tgt_buf[0]
                          << "  err=" << err << "%\n";
            }
            std::cout << "\n";
        }
    }
}
