#include "src/population.hpp"
#include "src/execute.hpp"
#include "src/print.hpp"
#include "src/problem.hpp"
#include "src/codegen.hpp"
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <limits>

// ── Tee streambuf: forwards everything written to cout into a log file too ───

class TeeStreambuf : public std::streambuf {
    std::streambuf* primary;
    std::streambuf* secondary;
public:
    TeeStreambuf(std::streambuf* a, std::streambuf* b) : primary(a), secondary(b) {}
    int overflow(int c) override {
        if (c == EOF) return !EOF;
        if (primary->sputc(char(c)) == EOF)   return EOF;
        if (secondary->sputc(char(c)) == EOF) return EOF;
        return c;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        primary->sputn(s, n);
        secondary->sputn(s, n);
        return n;
    }
};

int main() {
    // ── Run directory (runs/<timestamp>/) ─────────────────────────────────────
    std::time_t now = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&now));
    std::string run_dir = std::string("runs/") + ts;
    std::filesystem::create_directories(run_dir);

    // ── Tee all cout output into log.txt ──────────────────────────────────────
    std::ofstream log_file(run_dir + "/log.txt");
    TeeStreambuf tee(std::cout.rdbuf(), log_file.rdbuf());
    std::cout.rdbuf(&tee);

    // ── Problem definition ────────────────────────────────────────────────────
    ProblemDef problem;
    problem.n_inputs  = 1;
    problem.n_outputs = 1;
    problem.oracle    = [](const float* in, float* out) {
        out[0] = 1.0f / std::sqrt(in[0]);
    };

    InputSpec x0;
    x0.range          = {1e-30f, 0.999999f};
    x0.use_curriculum = false;
    x0.curriculum     = {
        { 0.25f,   4.0f    },
        { 0.0625f, 16.0f   },
        { 0.01f,   100.0f  },
        { 0.001f,  1000.0f },
        { 1e-4f,   1e4f    },
    };
    problem.inputs.push_back(x0);
    problem.curriculum_advance_thresh = 0.0001;
    finalize_problem(problem);

    // ── Run ───────────────────────────────────────────────────────────────────
    std::mt19937 rng(std::random_device{}());

    std::cout << "run_dir=" << run_dir << "\n";
    std::cout << "Initialising population (size=" << Population::SIZE << ")...\n";

    static Population pop;
    pop.init(rng, problem);

    double last_emitted_fit = std::numeric_limits<double>::max();

    for (int gen = 1; ; gen++) {
        pop.step(rng);

        const auto& best = pop.best();

        // Emit a new solution file whenever best fitness improves meaningfully.
        if (best.fit < last_emitted_fit * 0.999) {
            last_emitted_fit = best.fit;
            emit_solution(best.prog, gen, best.fit, run_dir);
        }

        if (gen % 100 == 0) {
            std::cout << "gen=" << gen
                      << "  fit=" << best.fit
                      << "  instrs=" << best.prog.num_instrs
                      << "  stage=" << pop.curriculum_stage
                      << "  gstag=" << pop.global_stagnation
                      << (pop.global_stagnation >= Population::GSTAG_ENABLE_CACHE
                          ? "  novel=" + std::to_string(pop.novelty_seen.size())
                            + "  frontier=" + std::to_string(pop.frontier_queue.size())
                          : "")
                      << (pop.hot_burst_remaining > 0 ? "  HOT" : "")
                      << "\n";
        }

        if (gen % 10000 == 0) {
            std::cout << "\n--- gen " << gen << " best ---\n";
            print_program(best.prog);
            std::cout << "Sample outputs:\n";
            float in_buf[1], out_buf[1], tgt_buf[1];
            for (float x : {0.25f, 1.0f, 4.0f, 9.0f, 16.0f, 100.0f}) {
                in_buf[0] = x;
                execute(best.prog, in_buf, 1, out_buf, 1);
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
