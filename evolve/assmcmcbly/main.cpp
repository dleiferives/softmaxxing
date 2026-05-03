#include "population.hpp"
#include "execute.hpp"
#include "print.hpp"
#include <iostream>
#include <random>
#include <cmath>

int main() {
    std::mt19937 rng(std::random_device{}());

    std::cout << "Initialising population (size=" << Population::SIZE << ")...\n";

    static Population pop;
    pop.init(rng);

    for (int gen = 1; ; gen++) {
        pop.step(rng);

        if (gen % 100 == 0) {
            const auto& b = pop.best();
            std::cout << "gen=" << gen
                      << "  fit=" << b.fit
                      << "  chroms=" << int(b.prog.num_chroms)
                      << "  instrs=" << b.prog.num_instrs
                      << "  species=" << pop.species.size()
                      << "  gstag=" << pop.global_stagnation
                      << (pop.hot_burst_remaining > 0 ? "  HOT" : "")
                      << "\n";
        }

        if (gen % 10000 == 0) {
            std::cout << "\n--- gen " << gen << " best ---\n";
            print_program(pop.best().prog);
            std::cout << "Sample outputs:\n";
            for (float x : {0.25f, 1.0f, 4.0f, 9.0f, 16.0f, 100.0f}) {
                float got    = execute(pop.best().prog, x);
                float target = 1.0f / std::sqrt(x);
                float err    = std::abs(got - target) / target * 100.0f;
                std::cout << "  x=" << x
                          << "  got=" << got
                          << "  target=" << target
                          << "  err=" << err << "%\n";
            }
            std::cout << "\n";
        }
    }
}
