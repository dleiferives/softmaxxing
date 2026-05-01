#include "population.hpp"
#include "fitness.hpp"
#include "mutate.hpp"
#include "random.hpp"
#include "hardening.hpp"
#include <algorithm>
#include <cstring>

void Population::sort_pop() {
    std::sort(indivs, indivs + SIZE,
              [](const Individual& a, const Individual& b){ return a.fit < b.fit; });
}

void Population::init(std::mt19937& rng) {
    for (auto& ind : indivs) {
        ind.prog = random_program(rng);
        ind.fit  = fitness(ind.prog);
    }
    sort_pop();
}

int Population::select(std::mt19937& rng) const {
    std::uniform_int_distribution<int> d(0, SIZE - 1);
    int best = d(rng);
    for (int i = 1; i < TOURNAMENT; i++) {
        int c = d(rng);
        if (indivs[c].fit < indivs[best].fit) best = c;
    }
    return best;
}

void Population::step(std::mt19937& rng) {
    generation++;

    // Recompute hardness for elite individuals periodically
    if (generation % HARDEN_INTERVAL == 0) {
        for (int i = 0; i < ELITE; i++)
            indivs[i].hardness.recompute(indivs[i].prog, indivs[i].fit);
    }

    std::uniform_real_distribution<double> coin(0.0, 1.0);

    Individual next[SIZE];
    for (int i = 0; i < ELITE; i++) next[i] = indivs[i];

    for (int i = ELITE; i < SIZE; i++) {
        int     p = select(rng);
        Program child;
        if (coin(rng) < MUT_RATE) {
            child = mutate(indivs[p].prog, indivs[p].hardness, rng);
        } else {
            int q = select(rng);
            child = crossover(indivs[p].prog, indivs[q].prog, rng);
        }
        double f = fitness(child);
        next[i]  = { child, f, {} };  // hardness reset for new individuals
    }

    memcpy(indivs, next, sizeof(indivs));
    sort_pop();
}
