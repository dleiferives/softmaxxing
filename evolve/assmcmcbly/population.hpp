#pragma once
#include "program.hpp"
#include "hardening.hpp"
#include <limits>
#include <random>

struct Individual {
    Program  prog;
    double   fit      = std::numeric_limits<double>::max();
    Hardness hardness = {};
};

struct Population {
    static constexpr int    SIZE             = 128;
    static constexpr int    ELITE            = 8;
    static constexpr int    TOURNAMENT       = 5;
    static constexpr double MUT_RATE         = 0.7;
    static constexpr int    HARDEN_INTERVAL  = 100; // recompute hardness every N generations

    Individual indivs[SIZE];
    int        generation = 0;

    void init    (std::mt19937& rng);
    void step    (std::mt19937& rng);
    void sort_pop();
    int  select  (std::mt19937& rng) const;

    const Individual& best() const { return indivs[0]; }
};
