#pragma once
#include "program.hpp"
#include "hardening.hpp"
#include "speciation.hpp"
#include "fitness.hpp"
#include <limits>
#include <random>
#include <vector>

struct Individual {
    Program  prog;
    double   fit         = std::numeric_limits<double>::max();
    Hardness hardness    = {};
    float    case_err[N_CASES] = {};
};

struct Population {
    static constexpr int    SIZE            = 128;
    static constexpr int    TOURNAMENT      = 4;
    static constexpr double MUT_RATE        = 0.7;
    static constexpr int    HARDEN_INTERVAL = 100;

    // NEAT speciation parameters
    static constexpr double COMPAT_THRESH   = 2.0;  // mean |output diff| threshold for same species
    static constexpr int    MAX_SPECIES     = 16;   // hard cap — join closest species when full
    static constexpr int    STAG_LIMIT      = 100;  // gens without improvement → species culled

    Individual           indivs[SIZE];
    int                  generation = 0;
    std::vector<Species> species;

    void init     (std::mt19937& rng);
    void step     (std::mt19937& rng);
    void sort_pop ();
    void assign_species();
    int  select_in_species(const Species& s, std::mt19937& rng) const;

    const Individual& best() const { return indivs[0]; }
};
