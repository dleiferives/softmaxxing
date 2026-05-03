#pragma once
#include "program.hpp"
#include "hardening.hpp"
#include "speciation.hpp"
#include "fitness.hpp"
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

struct Individual {
    Program  prog;
    double   fit         = std::numeric_limits<double>::max();
    Hardness hardness    = {};
    float    case_err[N_CASES] = {};
};

struct CurriculumStage { float lo, hi; };

struct Population {
    static constexpr int    SIZE            = 128;
    static constexpr int    TOURNAMENT      = 4;
    static constexpr double MUT_RATE        = 0.7;
    static constexpr int    HARDEN_INTERVAL = 100;

    // NEAT speciation parameters
    static constexpr double COMPAT_THRESH   = 2.0;
    static constexpr int    MAX_SPECIES     = 16;
    static constexpr int    STAG_LIMIT      = 100;

    // Global stagnation → hot-burst exploration
    static constexpr int   GSTAG_HOT_TRIGGER    = 1000;
    static constexpr int   GSTAG_ENABLE_CACHE   = 20000; // enable novelty cache after this many stagnant gens
    static constexpr int   HOT_BURST_DURATION = 500;
    static constexpr float HOT_EPSILON_SCALE  = 50.0f;
    static constexpr int   HOT_STAG_MULTIPLIER = 10;

    // Curriculum: progressively widen the test input range
    static constexpr CurriculumStage CURRICULUM[] = {
        { 0.25f,   4.0f    },   // stage 0 — starting range
        { 0.0625f, 16.0f   },   // stage 1
        { 0.01f,   100.0f  },   // stage 2
        { 0.001f,  1000.0f },   // stage 3
        { 1e-4f,   1e4f    },   // stage 4 — final
    };
    static constexpr int    N_CURRICULUM              = 5;
    static constexpr double CURRICULUM_ADVANCE_THRESH = 0.01;

    Individual                    indivs[SIZE];
    int                           generation          = 0;
    double                        global_best_fit     = std::numeric_limits<double>::max();
    int                           global_stagnation   = 0;
    int                           hot_burst_remaining = 0;
    int                           curriculum_stage    = 0;
    std::unordered_set<uint32_t>  eval_cache;          // behavioral hashes seen since last improvement
    std::vector<Species>          species;

    void init     (std::mt19937& rng);
    void step     (std::mt19937& rng);
    void sort_pop ();
    void assign_species();
    int  select_in_species(const Species& s, std::mt19937& rng) const;

    const Individual& best() const { return indivs[0]; }
};
