#pragma once
#include "program.hpp"
#include "hardening.hpp"
#include "speciation.hpp"
#include "fitness.hpp"
#include "bandit.hpp"
#include <limits>
#include <list>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Individual {
    Program  prog;
    double   fit              = std::numeric_limits<double>::max();
    Hardness hardness         = {};
    float    case_err[N_CASES] = {};
};

struct Population {
    static constexpr int    SIZE            = 128;
    static constexpr int    TOURNAMENT      = 4;
    static constexpr double MUT_RATE        = 0.7;
    static constexpr int    HARDEN_INTERVAL = 100;

    // NEAT speciation parameters
    static constexpr double COMPAT_THRESH   = 2.0;
    static constexpr int    MAX_SPECIES     = 16;
    static constexpr int    STAG_LIMIT      = 800;

    // Global stagnation → hot-burst exploration
    static constexpr int   GSTAG_HOT_TRIGGER   = 1000;
    static constexpr int   GSTAG_ENABLE_CACHE  = 3000;
    static constexpr int   HOT_BURST_DURATION  = 500;
    static constexpr float HOT_EPSILON_SCALE   = 50.0f;
    static constexpr int   HOT_STAG_MULTIPLIER = 1;

    // LRU eval cache
    static constexpr int EVAL_LRU_SIZE = 16384;

    struct EvalEntry {
        double fit;
        float  case_err[N_CASES];
    };

    using EvalKey     = uint64_t;
    using EvalLRUList = std::list<std::pair<EvalKey, EvalEntry>>;
    using EvalLRUMap  = std::unordered_map<EvalKey, EvalLRUList::iterator>;

    Individual                   indivs[SIZE];
    int                          generation          = 0;
    double                       global_best_fit     = std::numeric_limits<double>::max();
    int                          global_stagnation   = 0;
    int                          hot_burst_remaining = 0;
    int                          curriculum_stage    = 0;
    std::unordered_set<uint32_t> novelty_seen;
    EvalLRUList                  eval_lru_list;
    EvalLRUMap                   eval_lru_map;
    std::vector<Species>         species;
    Bandit                       mutation_bandit;

    const ProblemDef*  problem             = nullptr;
    std::vector<float> current_test_inputs;

    void init     (std::mt19937& rng, const ProblemDef& p);
    void step     (std::mt19937& rng);
    void sort_pop ();
    void assign_species();
    int  select_in_species(const Species& s, std::mt19937& rng) const;

    const Individual& best() const { return indivs[0]; }

    static EvalKey program_hash(const Program& p);
    bool           eval_lru_get(EvalKey k, EvalEntry& out);
    void           eval_lru_put(EvalKey k, const EvalEntry& e);

    void eval_individual(Individual& ni, bool penalize_length = true);
};
