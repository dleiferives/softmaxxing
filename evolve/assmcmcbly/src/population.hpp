#pragma once
#include "program.hpp"
#include "hardening.hpp"
#include "speciation.hpp"
#include "fitness.hpp"
#include <limits>
#include <list>
#include <random>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Individual {
    Program  prog;
    double   fit               = std::numeric_limits<double>::max();
    Hardness hardness          = {};
    float    case_err[N_CASES] = {};
};

struct Population {
    static constexpr int    SIZE              = 128;
    static constexpr int    N_ISLANDS         = 4;
    static constexpr int    ISLAND_SIZE       = SIZE / N_ISLANDS;  // 32
    static constexpr int    MIGRATE_INTERVAL  = 100;
    static constexpr int    ISLAND_STAG_LIMIT = 600;
    static constexpr double MUT_RATE          = 0.7;
    static constexpr int    HARDEN_INTERVAL   = 100;

    // Global stagnation → hot-burst exploration
    static constexpr int   GSTAG_HOT_TRIGGER  = 1000;
    static constexpr int   GSTAG_ENABLE_CACHE = 3000;
    static constexpr int   HOT_BURST_DURATION = 500;
    static constexpr float HOT_EPSILON_SCALE  = 50.0f;

    // LRU eval cache — avoids re-evaluating identical programs
    static constexpr int EVAL_LRU_SIZE = 65536;

    // Frontier BFS queue target size with hysteresis band [90%, 110%]
    static constexpr int FRONTIER_TARGET = 1024;

    struct Island {
        Individual indivs[ISLAND_SIZE];
        double     best_fit   = std::numeric_limits<double>::max();
        int        stagnation = 0;
    };

    struct EvalEntry {
        double fit;
        float  case_err[N_CASES];
    };

    using EvalKey     = uint64_t;
    using EvalLRUList = std::list<std::pair<EvalKey, EvalEntry>>;
    using EvalLRUMap  = std::unordered_map<EvalKey, EvalLRUList::iterator>;

    Island                       islands[N_ISLANDS];
    int                          generation          = 0;
    double                       global_best_fit     = std::numeric_limits<double>::max();
    int                          global_stagnation   = 0;
    int                          hot_burst_remaining = 0;
    int                          curriculum_stage    = 0;
    std::unordered_set<uint32_t>               novelty_seen;
    std::deque<std::pair<uint32_t, Program>>   frontier_queue;  // (behavior_hash, prog) — unevaluated candidates
    std::unordered_set<uint32_t>               frontier_hashes; // dedup: hashes currently in frontier_queue
    bool                                       frontier_boost_on = true;
    uint64_t                                   total_mutations   = 0;
    EvalLRUList                  eval_lru_list;
    EvalLRUMap                   eval_lru_map;

    const ProblemDef*  problem             = nullptr;
    std::vector<float> current_test_inputs;

    void init            (std::mt19937& rng, const ProblemDef& p);
    void step            (std::mt19937& rng);
    void sort_island     (Island& isl);
    void migrate         (std::mt19937& rng);
    int  select_in_island(const Island& isl, std::mt19937& rng) const;

    const Individual& best() const;

    static EvalKey program_hash(const Program& p);
    bool           eval_lru_get(EvalKey k, EvalEntry& out);
    void           eval_lru_put(EvalKey k, const EvalEntry& e);
    void           eval_individual(Individual& ni, bool penalize_length = true);
};
