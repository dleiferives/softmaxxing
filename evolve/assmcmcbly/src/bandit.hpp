#pragma once
#include <random>
#include <cmath>

// Dynamic Multi-Armed Bandit for adaptive mutation operator selection.
// Uses UCB1 for arm selection.  reset() is called externally on landscape
// shifts (curriculum advance, hot-burst trigger, best-fitness improvement).
struct Bandit {
    static constexpr int    N_ARMS = 9;   // one per mutation operator kind 0–8
    static constexpr double UCB_C  = 2.0; // exploration constant

    struct Arm {
        double reward_sum = 0.0;
        int    pulls      = 0;
    } arms[N_ARMS];

    int total_pulls = 0;

    // Select an arm via UCB1.  Untried arms are played first in order.
    int  select(std::mt19937& rng) const;

    // Record outcome: reward = max(0, parent_fit - child_fit).
    void update(int arm, double reward);

    // Hard reset: clear all statistics (called on landscape shift).
    void reset();
};
