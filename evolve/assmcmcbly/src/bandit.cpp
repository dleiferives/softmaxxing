#include "bandit.hpp"
#include <limits>

int Bandit::select(std::mt19937& /*rng*/) const {
    // Play each arm at least once before exploiting
    for (int i = 0; i < N_ARMS; i++)
        if (arms[i].pulls == 0) return i;

    // UCB1
    double best_score = -std::numeric_limits<double>::infinity();
    int    best_arm   = 0;
    double log_total  = std::log(double(total_pulls));

    for (int i = 0; i < N_ARMS; i++) {
        double mean  = arms[i].reward_sum / arms[i].pulls;
        double bonus = UCB_C * std::sqrt(log_total / arms[i].pulls);
        double score = mean + bonus;
        if (score > best_score) { best_score = score; best_arm = i; }
    }
    return best_arm;
}

void Bandit::update(int arm, double reward) {
    arms[arm].reward_sum += reward;
    arms[arm].pulls++;
    total_pulls++;
}

void Bandit::reset() {
    for (int i = 0; i < N_ARMS; i++) {
        arms[i].reward_sum = 0.0;
        arms[i].pulls      = 0;
    }
    total_pulls = 0;
}
