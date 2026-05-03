#include "population.hpp"
#include "fitness.hpp"
#include "mutate.hpp"
#include "random.hpp"
#include "hardening.hpp"
#include "innovation.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>

void Population::sort_pop() {
    std::sort(indivs, indivs + SIZE,
              [](const Individual& a, const Individual& b){ return a.fit < b.fit; });
}

// NEAT minimal initialisation: all individuals share the same seed genome so
// innovation numbers are meaningful from generation 1.  Field mutations and MCMC
// quickly improve the constant; structural growth mutations then create new species.
void Population::init(std::mt19937& rng) {
    (void)rng;

    Program seed = {};
    Instr&  si   = seed.instrs[0];
    si.op    = Op::LOADI;
    si.dst   = 0;
    si.src1  = 0;
    si.src2  = 0;
    si.lit.i = 0;
    si.innov = next_innovation();   // global innovation #1 shared by all
    seed.chrom_lens[0] = 1;
    seed.num_chroms    = 1;
    seed.num_instrs    = 1;

    Individual seed_ind;
    seed_ind.prog = seed;
    seed_ind.fit  = fitness_and_cases(seed, seed_ind.case_err);
    for (auto& ind : indivs)
        ind = seed_ind;
    sort_pop();

    species.clear();
    Species s0;
    s0.rep      = seed;
    s0.best_fit = seed_ind.fit;
    compute_fingerprint(seed, s0.rep_fp);
    for (int i = 0; i < SIZE; i++) s0.members.push_back(i);
    species.push_back(std::move(s0));
}

void Population::assign_species() {
    for (auto& s : species) s.members.clear();

    // Precompute fingerprint for each individual once
    float ind_fp[SIZE][N_BEH];
    for (int i = 0; i < SIZE; i++)
        compute_fingerprint(indivs[i].prog, ind_fp[i]);

    for (int i = 0; i < SIZE; i++) {
        double best_dist = std::numeric_limits<double>::max();
        int    best_si   = -1;
        for (int si = 0; si < int(species.size()); si++) {
            double d = behavioral_distance(ind_fp[i], species[si].rep_fp);
            if (d < best_dist) { best_dist = d; best_si = si; }
        }

        bool can_create = (int(species.size()) < MAX_SPECIES) && (best_dist >= COMPAT_THRESH);
        if (can_create) {
            Species ns;
            ns.rep      = indivs[i].prog;
            ns.best_fit = indivs[i].fit;
            std::copy(ind_fp[i], ind_fp[i] + N_BEH, ns.rep_fp);
            ns.members.push_back(i);
            species.push_back(std::move(ns));
        } else {
            // Join best matching species (either within threshold or closest when capped)
            species[best_si].members.push_back(i);
        }
    }

    species.erase(
        std::remove_if(species.begin(), species.end(),
            [](const Species& s){ return s.members.empty(); }),
        species.end());
}

int Population::select_in_species(const Species& s, std::mt19937& rng) const {
    int n = int(s.members.size());
    if (n == 1) return s.members[0];

    // Epsilon-lexicase: shuffle test cases, filter candidates round by round.
    // A candidate survives a round if its per-case error is within epsilon of
    // the best error seen on that case among the remaining pool.
    int cand[SIZE], n_cand = n;
    for (int i = 0; i < n; i++) cand[i] = s.members[i];

    int cases[N_CASES];
    std::iota(cases, cases + N_CASES, 0);
    std::shuffle(cases, cases + N_CASES, rng);

    int next_cand[SIZE];
    for (int ci : cases) {
        if (n_cand == 1) break;

        float min_err = std::numeric_limits<float>::infinity();
        for (int i = 0; i < n_cand; i++)
            min_err = std::min(min_err, indivs[cand[i]].case_err[ci]);

        float eps_scale = (hot_burst_remaining > 0) ? HOT_EPSILON_SCALE : 1.0f;
        float epsilon   = (min_err * 0.1f + 1e-6f) * eps_scale;

        int n_next = 0;
        for (int i = 0; i < n_cand; i++)
            if (indivs[cand[i]].case_err[ci] <= min_err + epsilon)
                next_cand[n_next++] = cand[i];

        if (n_next > 0) {
            n_cand = n_next;
            std::copy(next_cand, next_cand + n_next, cand);
        }
    }

    return cand[std::uniform_int_distribution<int>(0, n_cand - 1)(rng)];
}

void Population::step(std::mt19937& rng) {
    generation++;

    // Track global stagnation. Counter resets only on genuine improvement;
    // every GSTAG_HOT_TRIGGER gens of continuous stagnation fires a new burst.
    {
        double cur_best = indivs[0].fit;
        if (cur_best < global_best_fit * 0.999) {
            global_best_fit     = cur_best;
            global_stagnation   = 0;
            hot_burst_remaining = 0;
        } else {
            global_stagnation++;
        }
        if (global_stagnation > 0 &&
            global_stagnation % GSTAG_HOT_TRIGGER == 0 &&
            hot_burst_remaining == 0) {
            hot_burst_remaining = HOT_BURST_DURATION;
        }
        if (hot_burst_remaining > 0) hot_burst_remaining--;
    }

    assign_species();

    // Recompute hardness for each species champion periodically
    if (generation % HARDEN_INTERVAL == 0) {
        for (const auto& s : species) {
            if (s.members.empty()) continue;
            int champ = s.members[0];
            for (int idx : s.members)
                if (indivs[idx].fit < indivs[champ].fit) champ = idx;
            indivs[champ].hardness.recompute(indivs[champ].prog, indivs[champ].fit);
        }
    }

    // Update stagnation counters and compute offspring scores
    int nsp = int(species.size());
    std::vector<double> scores(nsp, 0.0);
    double total_score = 0.0;

    for (int si = 0; si < nsp; si++) {
        auto& s = species[si];
        double best = std::numeric_limits<double>::max();
        for (int idx : s.members)
            best = std::min(best, indivs[idx].fit);

        if (best < s.best_fit * 0.999) {
            s.best_fit   = best;
            s.stagnation = 0;
        } else {
            s.stagnation++;
        }

        int eff_stag_limit = (hot_burst_remaining > 0)
                           ? STAG_LIMIT * HOT_STAG_MULTIPLIER
                           : STAG_LIMIT;
        if (s.stagnation >= eff_stag_limit) continue;

        // Fitness sharing: average adjusted fitness = avg(1/fit) / species_size
        // Dividing by size penalises large species and gives small exploratory ones a fair share.
        for (int idx : s.members)
            scores[si] += 1.0 / (indivs[idx].fit + 1e-10);
        scores[si] /= double(s.members.size());
        total_score += scores[si];
    }

    // If every species stagnated, revive the one with the best champion
    if (total_score == 0.0) {
        int best_si = 0;
        double best_fit = std::numeric_limits<double>::max();
        for (int si = 0; si < nsp; si++)
            for (int idx : species[si].members)
                if (indivs[idx].fit < best_fit) { best_fit = indivs[idx].fit; best_si = si; }
        species[best_si].stagnation = 0;
        for (int idx : species[best_si].members)
            scores[best_si] += 1.0 / (indivs[idx].fit + 1e-10);
        scores[best_si] /= double(species[best_si].members.size());
        total_score = scores[best_si];
    }

    // Each species keeps its champion; remaining slots are allocated proportionally
    int champion_slots = nsp;  // one champion per species
    int offspring_slots = SIZE - champion_slots;
    if (offspring_slots < 0) offspring_slots = 0;

    std::vector<int> offspring(nsp, 0);
    int assigned = 0;
    for (int si = 0; si < nsp; si++) {
        offspring[si] = int(scores[si] / total_score * offspring_slots);
        assigned += offspring[si];
    }
    // Remainder to highest-scoring species
    int rem = offspring_slots - assigned;
    if (rem > 0 && nsp > 0) {
        int best_si = 0;
        for (int si = 1; si < nsp; si++)
            if (scores[si] > scores[best_si]) best_si = si;
        offspring[best_si] += rem;
    }

    Individual next[SIZE];
    int next_count = 0;

    std::uniform_real_distribution<double> coin(0.0, 1.0);

    // Species champions survive; update representatives to current champions
    for (auto& s : species) {
        if (next_count >= SIZE || s.members.empty()) continue;
        int champ = s.members[0];
        for (int idx : s.members)
            if (indivs[idx].fit < indivs[champ].fit) champ = idx;
        next[next_count++] = indivs[champ];
        s.rep = indivs[champ].prog;
        compute_fingerprint(s.rep, s.rep_fp);
    }

    // Generate offspring within each species
    for (int si = 0; si < nsp && next_count < SIZE; si++) {
        const auto& s = species[si];
        if (s.members.empty()) continue;
        for (int j = 0; j < offspring[si] && next_count < SIZE; j++) {
            int p = select_in_species(s, rng);
            Program child;
            if (int(s.members.size()) == 1 || coin(rng) < MUT_RATE) {
                child = mutate(indivs[p].prog, indivs[p].hardness, rng);
            } else {
                int q = select_in_species(s, rng);
                child = crossover(indivs[p].prog, indivs[p].fit,
                                  indivs[q].prog, indivs[q].fit, rng);
            }
            Individual& ni = next[next_count++];
            ni.prog     = child;
            ni.hardness = {};
            ni.fit      = fitness_and_cases(child, ni.case_err);
        }
    }

    // Fill any rounding remainder (pick random non-stagnant species)
    while (next_count < SIZE) {
        int si = int(std::uniform_int_distribution<int>(0, nsp - 1)(rng));
        const auto& s = species[si];
        if (s.members.empty()) continue;
        int p = select_in_species(s, rng);
        Program child = mutate(indivs[p].prog, indivs[p].hardness, rng);
        Individual& ni = next[next_count++];
        ni.prog     = child;
        ni.hardness = {};
        ni.fit      = fitness_and_cases(child, ni.case_err);
    }

    memcpy(indivs, next, sizeof(indivs));
    sort_pop();

    // Curriculum advancement: when best fitness crosses the threshold, widen
    // the test range to the next stage and recompute everyone's fitness.
    if (curriculum_stage < N_CURRICULUM - 1 &&
        indivs[0].fit < CURRICULUM_ADVANCE_THRESH) {
        curriculum_stage++;
        const auto& st = CURRICULUM[curriculum_stage];
        set_curriculum_range(st.lo, st.hi);
        for (int i = 0; i < SIZE; i++)
            indivs[i].fit = fitness_and_cases(indivs[i].prog, indivs[i].case_err);
        sort_pop();
        // Reset global stagnation so hot-burst doesn't fire immediately
        global_best_fit     = indivs[0].fit;
        global_stagnation   = 0;
        hot_burst_remaining = 0;
        std::cout << "*** curriculum stage " << curriculum_stage
                  << "  range=[" << st.lo << ", " << st.hi << "]"
                  << "  best_fit=" << indivs[0].fit << "\n";
    }
}
