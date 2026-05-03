#include "population.hpp"
#include "fitness.hpp"
#include "mutate.hpp"
#include "random.hpp"
#include "hardening.hpp"
#include "innovation.hpp"
#include <algorithm>
#include <cstring>
#include <limits>

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

    double seed_fit = fitness(seed);
    for (auto& ind : indivs) {
        ind.prog = seed;
        ind.fit  = seed_fit;
    }
    sort_pop();

    species.clear();
    Species s0;
    s0.rep      = seed;
    s0.best_fit = seed_fit;
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
    std::uniform_int_distribution<int> d(0, n - 1);
    int best  = s.members[d(rng)];
    int tries = std::min(TOURNAMENT, n);
    for (int i = 1; i < tries; i++) {
        int c = s.members[d(rng)];
        double fc = indivs[c].fit, fb = indivs[best].fit;
        if (fc < fb * 0.999) {
            best = c;  // clearly better fitness
        } else if (fc < fb * 1.001 &&
                   indivs[c].prog.num_instrs < indivs[best].prog.num_instrs) {
            best = c;  // fitness tied — prefer fewer instructions (parsimony)
        }
    }
    return best;
}

void Population::step(std::mt19937& rng) {
    generation++;

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

        if (s.stagnation >= STAG_LIMIT) continue;  // stagnant: champion survives but no offspring

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
            double f = fitness(child);
            next[next_count++] = { child, f, {} };
        }
    }

    // Fill any rounding remainder (pick random non-stagnant species)
    while (next_count < SIZE) {
        int si = int(std::uniform_int_distribution<int>(0, nsp - 1)(rng));
        const auto& s = species[si];
        if (s.members.empty()) continue;
        int p = select_in_species(s, rng);
        Program child = mutate(indivs[p].prog, indivs[p].hardness, rng);
        next[next_count++] = { child, fitness(child), {} };
    }

    memcpy(indivs, next, sizeof(indivs));
    sort_pop();
}
