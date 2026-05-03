#include "population.hpp"
#include "fitness.hpp"
#include "mutate.hpp"
#include "random.hpp"
#include "hardening.hpp"
#include "innovation.hpp"
#include "dag.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>

static uint32_t behavior_hash(const float fp[N_BEH]) {
    uint32_t h = 0x811c9dc5u;
    for (int i = 0; i < N_BEH; i++) {
        uint32_t bits;
        __builtin_memcpy(&bits, &fp[i], sizeof(bits));
        h ^= bits;
        h *= 0x01000193u;
    }
    return h;
}

Population::EvalKey Population::program_hash(const Program& p) {
    uint64_t h = 14695981039346656037ULL;
    auto mix = [&](uint8_t b) { h ^= b; h *= 1099511628211ULL; };
    mix(uint8_t(p.n_nodes));
    mix(uint8_t(p.n_nodes >> 8));
    mix(uint8_t(p.n_inputs));
    mix(uint8_t(p.output_node));
    mix(uint8_t(p.output_node >> 8));
    for (int i = p.n_inputs; i < p.n_nodes; i++) {
        const Node& nd = p.nodes[i];
        mix(uint8_t(nd.op));
        mix(uint8_t(nd.src1)); mix(uint8_t(nd.src1 >> 8));
        mix(uint8_t(nd.src2)); mix(uint8_t(nd.src2 >> 8));
        uint32_t lit; __builtin_memcpy(&lit, &nd.lit, sizeof(lit));
        mix(lit & 0xFF); mix((lit >> 8) & 0xFF);
        mix((lit >> 16) & 0xFF); mix((lit >> 24) & 0xFF);
    }
    return h;
}

bool Population::eval_lru_get(EvalKey k, EvalEntry& out) {
    auto it = eval_lru_map.find(k);
    if (it == eval_lru_map.end()) return false;
    eval_lru_list.splice(eval_lru_list.begin(), eval_lru_list, it->second);
    out = it->second->second;
    return true;
}

void Population::eval_lru_put(EvalKey k, const EvalEntry& e) {
    auto it = eval_lru_map.find(k);
    if (it != eval_lru_map.end()) {
        eval_lru_list.splice(eval_lru_list.begin(), eval_lru_list, it->second);
        it->second->second = e;
        return;
    }
    if (int(eval_lru_map.size()) >= EVAL_LRU_SIZE) {
        auto last = std::prev(eval_lru_list.end());
        eval_lru_map.erase(last->first);
        eval_lru_list.erase(last);
    }
    eval_lru_list.emplace_front(k, e);
    eval_lru_map[k] = eval_lru_list.begin();
}

void Population::eval_individual(Individual& ni, bool penalize_length) {
    EvalKey k = program_hash(ni.prog);
    EvalEntry entry;
    if (penalize_length && eval_lru_get(k, entry)) {
        ni.fit = entry.fit;
        std::memcpy(ni.case_err, entry.case_err, sizeof(ni.case_err));
    } else {
        ni.fit = fitness_and_cases(ni.prog, ni.case_err, *problem,
                                   current_test_inputs, penalize_length);
        if (penalize_length) {
            entry.fit = ni.fit;
            std::memcpy(entry.case_err, ni.case_err, sizeof(entry.case_err));
            eval_lru_put(k, entry);
        }
    }
}

void Population::sort_pop() {
    std::sort(indivs, indivs + SIZE, [](const Individual& a, const Individual& b) {
        double ka = a.fit + 1e-6 * a.prog.n_nodes;
        double kb = b.fit + 1e-6 * b.prog.n_nodes;
        return ka < kb;
    });
}

void Population::init(std::mt19937& /*rng*/, const ProblemDef& p) {
    problem          = &p;
    curriculum_stage = 0;
    current_test_inputs = make_test_inputs(p, 0, N_CASES);
    mutation_bandit.reset();

    // Seed: n_inputs input terminals + one LOADF 0.0 output node
    Program seed     = {};
    seed.n_inputs    = uint16_t(p.n_inputs);
    seed.n_nodes     = uint16_t(p.n_inputs + 1);
    Node& fn         = seed.nodes[p.n_inputs];
    fn.op            = Op::LOADF;
    fn.lit.f         = 0.0f;
    fn.innov         = next_innovation();
    seed.output_node = uint16_t(p.n_inputs);

    Individual seed_ind;
    seed_ind.prog = seed;
    eval_individual(seed_ind);
    for (auto& ind : indivs) ind = seed_ind;
    sort_pop();

    species.clear();
    Species s0;
    s0.rep      = seed;
    s0.best_fit = seed_ind.fit;
    for (int i = 0; i < SIZE; i++) s0.members.push_back(i);
    species.push_back(std::move(s0));
}

void Population::assign_species() {
    for (auto& s : species) s.members.clear();

    for (int i = 0; i < SIZE; i++) {
        double best_dist = std::numeric_limits<double>::max();
        int    best_si   = -1;
        for (int si = 0; si < int(species.size()); si++) {
            double d = neat_distance(indivs[i].prog, species[si].rep);
            if (d < best_dist) { best_dist = d; best_si = si; }
        }

        bool can_create = (int(species.size()) < MAX_SPECIES) &&
                          (best_dist >= COMPAT_THRESH);
        if (can_create) {
            Species ns;
            ns.rep      = indivs[i].prog;
            ns.best_fit = indivs[i].fit;
            ns.members.push_back(i);
            species.push_back(std::move(ns));
        } else {
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

    // Prefer programs with fewer total nodes (includes inactive — proxy for size)
    int min_len = std::numeric_limits<int>::max();
    for (int i = 0; i < n_cand; i++)
        min_len = std::min(min_len, int(indivs[cand[i]].prog.n_nodes));

    int short_cand[SIZE], n_short = 0;
    for (int i = 0; i < n_cand; i++)
        if (int(indivs[cand[i]].prog.n_nodes) == min_len)
            short_cand[n_short++] = cand[i];

    return short_cand[std::uniform_int_distribution<int>(0, n_short - 1)(rng)];
}

void Population::step(std::mt19937& rng) {
    generation++;

    // ── Global stagnation & hot-burst ─────────────────────────────────────────
    {
        double cur_best = indivs[0].fit;
        if (cur_best < global_best_fit * 0.999) {
            global_best_fit     = cur_best;
            global_stagnation   = 0;
            hot_burst_remaining = 0;
            novelty_seen.clear();
            mutation_bandit.reset();
        } else {
            global_stagnation++;
        }
        if (global_stagnation > 0 &&
            global_stagnation % GSTAG_HOT_TRIGGER == 0 &&
            hot_burst_remaining == 0) {
            hot_burst_remaining = HOT_BURST_DURATION;
            mutation_bandit.reset();
        }
        if (hot_burst_remaining > 0) hot_burst_remaining--;
    }

    assign_species();

    // ── Hardening ─────────────────────────────────────────────────────────────
    if (generation % HARDEN_INTERVAL == 0) {
        for (const auto& s : species) {
            if (s.members.empty()) continue;
            int champ = s.members[0];
            for (int idx : s.members)
                if (indivs[idx].fit < indivs[champ].fit) champ = idx;
            indivs[champ].hardness.recompute(indivs[champ].prog, indivs[champ].fit,
                                             *problem, current_test_inputs);
        }
    }

    // ── Species stagnation update ─────────────────────────────────────────────
    int eff_stag_limit = (hot_burst_remaining > 0)
                       ? STAG_LIMIT * HOT_STAG_MULTIPLIER
                       : STAG_LIMIT;

    for (auto& s : species) {
        double best = std::numeric_limits<double>::max();
        for (int idx : s.members) best = std::min(best, indivs[idx].fit);
        if (best < s.best_fit * 0.999) { s.best_fit = best; s.stagnation = 0; }
        else s.stagnation++;
    }

    // ── Cull stagnant species; always keep species containing indivs[0] ───────
    species.erase(
        std::remove_if(species.begin(), species.end(),
            [&](const Species& s) {
                if (s.stagnation < eff_stag_limit) return false;
                for (int idx : s.members) if (idx == 0) return false;
                return true;
            }),
        species.end());

    int nsp = int(species.size());

    // ── Score surviving species ───────────────────────────────────────────────
    std::vector<double> scores(nsp, 0.0);
    double total_score = 0.0;
    for (int si = 0; si < nsp; si++) {
        const auto& s = species[si];
        for (int idx : s.members)
            scores[si] += 1.0 / (indivs[idx].fit + 1e-10);
        scores[si] /= double(s.members.size());
        total_score += scores[si];
    }

    if (total_score == 0.0) {
        int best_si = 0;
        double best_fit = std::numeric_limits<double>::max();
        for (int si = 0; si < nsp; si++)
            for (int idx : species[si].members)
                if (indivs[idx].fit < best_fit) {
                    best_fit = indivs[idx].fit; best_si = si;
                }
        for (int idx : species[best_si].members)
            scores[best_si] += 1.0 / (indivs[idx].fit + 1e-10);
        scores[best_si] /= double(species[best_si].members.size());
        total_score = scores[best_si];
    }

    int champion_slots  = nsp;
    int offspring_slots = SIZE - champion_slots;
    if (offspring_slots < 0) offspring_slots = 0;

    std::vector<int> offspring(nsp, 0);
    int assigned = 0;
    for (int si = 0; si < nsp; si++) {
        offspring[si] = int(scores[si] / total_score * offspring_slots);
        assigned += offspring[si];
    }
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

    bool cache_active = (global_stagnation >= GSTAG_ENABLE_CACHE);

    auto finalize_child = [&](Program& child, const Hardness& parent_hardness) {
        if (cache_active) {
            float fp[N_BEH];
            compute_fingerprint(child, fp, *problem);
            uint32_t h = behavior_hash(fp);
            for (int attempt = 0; novelty_seen.count(h) && attempt < 512; attempt++) {
                int fkind = mutation_bandit.select(rng);
                child = mutate(child, parent_hardness, fkind, rng,
                               *problem, current_test_inputs);
                compute_fingerprint(child, fp, *problem);
                h = behavior_hash(fp);
            }
            novelty_seen.insert(h);
        }
    };

    // ── Elites: one champion per species ──────────────────────────────────────
    for (auto& s : species) {
        if (next_count >= SIZE || s.members.empty()) continue;
        int champ = s.members[0];
        for (int idx : s.members)
            if (indivs[idx].fit < indivs[champ].fit) champ = idx;
        next[next_count++] = indivs[champ];
        s.rep = indivs[champ].prog;
    }

    // ── Offspring ─────────────────────────────────────────────────────────────
    for (int si = 0; si < nsp && next_count < SIZE; si++) {
        const auto& s = species[si];
        if (s.members.empty()) continue;
        for (int j = 0; j < offspring[si] && next_count < SIZE; j++) {
            int p = select_in_species(s, rng);
            double parent_fit = indivs[p].fit;
            Program child;
            int mut_kind = -1;

            if (int(s.members.size()) == 1 || coin(rng) < MUT_RATE) {
                mut_kind = mutation_bandit.select(rng);
                child = mutate(indivs[p].prog, indivs[p].hardness,
                               mut_kind, rng, *problem, current_test_inputs);
            } else {
                int q = select_in_species(s, rng);
                child = crossover(indivs[p].prog, indivs[p].fit,
                                  indivs[q].prog, indivs[q].fit, rng);
            }
            finalize_child(child, indivs[p].hardness);
            Individual& ni = next[next_count++];
            ni.prog     = child;
            ni.hardness = {};
            eval_individual(ni, !cache_active);

            if (mut_kind >= 0) {
                double reward = std::max(0.0, parent_fit - ni.fit);
                mutation_bandit.update(mut_kind, reward);
            }
        }
    }

    // ── Fill remaining slots ──────────────────────────────────────────────────
    while (next_count < SIZE) {
        int si = int(std::uniform_int_distribution<int>(0, nsp - 1)(rng));
        const auto& s = species[si];
        if (s.members.empty()) continue;
        int p = select_in_species(s, rng);
        double parent_fit = indivs[p].fit;
        int mut_kind = mutation_bandit.select(rng);
        Program child = mutate(indivs[p].prog, indivs[p].hardness,
                               mut_kind, rng, *problem, current_test_inputs);
        finalize_child(child, indivs[p].hardness);
        Individual& ni = next[next_count++];
        ni.prog     = child;
        ni.hardness = {};
        eval_individual(ni, !cache_active);
        double reward = std::max(0.0, parent_fit - ni.fit);
        mutation_bandit.update(mut_kind, reward);
    }

    memcpy(indivs, next, sizeof(indivs));
    sort_pop();

    // ── Curriculum advancement ────────────────────────────────────────────────
    int n_stages = problem_n_stages(*problem);
    if (curriculum_stage < n_stages - 1 &&
        indivs[0].fit < problem->curriculum_advance_thresh) {
        curriculum_stage++;
        current_test_inputs = make_test_inputs(*problem, curriculum_stage, N_CASES);

        eval_lru_list.clear();
        eval_lru_map.clear();
        novelty_seen.clear();
        mutation_bandit.reset();

        for (int i = 0; i < SIZE; i++) eval_individual(indivs[i]);
        sort_pop();
        global_best_fit     = indivs[0].fit;
        global_stagnation   = 0;
        hot_burst_remaining = 0;

        std::cout << "*** curriculum stage " << curriculum_stage << "\n";
        for (int j = 0; j < problem->n_inputs; j++) {
            auto [lo, hi] = problem->inputs[j].range_at(curriculum_stage);
            std::cout << "    input[" << j << "] range=[" << lo << ", " << hi << "]\n";
        }
        std::cout << "    best_fit=" << indivs[0].fit << "\n";
    }
}
