#include "population.hpp"
#include "fitness.hpp"
#include "mutate.hpp"
#include "random.hpp"
#include "hardening.hpp"
#include "innovation.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>

Population::EvalKey Population::program_hash(const Program& p) {
    uint64_t h = 14695981039346656037ULL;
    auto mix = [&](uint8_t b) { h ^= b; h *= 1099511628211ULL; };
    mix(p.num_chroms);
    mix(uint8_t(p.num_instrs));
    mix(uint8_t(p.num_instrs >> 8));
    for (int k = 0; k < p.num_chroms; k++)
        mix(p.chrom_lens[k]);
    for (int i = 0; i < p.num_instrs; i++) {
        const Instr& ins = p.instrs[i];
        mix(uint8_t(ins.op));
        mix(ins.dst);
        mix(ins.src1);
        mix(ins.src2);
        uint32_t lit; __builtin_memcpy(&lit, &ins.lit, sizeof(lit));
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
        ni.fit = fitness_and_cases(ni.prog, ni.case_err, *problem, current_test_inputs, penalize_length);
        if (penalize_length) {
            entry.fit = ni.fit;
            std::memcpy(entry.case_err, ni.case_err, sizeof(entry.case_err));
            eval_lru_put(k, entry);
        }
    }
}

void Population::sort_pop() {
    std::sort(indivs, indivs + SIZE, [](const Individual& a, const Individual& b) {
        double ka = a.fit + 1e-6 * a.prog.num_instrs;
        double kb = b.fit + 1e-6 * b.prog.num_instrs;
        return ka < kb;
    });
}

float Population::compute_novelty(const float fp[N_BEH]) const {
    if (novelty_archive.empty()) return 1.0f;

    int n = int(novelty_archive.size());
    // Compute distances; partial_sort to find kNN without full sort
    static thread_local float dists[ARCHIVE_MAX];
    for (int i = 0; i < n; i++) {
        float d = 0.0f;
        for (int j = 0; j < N_BEH; j++) {
            float diff = fp[j] - novelty_archive[i].fp[j];
            d += diff * diff;
        }
        dists[i] = std::sqrt(d);
    }

    int k = std::min(FRONTIER_K, n);
    std::partial_sort(dists, dists + k, dists + n);
    float avg = 0.0f;
    for (int i = 0; i < k; i++) avg += dists[i];
    return avg / float(k);
}

void Population::add_to_frontier(const Program& prog, const Hardness& hardness,
                                   float fitness, const float fp[N_BEH]) {
    float novelty = compute_novelty(fp);

    // Rolling archive: evict oldest when full
    ArchiveEntry ae;
    std::memcpy(ae.fp, fp, sizeof(ae.fp));
    ae.fitness = fitness;
    if (int(novelty_archive.size()) >= ARCHIVE_MAX)
        novelty_archive.erase(novelty_archive.begin());
    novelty_archive.push_back(ae);

    FrontierEntry fe;
    fe.prog     = prog;
    fe.hardness = hardness;
    fe.fitness  = fitness;
    fe.novelty  = novelty;
    fe.priority = novelty + 1.0f / (fitness + 1e-6f);

    // Insert maintaining descending priority order
    auto pos = std::lower_bound(frontier.begin(), frontier.end(), fe.priority,
        [](const FrontierEntry& e, float p) { return e.priority > p; });
    frontier.insert(pos, std::move(fe));

    if (int(frontier.size()) > FRONTIER_MAX)
        frontier.pop_back();
}

int Population::sample_frontier_idx(std::mt19937& rng) const {
    int n = int(frontier.size());
    if (n == 0) return -1;

    // Exponentially decaying weights by rank so top entries are sampled more
    static thread_local float w[FRONTIER_MAX];
    for (int i = 0; i < n; i++)
        w[i] = std::exp(-0.08f * float(i));

    std::discrete_distribution<int> dist(w, w + n);
    return dist(rng);
}

void Population::init(std::mt19937& rng, const ProblemDef& p) {
    (void)rng;
    problem = &p;
    curriculum_stage = 0;

    current_test_inputs = make_test_inputs(p, 0, N_CASES);

    Program seed = {};
    Instr&  si   = seed.instrs[0];
    si.op    = Op::LOADI;
    si.dst   = 0;
    si.src1  = 0;
    si.src2  = 0;
    si.lit.i = 0;
    si.innov = next_innovation();
    seed.chrom_lens[0] = 1;
    seed.num_chroms    = 1;
    seed.num_instrs    = 1;

    Individual seed_ind;
    seed_ind.prog = seed;
    eval_individual(seed_ind);
    for (auto& ind : indivs)
        ind = seed_ind;
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

        bool can_create = (int(species.size()) < MAX_SPECIES) && (best_dist >= COMPAT_THRESH);
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

    int min_len = std::numeric_limits<int>::max();
    for (int i = 0; i < n_cand; i++)
        min_len = std::min(min_len, int(indivs[cand[i]].prog.num_instrs));

    int short_cand[SIZE], n_short = 0;
    for (int i = 0; i < n_cand; i++)
        if (int(indivs[cand[i]].prog.num_instrs) == min_len)
            short_cand[n_short++] = cand[i];

    return short_cand[std::uniform_int_distribution<int>(0, n_short - 1)(rng)];
}

void Population::step(std::mt19937& rng) {
    generation++;

    {
        double cur_best = indivs[0].fit;
        if (cur_best < global_best_fit * 0.999) {
            global_best_fit     = cur_best;
            global_stagnation   = 0;
            hot_burst_remaining = 0;
            novelty_archive.clear();
            frontier.clear();
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

    // --- Stagnation update ---
    int eff_stag_limit = (hot_burst_remaining > 0)
                       ? STAG_LIMIT * HOT_STAG_MULTIPLIER
                       : STAG_LIMIT;

    for (auto& s : species) {
        double best = std::numeric_limits<double>::max();
        for (int idx : s.members)
            best = std::min(best, indivs[idx].fit);
        if (best < s.best_fit * 0.999) { s.best_fit = best; s.stagnation = 0; }
        else s.stagnation++;
    }

    // --- Cull stagnant species; always keep the one containing indivs[0] ---
    species.erase(
        std::remove_if(species.begin(), species.end(),
            [&](const Species& s) {
                if (s.stagnation < eff_stag_limit) return false;
                for (int idx : s.members) if (idx == 0) return false;
                return true;
            }),
        species.end());

    int nsp = int(species.size());

    // --- Score surviving species ---
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
                if (indivs[idx].fit < best_fit) { best_fit = indivs[idx].fit; best_si = si; }
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

    // Helper: eval child, then add to frontier
    auto make_and_add = [&](Program& child, const Hardness& parent_hardness) {
        Individual& ni = next[next_count++];
        ni.prog     = child;
        ni.hardness = {};
        eval_individual(ni, true);

        float fp[N_BEH];
        compute_fingerprint(child, fp, *problem);
        add_to_frontier(child, parent_hardness, float(ni.fit), fp);
    };

    // Champions
    for (auto& s : species) {
        if (next_count >= SIZE || s.members.empty()) continue;
        int champ = s.members[0];
        for (int idx : s.members)
            if (indivs[idx].fit < indivs[champ].fit) champ = idx;
        next[next_count++] = indivs[champ];
        s.rep = indivs[champ].prog;
    }

    // Offspring
    for (int si = 0; si < nsp && next_count < SIZE; si++) {
        const auto& s = species[si];
        if (s.members.empty()) continue;
        for (int j = 0; j < offspring[si] && next_count < SIZE; j++) {
            Program  child;
            Hardness parent_hardness;

            // When cache is active and frontier is populated, prefer frontier parents
            if (cache_active && !frontier.empty() && coin(rng) < FRONTIER_SAMPLE_RATE) {
                int fi = sample_frontier_idx(rng);
                child          = mutate(frontier[fi].prog, frontier[fi].hardness, rng, *problem, current_test_inputs);
                parent_hardness = frontier[fi].hardness;
            } else {
                int p = select_in_species(s, rng);
                if (int(s.members.size()) == 1 || coin(rng) < MUT_RATE) {
                    child          = mutate(indivs[p].prog, indivs[p].hardness, rng, *problem, current_test_inputs);
                    parent_hardness = indivs[p].hardness;
                } else {
                    int q = select_in_species(s, rng);
                    child          = crossover(indivs[p].prog, indivs[p].fit,
                                               indivs[q].prog, indivs[q].fit, rng);
                    parent_hardness = indivs[p].hardness;
                }
            }

            make_and_add(child, parent_hardness);
        }
    }

    // Fill remaining slots
    while (next_count < SIZE) {
        Program  child;
        Hardness parent_hardness;

        if (cache_active && !frontier.empty() && coin(rng) < FRONTIER_SAMPLE_RATE) {
            int fi = sample_frontier_idx(rng);
            child          = mutate(frontier[fi].prog, frontier[fi].hardness, rng, *problem, current_test_inputs);
            parent_hardness = frontier[fi].hardness;
        } else {
            int si = int(std::uniform_int_distribution<int>(0, nsp - 1)(rng));
            const auto& s = species[si];
            if (s.members.empty()) continue;
            int p = select_in_species(s, rng);
            child          = mutate(indivs[p].prog, indivs[p].hardness, rng, *problem, current_test_inputs);
            parent_hardness = indivs[p].hardness;
        }

        make_and_add(child, parent_hardness);
    }

    memcpy(indivs, next, sizeof(indivs));
    sort_pop();

    // Curriculum advancement
    int n_stages = problem_n_stages(*problem);
    if (curriculum_stage < n_stages - 1 &&
        indivs[0].fit < problem->curriculum_advance_thresh) {
        curriculum_stage++;
        current_test_inputs = make_test_inputs(*problem, curriculum_stage, N_CASES);

        eval_lru_list.clear();
        eval_lru_map.clear();
        novelty_archive.clear();
        frontier.clear();
        for (int i = 0; i < SIZE; i++)
            eval_individual(indivs[i]);
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
