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

void Population::eval_individual(Individual& ni) {
    EvalKey k = program_hash(ni.prog);
    EvalEntry entry;
    if (eval_lru_get(k, entry)) {
        ni.fit = entry.fit;
        std::memcpy(ni.case_err, entry.case_err, sizeof(ni.case_err));
    } else {
        ni.fit = fitness_and_cases(ni.prog, ni.case_err, *problem, current_test_inputs);
        entry.fit = ni.fit;
        std::memcpy(entry.case_err, ni.case_err, sizeof(entry.case_err));
        eval_lru_put(k, entry);
    }
}

void Population::sort_pop() {
    const int N = SIZE;

    // ── Non-dominated sort ─────────────────────────────────────────────────
    // Two objectives: fit (minimize), num_instrs (minimize).
    int dom_count[N];         // # of individuals that dominate me
    int dom_by[N][N];         // individuals that I dominate
    int dom_by_n[N];
    memset(dom_count, 0, sizeof(dom_count));
    memset(dom_by_n,  0, sizeof(dom_by_n));

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double fi = indivs[i].fit,           fj = indivs[j].fit;
            int    li = indivs[i].prog.num_instrs, lj = indivs[j].prog.num_instrs;
            bool i_le = (fi <= fj && li <= lj), i_lt = (fi < fj || li < lj);
            bool j_le = (fj <= fi && lj <= li), j_lt = (fj < fi || lj < li);
            if (i_le && i_lt) { dom_by[i][dom_by_n[i]++] = j; dom_count[j]++; }
            else if (j_le && j_lt) { dom_by[j][dom_by_n[j]++] = i; dom_count[i]++; }
        }
    }

    int current_front[N], next_front[N];
    int cf_n = 0;
    for (int i = 0; i < N; i++) {
        indivs[i].rank = -1;
        if (dom_count[i] == 0) { indivs[i].rank = 0; current_front[cf_n++] = i; }
    }

    int cur_rank = 0;
    while (cf_n > 0) {
        int nf_n = 0;
        for (int fi = 0; fi < cf_n; fi++) {
            int i = current_front[fi];
            for (int k = 0; k < dom_by_n[i]; k++) {
                int j = dom_by[i][k];
                if (--dom_count[j] == 0) {
                    indivs[j].rank = cur_rank + 1;
                    next_front[nf_n++] = j;
                }
            }
        }
        cur_rank++;
        memcpy(current_front, next_front, nf_n * sizeof(int));
        cf_n = nf_n;
    }

    // ── Crowding distance ──────────────────────────────────────────────────
    for (int i = 0; i < N; i++) indivs[i].crowding_dist = 0.0;

    for (int r = 0; r <= cur_rank; r++) {
        int members[N], nm = 0;
        for (int i = 0; i < N; i++)
            if (indivs[i].rank == r) members[nm++] = i;
        if (nm == 0) continue;
        if (nm <= 2) {
            for (int k = 0; k < nm; k++) indivs[members[k]].crowding_dist = 1e18;
            continue;
        }

        auto accum_obj = [&](auto get_val) {
            std::sort(members, members + nm,
                      [&](int a, int b){ return get_val(a) < get_val(b); });
            indivs[members[0]].crowding_dist    = 1e18;
            indivs[members[nm-1]].crowding_dist = 1e18;
            double range = get_val(members[nm-1]) - get_val(members[0]);
            if (range > 1e-12) {
                for (int k = 1; k < nm - 1; k++) {
                    if (indivs[members[k]].crowding_dist >= 1e17) continue;
                    indivs[members[k]].crowding_dist +=
                        (get_val(members[k+1]) - get_val(members[k-1])) / range;
                }
            }
        };
        accum_obj([&](int i) -> double { return indivs[i].fit; });
        accum_obj([&](int i) -> double { return double(indivs[i].prog.num_instrs); });
    }

    // ── Sort: rank ASC, crowding_dist DESC ─────────────────────────────────
    std::sort(indivs, indivs + N, [](const Individual& a, const Individual& b) {
        if (a.rank != b.rank) return a.rank < b.rank;
        return a.crowding_dist > b.crowding_dist;
    });
}

void Population::init(std::mt19937& rng, const ProblemDef& p) {
    (void)rng;
    problem = &p;
    curriculum_stage = 0;

    // Stage 0: first curriculum entry if any, otherwise full range.
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

    // Only consider members with the best (lowest) rank in this species.
    int min_rank = std::numeric_limits<int>::max();
    for (int i = 0; i < n; i++)
        min_rank = std::min(min_rank, indivs[s.members[i]].rank);

    int cand[SIZE], n_cand = 0;
    for (int i = 0; i < n; i++)
        if (indivs[s.members[i]].rank == min_rank) cand[n_cand++] = s.members[i];

    if (n_cand == 1) return cand[0];

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

    {
        // Use min MSRE across rank-0 individuals as the scalar progress signal.
        double cur_best = std::numeric_limits<double>::max();
        for (int i = 0; i < SIZE; i++) {
            if (indivs[i].rank > 0) break;  // rank-0 are at the front after sort
            cur_best = std::min(cur_best, indivs[i].fit);
        }
        if (cur_best < global_best_fit * 0.999) {
            global_best_fit     = cur_best;
            global_stagnation   = 0;
            hot_burst_remaining = 0;
            novelty_seen.clear();
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

    // --- Cull stagnant species; always keep the one containing indivs[0] (global best) ---
    species.erase(
        std::remove_if(species.begin(), species.end(),
            [&](const Species& s) {
                if (s.stagnation < eff_stag_limit) return false;
                for (int idx : s.members) if (idx == 0) return false;
                return true;
            }),
        species.end());

    int nsp = int(species.size());

    // --- Score surviving species (rank-weighted: rank-0 member = weight 1, rank-k = 1/(k+1)) ---
    std::vector<double> scores(nsp, 0.0);
    double total_score = 0.0;
    for (int si = 0; si < nsp; si++) {
        const auto& s = species[si];
        for (int idx : s.members)
            scores[si] += 1.0 / (1.0 + indivs[idx].rank);
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
            scores[best_si] += 1.0 / (1.0 + indivs[idx].rank);
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
            for (int attempt = 0; novelty_seen.count(h) && attempt < 64; attempt++) {
                child = mutate(child, parent_hardness, rng, *problem, current_test_inputs);
                compute_fingerprint(child, fp, *problem);
                h = behavior_hash(fp);
            }
            novelty_seen.insert(h);
        }
    };

    for (auto& s : species) {
        if (next_count >= SIZE || s.members.empty()) continue;
        int champ = s.members[0];
        for (int idx : s.members) {
            const Individual& ic = indivs[idx], &cc = indivs[champ];
            if (ic.rank < cc.rank ||
                (ic.rank == cc.rank && ic.crowding_dist > cc.crowding_dist))
                champ = idx;
        }
        next[next_count++] = indivs[champ];
        s.rep = indivs[champ].prog;
    }

    for (int si = 0; si < nsp && next_count < SIZE; si++) {
        const auto& s = species[si];
        if (s.members.empty()) continue;
        for (int j = 0; j < offspring[si] && next_count < SIZE; j++) {
            int p = select_in_species(s, rng);
            Program child;
            if (int(s.members.size()) == 1 || coin(rng) < MUT_RATE) {
                child = mutate(indivs[p].prog, indivs[p].hardness, rng, *problem, current_test_inputs);
            } else {
                int q = select_in_species(s, rng);
                child = crossover(indivs[p].prog, indivs[p].fit,
                                  indivs[q].prog, indivs[q].fit, rng);
            }
            finalize_child(child, indivs[p].hardness);
            Individual& ni = next[next_count++];
            ni.prog     = child;
            ni.hardness = {};
            eval_individual(ni);
        }
    }

    while (next_count < SIZE) {
        int si = int(std::uniform_int_distribution<int>(0, nsp - 1)(rng));
        const auto& s = species[si];
        if (s.members.empty()) continue;
        int p = select_in_species(s, rng);
        Program child = mutate(indivs[p].prog, indivs[p].hardness, rng, *problem, current_test_inputs);
        finalize_child(child, indivs[p].hardness);
        Individual& ni = next[next_count++];
        ni.prog     = child;
        ni.hardness = {};
        eval_individual(ni);
    }

    memcpy(indivs, next, sizeof(indivs));
    sort_pop();

    // Curriculum advancement: when best fitness crosses the threshold, widen
    // each input to its next stage and recompute everyone's fitness.
    int n_stages = problem_n_stages(*problem);
    // Compute best MSRE on the rank-0 front for curriculum and reporting.
    double front_best_msre = std::numeric_limits<double>::max();
    for (int i = 0; i < SIZE && indivs[i].rank == 0; i++)
        front_best_msre = std::min(front_best_msre, indivs[i].fit);

    if (curriculum_stage < n_stages - 1 &&
        front_best_msre < problem->curriculum_advance_thresh) {
        curriculum_stage++;
        current_test_inputs = make_test_inputs(*problem, curriculum_stage, N_CASES);

        // LRU entries are stale (different test range) — clear before reeval.
        eval_lru_list.clear();
        eval_lru_map.clear();
        novelty_seen.clear();
        for (int i = 0; i < SIZE; i++)
            eval_individual(indivs[i]);
        sort_pop();
        global_best_fit     = front_best_msre;
        global_stagnation   = 0;
        hot_burst_remaining = 0;

        std::cout << "*** curriculum stage " << curriculum_stage << "\n";
        for (int j = 0; j < problem->n_inputs; j++) {
            auto [lo, hi] = problem->inputs[j].range_at(curriculum_stage);
            std::cout << "    input[" << j << "] range=[" << lo << ", " << hi << "]\n";
        }
        std::cout << "    best_fit=" << front_best_msre << "\n";
    }
}
