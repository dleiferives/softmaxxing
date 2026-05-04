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
    mix(uint8_t(p.num_instrs));
    mix(uint8_t(p.num_instrs >> 8));
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

void Population::sort_island(Island& isl) {
    std::sort(isl.indivs, isl.indivs + ISLAND_SIZE, [](const Individual& a, const Individual& b) {
        double ka = a.fit + 1e-6 * a.prog.num_instrs;
        double kb = b.fit + 1e-6 * b.prog.num_instrs;
        return ka < kb;
    });
}

const Individual& Population::best() const {
    const Individual* b = &islands[0].indivs[0];
    for (int i = 1; i < N_ISLANDS; i++)
        if (islands[i].indivs[0].fit < b->fit) b = &islands[i].indivs[0];
    return *b;
}

void Population::init(std::mt19937& rng, const ProblemDef& p) {
    (void)rng;
    problem = &p;
    curriculum_stage    = 0;
    current_test_inputs = make_test_inputs(p, 0, N_CASES);

    Program seed = {};
    Instr&  si   = seed.instrs[0];
    si.op    = Op::LOADI;
    si.dst   = 0;
    si.src1  = 0;
    si.src2  = 0;
    si.lit.i = 0;
    si.innov = next_innovation();
    seed.num_instrs = 1;

    Individual seed_ind;
    seed_ind.prog = seed;
    eval_individual(seed_ind);

    for (int ii = 0; ii < N_ISLANDS; ii++) {
        for (auto& ind : islands[ii].indivs) ind = seed_ind;
        islands[ii].best_fit   = seed_ind.fit;
        islands[ii].stagnation = 0;
        sort_island(islands[ii]);
    }
}

int Population::select_in_island(const Island& isl, std::mt19937& rng) const {
    int cand[ISLAND_SIZE];
    for (int i = 0; i < ISLAND_SIZE; i++) cand[i] = i;
    int n_cand = ISLAND_SIZE;

    int cases[N_CASES];
    std::iota(cases, cases + N_CASES, 0);
    std::shuffle(cases, cases + N_CASES, rng);

    int next_cand[ISLAND_SIZE];
    for (int ci : cases) {
        if (n_cand == 1) break;

        float min_err = std::numeric_limits<float>::infinity();
        for (int i = 0; i < n_cand; i++)
            min_err = std::min(min_err, isl.indivs[cand[i]].case_err[ci]);

        float eps_scale = (hot_burst_remaining > 0) ? HOT_EPSILON_SCALE : 1.0f;
        float epsilon   = (min_err * 0.1f + 1e-6f) * eps_scale;

        int n_next = 0;
        for (int i = 0; i < n_cand; i++)
            if (isl.indivs[cand[i]].case_err[ci] <= min_err + epsilon)
                next_cand[n_next++] = cand[i];

        if (n_next > 0) {
            n_cand = n_next;
            std::copy(next_cand, next_cand + n_next, cand);
        }
    }

    int min_len = std::numeric_limits<int>::max();
    for (int i = 0; i < n_cand; i++)
        min_len = std::min(min_len, int(isl.indivs[cand[i]].prog.num_instrs));

    int short_cand[ISLAND_SIZE], n_short = 0;
    for (int i = 0; i < n_cand; i++)
        if (int(isl.indivs[cand[i]].prog.num_instrs) == min_len)
            short_cand[n_short++] = cand[i];

    return short_cand[std::uniform_int_distribution<int>(0, n_short - 1)(rng)];
}

void Population::migrate(std::mt19937& rng) {
    std::uniform_int_distribution<int> other_d(0, N_ISLANDS - 2);
    for (int ii = 0; ii < N_ISLANDS; ii++) {
        int jj = other_d(rng);
        if (jj >= ii) jj++;

        Island& src = islands[ii];
        Island& dst = islands[jj];

        const Individual& immigrant = src.indivs[0];  // best of source island

        // Replace worst slot in dst with a positional cross between immigrant and dst's best.
        Program cross = crossover_positional(immigrant.prog, immigrant.hardness,
                                             dst.indivs[0].prog, dst.indivs[0].hardness,
                                             rng);
        Individual& slot_cross = dst.indivs[ISLAND_SIZE - 1];
        slot_cross.prog     = cross;
        slot_cross.hardness = {};
        eval_individual(slot_cross);

        // Replace second-worst slot with raw clone of immigrant.
        dst.indivs[ISLAND_SIZE - 2] = immigrant;

        sort_island(dst);
    }
}

void Population::step(std::mt19937& rng) {
    generation++;

    // --- Global stagnation + hot burst ---
    {
        double cur_best = best().fit;
        if (cur_best < global_best_fit * 0.999) {
            global_best_fit     = cur_best;
            global_stagnation   = 0;
            hot_burst_remaining = 0;
            novelty_seen.clear();
            frontier_queue.clear();
            frontier_hashes.clear();
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

    bool cache_active = (global_stagnation >= GSTAG_ENABLE_CACHE);

    // Hysteresis control for frontier boost
    {
        int qs = int(frontier_queue.size());
        if (qs < FRONTIER_TARGET * 9 / 10)  frontier_boost_on = true;
        if (qs > FRONTIER_TARGET * 11 / 10) frontier_boost_on = false;
    }

    // Helper: push a candidate into the frontier if not already there or in novelty_seen.
    auto frontier_push = [&](uint32_t h, const Program& prog) {
        if (novelty_seen.count(h) || frontier_hashes.count(h)) return;
        frontier_hashes.insert(h);
        frontier_queue.push_back({h, prog});
    };

    // Helper: pop the front candidate; returns false if queue is empty.
    auto frontier_pop = [&](uint32_t& h_out, Program& prog_out) -> bool {
        if (frontier_queue.empty()) return false;
        auto [h, prog] = frontier_queue.front();
        frontier_queue.pop_front();
        frontier_hashes.erase(h);
        h_out    = h;
        prog_out = prog;
        return true;
    };

    // Periodic reseed: every 100K mutations, replace half the frontier with
    // random mutations from island individuals to break depth-first tunneling.
    if (cache_active && total_mutations > 0 && total_mutations % 100000 == 0) {
        int keep = int(frontier_queue.size()) / 2;
        while (int(frontier_queue.size()) > keep) {
            frontier_hashes.erase(frontier_queue.back().first);
            frontier_queue.pop_back();
        }
        std::uniform_int_distribution<int> isl_d(0, N_ISLANDS - 1);
        std::uniform_int_distribution<int> ind_d(0, ISLAND_SIZE - 1);
        int to_add = FRONTIER_TARGET / 2;
        for (int i = 0; i < to_add; i++) {
            const Individual& src = islands[isl_d(rng)].indivs[ind_d(rng)];
            Program cand = mutate(src.prog, src.hardness, rng, *problem, current_test_inputs);
            total_mutations++;
            float fp[N_BEH];
            compute_fingerprint(cand, fp, *problem);
            frontier_push(behavior_hash(fp), cand);
        }
    }

    // When queue is below target, generate one unevaluated candidate and queue it.
    // Loops forever — the candidate must be genuinely new (not in novelty_seen or frontier).
    auto enqueue_boost = [&](const Program& prog, const Hardness& h) {
        if (!frontier_boost_on) return;
        Program extra = prog;
        for (uint64_t i = 1; ; i++) {
            extra = mutate(extra, h, rng, *problem, current_test_inputs);
            total_mutations++;
            float fp2[N_BEH];
            compute_fingerprint(extra, fp2, *problem);
            uint32_t h2 = behavior_hash(fp2);
            if (!novelty_seen.count(h2) && !frontier_hashes.count(h2)) {
                frontier_push(h2, extra);
                return;
            }
            if (i == 100000)
                std::cout << "[frontier boost: 100K iters still searching]\n";
            if (i % 1000000 == 0)
                std::cout << "[frontier boost: " << i/1000000 << "M iters still searching]\n";
        }
    };

    auto finalize_child = [&](Program& child, const Hardness& parent_hardness) {
        if (!cache_active) return;
        float fp[N_BEH];
        compute_fingerprint(child, fp, *problem);
        uint32_t h = behavior_hash(fp);
        if (!novelty_seen.count(h)) {
            novelty_seen.insert(h);
            enqueue_boost(child, parent_hardness);
            return;
        }
        // Not novel — drain frontier candidates until we find one that is.
        for (uint64_t attempt = 1; ; attempt++) {
            uint32_t fh; Program fprog;
            if (frontier_pop(fh, fprog)) {
                if (!novelty_seen.count(fh)) {
                    child = fprog;
                    novelty_seen.insert(fh);
                    enqueue_boost(child, parent_hardness);
                    return;
                }
            } else {
                // Frontier empty — mutate in place as fallback.
                child = mutate(child, parent_hardness, rng, *problem, current_test_inputs);
                total_mutations++;
                compute_fingerprint(child, fp, *problem);
                h = behavior_hash(fp);
                if (!novelty_seen.count(h)) {
                    novelty_seen.insert(h);
                    enqueue_boost(child, parent_hardness);
                    return;
                }
            }
            if (attempt == 100000)
                std::cout << "[finalize_child: 100K attempts still searching]\n";
            if (attempt % 1000000 == 0)
                std::cout << "[finalize_child: " << attempt/1000000 << "M attempts still searching]\n";
        }
    };

    // --- Evolve each island independently ---
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    for (int ii = 0; ii < N_ISLANDS; ii++) {
        Island& isl = islands[ii];

        // Per-island stagnation
        double ibest = isl.indivs[0].fit;
        if (ibest < isl.best_fit * 0.999) { isl.best_fit = ibest; isl.stagnation = 0; }
        else                               { isl.stagnation++; }

        // Harden champion every HARDEN_INTERVAL
        //if (generation % HARDEN_INTERVAL == 0)
            //isl.indivs[0].hardness.recompute(isl.indivs[0].prog, isl.indivs[0].fit,
                                              //*problem, current_test_inputs);

        Individual next[ISLAND_SIZE];
        int next_count = 0;

        // Elitism: champion survives unchanged
        next[next_count++] = isl.indivs[0];
	if (cache_active && ii > 0) {
		for (int iii=1; iii< ISLAND_SIZE * 2 / 3; iii++){
		    next[next_count++] = isl.indivs[select_in_island(isl, rng)];
		}
	}

	if (!cache_active) {
		// If this island is stagnant, inject a random immigrant from another island
		// (extra migration on top of the periodic batch migration).
		if (isl.stagnation > 0 && isl.stagnation % ISLAND_STAG_LIMIT == 0) {
		    int src_ii = std::uniform_int_distribution<int>(0, N_ISLANDS - 2)(rng);
		    if (src_ii >= ii) src_ii++;
		    if (next_count < ISLAND_SIZE) {
			next[next_count] = islands[src_ii].indivs[0];
			next_count++;
		    }
		}
	}

        while (next_count < ISLAND_SIZE) {
            int p = select_in_island(isl, rng);
            Program child;
            if (cache_active || coin(rng) < MUT_RATE) {
                child = mutate(isl.indivs[p].prog, isl.indivs[p].hardness,
                               rng, *problem, current_test_inputs);
            } else {
                int q = select_in_island(isl, rng);
                child = crossover(isl.indivs[p].prog, isl.indivs[p].fit,
                                  isl.indivs[q].prog, isl.indivs[q].fit, rng);
            }
            finalize_child(child, isl.indivs[p].hardness);
            Individual& ni = next[next_count++];
            ni.prog     = child;
            ni.hardness = {};
            eval_individual(ni);
        }

        memcpy(isl.indivs, next, sizeof(isl.indivs));
        sort_island(isl);
    }

    // --- Periodic batch migration ---
    if (generation % MIGRATE_INTERVAL == 0)
        migrate(rng);

    // --- Curriculum advancement ---
    int n_stages = problem_n_stages(*problem);
    if (curriculum_stage < n_stages - 1 &&
        best().fit < problem->curriculum_advance_thresh) {
        curriculum_stage++;
        current_test_inputs = make_test_inputs(*problem, curriculum_stage, N_CASES);

        eval_lru_list.clear();
        eval_lru_map.clear();
        novelty_seen.clear();
        frontier_queue.clear();
        frontier_hashes.clear();

        for (int ii = 0; ii < N_ISLANDS; ii++) {
            for (int i = 0; i < ISLAND_SIZE; i++)
                eval_individual(islands[ii].indivs[i]);
            sort_island(islands[ii]);
            islands[ii].best_fit = islands[ii].indivs[0].fit;
        }

        global_best_fit     = best().fit;
        global_stagnation   = 0;
        hot_burst_remaining = 0;

        std::cout << "*** curriculum stage " << curriculum_stage << "\n";
        for (int j = 0; j < problem->n_inputs; j++) {
            auto [lo, hi] = problem->inputs[j].range_at(curriculum_stage);
            std::cout << "    input[" << j << "] range=[" << lo << ", " << hi << "]\n";
        }
        std::cout << "    best_fit=" << best().fit << "\n";
    }
}
