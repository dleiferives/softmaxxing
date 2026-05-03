#include "speciation.hpp"
#include "execute.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <climits>

void compute_fingerprint(const Program& p, float out[N_BEH], const ProblemDef& problem) {
    float out_val[1];
    for (int i = 0; i < N_BEH; i++) {
        const float* xs = &problem.behavior_samples[i * problem.n_inputs];
        execute(p, xs, problem.n_inputs, out_val, 1);
        out[i] = std::isfinite(out_val[0]) ? std::clamp(out_val[0], -1e6f, 1e6f) : 1e6f;
    }
}

double neat_distance(const Program& a, const Program& b) {
    int na = a.num_instrs, nb = b.num_instrs;
    if (na == 0 && nb == 0) return 0.0;
    if (na == 0 || nb == 0) return 1.0;

    // Sort instruction indices by innovation number for alignment.
    int idx_a[Program::MAX_INSTRS], idx_b[Program::MAX_INSTRS];
    for (int k = 0; k < na; k++) idx_a[k] = k;
    for (int k = 0; k < nb; k++) idx_b[k] = k;
    std::sort(idx_a, idx_a + na, [&](int x, int y){ return a.instrs[x].innov < a.instrs[y].innov; });
    std::sort(idx_b, idx_b + nb, [&](int x, int y){ return b.instrs[x].innov < b.instrs[y].innov; });

    uint32_t max_a   = a.instrs[idx_a[na - 1]].innov;
    uint32_t max_b   = b.instrs[idx_b[nb - 1]].innov;
    uint32_t min_max = std::min(max_a, max_b);

    int    i = 0, j = 0;
    int    matching = 0, disjoint = 0, excess = 0;
    double match_diff_sum = 0.0;

    while (i < na || j < nb) {
        uint32_t ia = (i < na) ? a.instrs[idx_a[i]].innov : UINT32_MAX;
        uint32_t ib = (j < nb) ? b.instrs[idx_b[j]].innov : UINT32_MAX;

        if (ia == ib) {
            const Instr& ga = a.instrs[idx_a[i]];
            const Instr& gb = b.instrs[idx_b[j]];
            double d = (ga.op != gb.op) ? 1.0 : 0.0;
            if (ga.op == gb.op) {
                // penalise literal difference for matching ops
                uint32_t la, lb;
                std::memcpy(&la, &ga.lit, 4);
                std::memcpy(&lb, &gb.lit, 4);
                if (la != lb) d += 0.3;
            }
            match_diff_sum += d;
            matching++;
            i++; j++;
        } else if (ia < ib) {
            if (ia <= min_max) disjoint++; else excess++;
            i++;
        } else {
            if (ib <= min_max) disjoint++; else excess++;
            j++;
        }
    }

    int N = std::max(na, nb);
    double avg_diff = matching > 0 ? match_diff_sum / matching : 0.0;
    // c1 * excess/N + c2 * disjoint/N + c3 * avg_matching_diff
    return 1.0 * excess / N + 1.0 * disjoint / N + 0.3 * avg_diff;
}
