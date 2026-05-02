#include "mutate.hpp"
#include "random.hpp"
#include "dag.hpp"
#include "fitness.hpp"
#include <cstring>
#include <cmath>

static constexpr float DEAD_WEIGHT = 10.0f;

// Pick an instruction index within a chromosome, weighted by liveness + hardness.
static int pick_instr(int cs, int clen, const bool live[],
                      const Hardness& hardness, std::mt19937& rng) {
    if (clen == 1) return 0;
    float w[Program::MAX_CHROM_LEN];
    float total = 0.0f;
    for (int i = 0; i < clen; i++) {
        float base = live[cs + i] ? 1.0f : DEAD_WEIGHT;
        w[i]  = base * hardness.weight(cs + i);
        total += w[i];
    }
    float r = std::uniform_real_distribution<float>(0.0f, total)(rng);
    for (int i = 0; i < clen; i++) {
        r -= w[i];
        if (r <= 0.0f) return i;
    }
    return clen - 1;
}

// Mutate a raw 32-bit literal value with one of several small perturbations.
static uint32_t perturb_lit(uint32_t v, std::mt19937& rng) {
    switch (std::uniform_int_distribution<int>(0, 4)(rng)) {
    case 0: // flip one random bit
        return v ^ (1u << (rng() & 31u));
    case 1: // add or subtract a small power-of-two
        { uint32_t delta = 1u << (rng() & 15u);
          return (rng() & 1) ? v + delta : v - delta; }
    case 2: // shift the whole value left or right by 1
        return (rng() & 1) ? (v << 1) : (v >> 1);
    case 3: // randomise just the lower 8 bits
        return (v & 0xFFFFFF00u) | (rng() & 0xFFu);
    default: // randomise just the upper 8 bits
        return (v & 0x00FFFFFFu) | ((rng() & 0xFFu) << 24);
    }
}

// Run a short MCMC over the literal in prog.instrs[instr_idx].
// Returns true (and updates prog in-place) if the best found value improves
// base_fitness by >= 1%.  Otherwise leaves prog unchanged and returns false.
static bool mcmc_constant(Program& prog, int instr_idx,
                          double base_fitness, std::mt19937& rng) {
    constexpr int    STEPS   = 200;
    constexpr double T_START = 2.0;
    constexpr double T_END   = 1e-3;

    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    uint32_t cur_val  = prog.instrs[instr_idx].lit.i;
    double   cur_fit  = base_fitness;
    uint32_t best_val = cur_val;
    double   best_fit = cur_fit;

    for (int step = 0; step < STEPS; step++) {
        double t    = double(step) / STEPS;
        double temp = T_START * std::pow(T_END / T_START, t);

        uint32_t cand_val = perturb_lit(cur_val, rng);
        prog.instrs[instr_idx].lit.i = int32_t(cand_val);
        double cand_fit = fitness(prog);

        double delta = cand_fit - cur_fit;
        if (delta < 0.0 || uniform01(rng) < std::exp(-delta / temp)) {
            cur_val = cand_val;
            cur_fit = cand_fit;
        }

        if (cur_fit < best_fit) {
            best_fit = cur_fit;
            best_val = cur_val;
        }
    }

    prog.instrs[instr_idx].lit.i = int32_t(best_val);

    if (best_fit < base_fitness * 0.99) {
        // >= 1% improvement: keep it
        return true;
    }
    // No meaningful improvement: restore original and signal fallback
    prog.instrs[instr_idx].lit.i = int32_t(cur_val);
    return false;
}

Program mutate(const Program& src, const Hardness& hardness, std::mt19937& rng) {
    Program m = src;
    if (m.num_chroms == 0) { append_random_chromosome(m, rng); return m; }

    // kind_d raw [0,9] maps to weighted cases:
    //   0,1,2 -> 0  mutate one field    30%
    //   3,4   -> 6  strip dead          20%
    //   5     -> 5  MCMC constant       10%
    //   6     -> 1  swap chromosomes    10%
    //   7     -> 2  replace chromosome  10%
    //   8     -> 4  remove chromosome   10%
    //   9     -> 7  insert dep-split    10%
    std::uniform_int_distribution<int> kind_d(0, 9);
    std::uniform_int_distribution<int> ci_d  (0, m.num_chroms - 1);
    std::uniform_int_distribution<int> op_d  (0, int(Op::COUNT) - 1);
    std::uniform_int_distribution<int> reg_d (0, Program::NUM_REGS - 1);

    int raw  = kind_d(rng);
    int kind = (raw <= 2) ? 0 : (raw <= 4) ? 6 : (raw == 5) ? 5 :
               (raw == 6) ? 1 : (raw == 7) ? 2 : (raw == 8) ? 4 : 7;

    int ci   = ci_d(rng);
    int cs   = m.chrom_start(ci);
    int clen = m.chrom_lens[ci];

    bool live[Program::MAX_INSTRS];
    compute_dag(m, live);

    auto do_field_mutate = [&]() {
        if (clen == 0) return;
        Instr& ins = m.instrs[cs + pick_instr(cs, clen, live, hardness, rng)];
        switch (std::uniform_int_distribution<int>(0, 4)(rng)) {
        case 0: ins.op   = Op(op_d(rng));          break;
        case 1: ins.dst  = uint8_t(reg_d(rng));    break;
        case 2: ins.src1 = uint8_t(reg_d(rng));    break;
        case 3: ins.src2 = uint8_t(reg_d(rng));    break;
        case 4: ins.lit  = random_instr(rng).lit;  break;
        }
    };

    switch (kind) {
    case 0: { // mutate one instruction field, biased toward dead/soft instrs
        do_field_mutate();
        break;
    }
    case 1: { // swap two chromosomes (reorder)
        if (m.num_chroms < 2) break;
        int cj = ci_d(rng);
        while (cj == ci) cj = ci_d(rng);

        int lo = (ci < cj) ? ci : cj;
        int hi = (ci < cj) ? cj : ci;
        int lo_s = m.chrom_start(lo), lo_l = m.chrom_lens[lo];
        int hi_s = m.chrom_start(hi), hi_l = m.chrom_lens[hi];

        Program tmp = m;
        tmp.num_instrs = 0;
        for (int k = 0; k < m.num_chroms; k++) {
            int ks = m.chrom_start(k), kl = m.chrom_lens[k];
            const Instr* src_ptr;
            int           src_len;
            if      (k == lo) { src_ptr = m.instrs + hi_s; src_len = hi_l; }
            else if (k == hi) { src_ptr = m.instrs + lo_s; src_len = lo_l; }
            else              { src_ptr = m.instrs + ks;   src_len = kl;   }
            memcpy(tmp.instrs + tmp.num_instrs, src_ptr, src_len * sizeof(Instr));
            tmp.chrom_lens[k]  = uint8_t(src_len);
            tmp.num_instrs    += uint16_t(src_len);
        }
        m = tmp;
        break;
    }
    case 2: { // replace chromosome ci with a random one
        int new_len = std::uniform_int_distribution<int>(1, Program::MAX_CHROM_LEN)(rng);
        int delta   = new_len - clen;
        if (m.num_instrs + delta > Program::MAX_INSTRS) break;

        int tail = int(m.num_instrs) - cs - clen;
        memmove(m.instrs + cs + new_len, m.instrs + cs + clen, tail * sizeof(Instr));
        for (int i = 0; i < new_len; i++)
            m.instrs[cs + i] = random_instr(rng);
        m.chrom_lens[ci] = uint8_t(new_len);
        m.num_instrs     = uint16_t(m.num_instrs + delta);
        break;
    }
    case 4: { // remove chromosome ci
        if (m.num_chroms <= 1) break;
        int tail = int(m.num_instrs) - cs - clen;
        memmove(m.instrs + cs, m.instrs + cs + clen, tail * sizeof(Instr));
        memmove(m.chrom_lens + ci, m.chrom_lens + ci + 1,
                (m.num_chroms - ci - 1) * sizeof(uint8_t));
        m.num_chroms--;
        m.num_instrs = uint16_t(m.num_instrs - clen);
        break;
    }
    case 5: { // MCMC constant: pick a LOADI/LOADF, run short MCMC on its value
        int const_idxs[Program::MAX_INSTRS];
        int n_consts = 0;
        for (int i = 0; i < m.num_instrs; i++)
            if (m.instrs[i].op == Op::LOADI || m.instrs[i].op == Op::LOADF)
                const_idxs[n_consts++] = i;

        if (n_consts == 0) { do_field_mutate(); break; }

        {
            int target = const_idxs[std::uniform_int_distribution<int>(0, n_consts - 1)(rng)];
            double base = fitness(m);
            if (!mcmc_constant(m, target, base, rng)) do_field_mutate();
        }
        break;
    }
    case 6: { // strip dead instructions: pick a dead instr and remove it, up to 5 times
        int n = std::uniform_int_distribution<int>(0, 5)(rng);
        for (int pass = 0; pass < n; pass++) {
            compute_dag(m, live);
            int dead[Program::MAX_INSTRS];
            int ndead = 0;
            for (int i = 0; i < m.num_instrs; i++)
                if (!live[i]) dead[ndead++] = i;
            if (ndead == 0) break;

            int idx = dead[std::uniform_int_distribution<int>(0, ndead - 1)(rng)];

            // find which chromosome owns idx
            int dk = 0;
            for (int k = 0; k < m.num_chroms; k++) {
                int ks = m.chrom_start(k);
                if (idx >= ks && idx < ks + m.chrom_lens[k]) { dk = k; break; }
            }

            int tail = int(m.num_instrs) - idx - 1;
            memmove(m.instrs + idx, m.instrs + idx + 1, tail * sizeof(Instr));
            m.num_instrs--;
            m.chrom_lens[dk]--;

            if (m.chrom_lens[dk] == 0) {
                if (m.num_chroms > 1) {
                    memmove(m.chrom_lens + dk, m.chrom_lens + dk + 1,
                            (m.num_chroms - dk - 1) * sizeof(uint8_t));
                    m.num_chroms--;
                } else {
                    // last chromosome emptied — seed with one random instruction
                    m.instrs[0]     = random_instr(rng);
                    m.chrom_lens[0] = 1;
                    m.num_instrs    = 1;
                }
            }
        }
        break;
    }
    case 7: { // insert one instruction that splits a live dependency edge
        if (m.num_instrs >= Program::MAX_INSTRS) break;

        // pick a live instruction P whose output register R we will intercept
        int live_idxs[Program::MAX_INSTRS];
        int nlive = 0;
        for (int i = 0; i < m.num_instrs; i++)
            if (live[i]) live_idxs[nlive++] = i;
        if (nlive == 0) break;

        int P = live_idxs[std::uniform_int_distribution<int>(0, nlive - 1)(rng)];
        uint8_t R = m.instrs[P].dst % Program::NUM_REGS;

        // new instruction reads R as src1 and writes back to R,
        // so all downstream readers of R now see its (possibly transformed) output
        Instr ni = random_instr(rng);
        ni.src1 = R;
        ni.dst  = R;

        // find chromosome containing P
        int pk = 0;
        for (int k = 0; k < m.num_chroms; k++) {
            int ks = m.chrom_start(k);
            if (P >= ks && P < ks + m.chrom_lens[k]) { pk = k; break; }
        }

        int insert_at = P + 1;
        int tail = int(m.num_instrs) - insert_at;
        memmove(m.instrs + insert_at + 1, m.instrs + insert_at, tail * sizeof(Instr));
        m.instrs[insert_at] = ni;
        m.num_instrs++;

        if (m.chrom_lens[pk] < Program::MAX_CHROM_LEN) {
            m.chrom_lens[pk]++;
        } else if (m.num_chroms < Program::MAX_CHROMOSOMES) {
            // chromosome is full — open a new 1-instruction chromosome after pk
            memmove(m.chrom_lens + pk + 2, m.chrom_lens + pk + 1,
                    (m.num_chroms - pk - 1) * sizeof(uint8_t));
            m.chrom_lens[pk + 1] = 1;
            m.num_chroms++;
        } else {
            // no room anywhere — revert
            memmove(m.instrs + insert_at, m.instrs + insert_at + 1, tail * sizeof(Instr));
            m.num_instrs--;
        }
        break;
    }
    }
    return m;
}

Program crossover(const Program& a, const Program& b, std::mt19937& rng) {
    if (a.num_chroms == 0) return b;
    if (b.num_chroms == 0) return a;

    int cut_a = std::uniform_int_distribution<int>(0, a.num_chroms)(rng);
    int cut_b = std::uniform_int_distribution<int>(0, b.num_chroms)(rng);

    Program child = {};

    int a_instrs = a.chrom_start(cut_a);
    memcpy(child.instrs,     a.instrs,     a_instrs * sizeof(Instr));
    memcpy(child.chrom_lens, a.chrom_lens, cut_a    * sizeof(uint8_t));
    child.num_chroms = uint8_t(cut_a);
    child.num_instrs = uint16_t(a_instrs);

    int b_start  = b.chrom_start(cut_b);
    int b_instrs = b.num_instrs - b_start;
    int b_chroms = b.num_chroms - cut_b;
    int total_instrs = child.num_instrs + b_instrs;
    int total_chroms = child.num_chroms + b_chroms;

    if (total_instrs <= Program::MAX_INSTRS && total_chroms <= Program::MAX_CHROMOSOMES) {
        memcpy(child.instrs     + child.num_instrs,  b.instrs     + b_start, b_instrs * sizeof(Instr));
        memcpy(child.chrom_lens + child.num_chroms,  b.chrom_lens + cut_b,   b_chroms * sizeof(uint8_t));
        child.num_instrs = uint16_t(total_instrs);
        child.num_chroms = uint8_t(total_chroms);
    }

    if (child.num_chroms == 0)
        append_random_chromosome(child, rng);

    return child;
}
