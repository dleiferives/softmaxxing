#include "mutate.hpp"
#include "random.hpp"
#include "dag.hpp"
#include "fitness.hpp"
#include <vector>
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
                          double base_fitness, std::mt19937& rng,
                          const ProblemDef& problem,
                          const std::vector<float>& test_inputs) {
    constexpr int    STEPS   = 200;
    constexpr double T_START = 2.0;
    constexpr double T_END   = 1e-3;

    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    uint32_t orig_val = prog.instrs[instr_idx].lit.i;
    uint32_t cur_val  = orig_val;
    double   cur_fit  = base_fitness;
    uint32_t best_val = cur_val;
    double   best_fit = cur_fit;

    for (int step = 0; step < STEPS; step++) {
        double t    = double(step) / STEPS;
        double temp = T_START * std::pow(T_END / T_START, t);

        uint32_t cand_val = perturb_lit(cur_val, rng);
        prog.instrs[instr_idx].lit.i = int32_t(cand_val);
        double cand_fit = fitness(prog, problem, test_inputs);

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

    if (best_fit < base_fitness) {
        return true;
    }
    // No improvement: restore original and signal fallback
    prog.instrs[instr_idx].lit.i = int32_t(orig_val);
    return false;
}

// Collect the set of registers that are legal to read at instruction position `pos`.
// A register is legal if it is an input register (0..n_inputs-1) or has been
// written as a dst by some instruction before `pos`.
static void legal_srcs_at(const Program& p, int n_inputs, int pos,
                           uint8_t out[], int& n_out) {
    bool defined[Program::NUM_REGS] = {};
    for (int r = 0; r < n_inputs && r < Program::NUM_REGS; r++) defined[r] = true;
    for (int i = 0; i < pos && i < p.num_instrs; i++)
        defined[p.instrs[i].dst % Program::NUM_REGS] = true;
    n_out = 0;
    for (int r = 0; r < Program::NUM_REGS; r++)
        if (defined[r]) out[n_out++] = uint8_t(r);
}

static uint8_t pick_legal(const uint8_t legal[], int n_legal, std::mt19937& rng) {
    if (n_legal == 0) return 0;
    return legal[std::uniform_int_distribution<int>(0, n_legal - 1)(rng)];
}

Program mutate(const Program& src, const Hardness& hardness, std::mt19937& rng,
               const ProblemDef& problem, const std::vector<float>& test_inputs) {
    Program m = src;
    if (m.num_chroms == 0) { append_random_chromosome(m, rng); return m; }

    // kind_d raw [0,19] maps to weighted cases:
    //   0..3  -> 0  field mutate        20%
    //   4..5  -> 6  strip dead          10%
    //   6..7  -> 5  MCMC constant       10%
    //   8     -> 1  swap chromosomes     5%
    //   9     -> 2  replace chromosome   5%
    //   10    -> 4  remove chromosome    5%
    //   11..12-> 7  insert dep-split    10%
    //   13..15-> 8  add instruction     15%
    //   16..17-> 9  split chromosome    10%
    //   18..19->10  merge chromosome    10%
    std::uniform_int_distribution<int> kind_d(0, 19);
    std::uniform_int_distribution<int> ci_d  (0, m.num_chroms - 1);
    std::uniform_int_distribution<int> op_d  (0, int(Op::COUNT) - 1);
    std::uniform_int_distribution<int> reg_d (0, Program::NUM_REGS - 1);

    int raw  = kind_d(rng);
    int kind = (raw <= 3) ? 0 : (raw <= 5) ? 6  : (raw <= 7) ? 5  :
               (raw == 8) ? 1 : (raw == 9) ? 2  : (raw == 10) ? 4 :
               (raw <= 12) ? 7 : (raw <= 15) ? 8 : (raw <= 17) ? 9 : 10;

    int ci   = ci_d(rng);
    int cs   = m.chrom_start(ci);
    int clen = m.chrom_lens[ci];

    bool live[Program::MAX_INSTRS];
    compute_dag(m, live);

    auto do_field_mutate = [&]() {
        if (clen == 0) return;
        int local_pos = pick_instr(cs, clen, live, hardness, rng);
        int abs_pos   = cs + local_pos;
        Instr& ins = m.instrs[abs_pos];
        uint8_t legal[Program::NUM_REGS]; int n_legal;
        legal_srcs_at(m, problem.n_inputs, abs_pos, legal, n_legal);
        switch (std::uniform_int_distribution<int>(0, 4)(rng)) {
        case 0: ins.op   = Op(op_d(rng));                   break;
        case 1: ins.dst  = uint8_t(reg_d(rng));             break;
        case 2: ins.src1 = pick_legal(legal, n_legal, rng); break;
        case 3: ins.src2 = pick_legal(legal, n_legal, rng); break;
        case 4: ins.lit  = random_instr(rng).lit;           break;
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
        for (int i = 0; i < new_len; i++) {
            m.instrs[cs + i] = random_instr(rng);
            uint8_t legal[Program::NUM_REGS]; int n_legal;
            legal_srcs_at(m, problem.n_inputs, cs + i, legal, n_legal);
            m.instrs[cs + i].src1 = pick_legal(legal, n_legal, rng);
            m.instrs[cs + i].src2 = pick_legal(legal, n_legal, rng);
        }
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
            double base = fitness(m, problem, test_inputs);
            if (!mcmc_constant(m, target, base, rng, problem, test_inputs)) do_field_mutate();
        }
        break;
    }
    case 6: { // strip dead instructions: pick a dead instr and remove it, up to 5 times
        int n = std::uniform_int_distribution<int>(0, 2)(rng);
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
                    Instr seed_instr = random_instr(rng);
                    uint8_t legal[Program::NUM_REGS]; int n_legal;
                    legal_srcs_at(m, problem.n_inputs, 0, legal, n_legal);
                    seed_instr.src1 = pick_legal(legal, n_legal, rng);
                    seed_instr.src2 = pick_legal(legal, n_legal, rng);
                    m.instrs[0]     = seed_instr;
                    m.chrom_lens[0] = 1;
                    m.num_instrs    = 1;
                }
            }
        }
        break;
    }
    case 7: { // intercept a live dep-edge: insert instruction with fresh dst register
        if (m.num_instrs >= Program::MAX_INSTRS) break;

        // Pick a live instruction P whose output register R we will intercept.
        int live_idxs[Program::MAX_INSTRS];
        int nlive = 0;
        for (int i = 0; i < m.num_instrs; i++)
            if (live[i]) live_idxs[nlive++] = i;
        if (nlive == 0) break;

        int P = live_idxs[std::uniform_int_distribution<int>(0, nlive - 1)(rng)];
        uint8_t R = m.instrs[P].dst % Program::NUM_REGS;

        // Legal source registers at the insertion point (after P): inputs + all dsts 0..P.
        bool defined[Program::NUM_REGS] = {};
        for (int r = 0; r < problem.n_inputs && r < Program::NUM_REGS; r++) defined[r] = true;
        for (int i = 0; i <= P; i++) defined[m.instrs[i].dst % Program::NUM_REGS] = true;
        uint8_t legal[Program::NUM_REGS]; int n_legal = 0;
        for (int r = 0; r < Program::NUM_REGS; r++)
            if (defined[r]) legal[n_legal++] = uint8_t(r);

        // Try to allocate a fresh destination register (never written anywhere in the program).
        bool used_as_dst[Program::NUM_REGS] = {};
        for (int i = 0; i < m.num_instrs; i++)
            used_as_dst[m.instrs[i].dst % Program::NUM_REGS] = true;
        uint8_t fresh[Program::NUM_REGS]; int n_fresh = 0;
        for (int r = 0; r < Program::NUM_REGS; r++)
            if (!used_as_dst[r]) fresh[n_fresh++] = uint8_t(r);

        uint8_t new_dst = (n_fresh > 0)
            ? fresh[std::uniform_int_distribution<int>(0, n_fresh - 1)(rng)]
            : R; // no room — write back to R

        // Build the intercepting instruction.
        Instr ni    = random_instr(rng);
        ni.src1     = R;                              // reads P's output
        ni.src2     = pick_legal(legal, n_legal, rng); // any legal reg as second operand
        ni.dst      = new_dst;

        // Find chromosome containing P.
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
            memmove(m.chrom_lens + pk + 2, m.chrom_lens + pk + 1,
                    (m.num_chroms - pk - 1) * sizeof(uint8_t));
            m.chrom_lens[pk + 1] = 1;
            m.num_chroms++;
        } else {
            // no room anywhere — revert
            memmove(m.instrs + insert_at, m.instrs + insert_at + 1, tail * sizeof(Instr));
            m.num_instrs--;
            break;
        }

        // Redirect the first downstream instruction that reads R to use new_dst instead.
        if (new_dst != R) {
            for (int i = insert_at + 1; i < m.num_instrs; i++) {
                Instr& q = m.instrs[i];
                bool hit = false;
                if (q.src1 == R) { q.src1 = new_dst; hit = true; }
                if (q.src2 == R) { q.src2 = new_dst; hit = true; }
                if (hit) break;
            }
        }
        break;
    }
    case 8: { // add instruction: insert a random new instruction at a random position in ci
        if (m.num_instrs >= Program::MAX_INSTRS) break;
        if (clen >= Program::MAX_CHROM_LEN) break;

        // insert at a random position within the chromosome [cs, cs+clen]
        int pos = cs + std::uniform_int_distribution<int>(0, clen)(rng);
        int tail = int(m.num_instrs) - pos;
        memmove(m.instrs + pos + 1, m.instrs + pos, tail * sizeof(Instr));

        Instr ni = random_instr(rng);
        uint8_t legal[Program::NUM_REGS]; int n_legal;
        legal_srcs_at(m, problem.n_inputs, pos, legal, n_legal);
        ni.src1 = pick_legal(legal, n_legal, rng);
        ni.src2 = pick_legal(legal, n_legal, rng);
        // bias src1 toward the live register written just before insertion point
        if (pos > 0 && live[pos - 1])
            ni.src1 = m.instrs[pos - 1].dst % Program::NUM_REGS;
        m.instrs[pos] = ni;
        m.num_instrs++;
        m.chrom_lens[ci]++;
        break;
    }
    case 9: { // split chromosome: cut ci at a random internal point into two chromosomes
        if (clen < 2) break;
        if (m.num_chroms >= Program::MAX_CHROMOSOMES) break;

        int cut = std::uniform_int_distribution<int>(1, clen - 1)(rng);
        int lo_len = cut;
        int hi_len = clen - cut;

        // shift chrom_lens to open a slot after ci
        memmove(m.chrom_lens + ci + 2, m.chrom_lens + ci + 1,
                (m.num_chroms - ci - 1) * sizeof(uint8_t));
        m.chrom_lens[ci]     = uint8_t(lo_len);
        m.chrom_lens[ci + 1] = uint8_t(hi_len);
        m.num_chroms++;
        // instrs array is unchanged — the split is purely in chrom_lens
        break;
    }
    case 10: { // merge chromosome: join ci and ci+1 into one if they fit
        if (m.num_chroms < 2) break;
        int cj = (ci + 1) % m.num_chroms;  // wrap so last chrom can merge with first
        if (cj == 0) { ci = m.num_chroms - 1; cj = 0; }  // prefer ci < cj
        if (ci > cj) std::swap(ci, cj);

        int ci_len = m.chrom_lens[ci];
        int cj_len = m.chrom_lens[cj];
        if (ci_len + cj_len > Program::MAX_CHROM_LEN) break;

        // cj must immediately follow ci for a simple merge (no instr move needed
        // when they are already adjacent, which they are since we picked cj=ci+1)
        m.chrom_lens[ci] = uint8_t(ci_len + cj_len);
        memmove(m.chrom_lens + cj, m.chrom_lens + cj + 1,
                (m.num_chroms - cj - 1) * sizeof(uint8_t));
        m.num_chroms--;
        break;
    }
    }
    return m;
}

// Innovation-aligned crossover (NEAT-style).
// The fitter parent's structure (instruction order + chromosome boundaries) is the
// backbone.  At each instruction, if the weaker parent carries the same innovation
// number, we have a 40% chance to swap that instruction's fields in — keeping the
// innovation number itself unchanged so genomic distance stays correct.
// Excess/disjoint genes from the weaker parent are discarded; structural growth
// is handled by the dedicated growth mutations instead.
Program crossover(const Program& a, double fa, const Program& b, double fb, std::mt19937& rng) {
    if (a.num_chroms == 0) return b;
    if (b.num_chroms == 0) return a;

    const Program& fitter = (fa <= fb) ? a : b;
    const Program& weaker = (fa <= fb) ? b : a;

    Program child = fitter;  // start with fitter parent's structure

    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    int nw = weaker.num_instrs;
    for (int i = 0; i < child.num_instrs; i++) {
        uint32_t innov = child.instrs[i].innov;
        for (int j = 0; j < nw; j++) {
            if (weaker.instrs[j].innov == innov) {
                if (coin(rng) < 0.4f) {
                    // swap all fields except innov
                    Instr tmp        = weaker.instrs[j];
                    tmp.innov        = innov;
                    child.instrs[i]  = tmp;
                }
                break;
            }
        }
    }

    return child;
}
