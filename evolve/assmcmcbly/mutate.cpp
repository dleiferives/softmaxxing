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

    std::uniform_int_distribution<int> kind_d(0, 5);  // 0-4 existing + 5 = constant MCMC
    std::uniform_int_distribution<int> ci_d  (0, m.num_chroms - 1);
    std::uniform_int_distribution<int> op_d  (0, int(Op::COUNT) - 1);
    std::uniform_int_distribution<int> reg_d (0, Program::NUM_REGS - 1);

    int kind = kind_d(rng);
    int ci   = ci_d(rng);
    int cs   = m.chrom_start(ci);
    int clen = m.chrom_lens[ci];

    bool live[Program::MAX_INSTRS];
    compute_dag(m, live);

    switch (kind) {
    case 0: { // mutate one instruction field, biased toward dead/soft instrs
        if (clen == 0) break;
        Instr& ins = m.instrs[cs + pick_instr(cs, clen, live, hardness, rng)];
        switch (std::uniform_int_distribution<int>(0, 4)(rng)) {
        case 0: ins.op   = Op(op_d(rng));          break;
        case 1: ins.dst  = uint8_t(reg_d(rng));    break;
        case 2: ins.src1 = uint8_t(reg_d(rng));    break;
        case 3: ins.src2 = uint8_t(reg_d(rng));    break;
        case 4: ins.lit  = random_instr(rng).lit;  break;
        }
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
    case 3: { // insert a new random chromosome after ci
        if (m.num_chroms >= Program::MAX_CHROMOSOMES) break;
        int new_len = std::uniform_int_distribution<int>(1, Program::MAX_CHROM_LEN)(rng);
        if (m.num_instrs + new_len > Program::MAX_INSTRS) break;

        int ins_at = cs + clen;
        int tail   = int(m.num_instrs) - ins_at;
        memmove(m.instrs + ins_at + new_len, m.instrs + ins_at, tail * sizeof(Instr));
        for (int i = 0; i < new_len; i++)
            m.instrs[ins_at + i] = random_instr(rng);
        memmove(m.chrom_lens + ci + 2, m.chrom_lens + ci + 1,
                (m.num_chroms - ci - 1) * sizeof(uint8_t));
        m.chrom_lens[ci + 1] = uint8_t(new_len);
        m.num_chroms++;
        m.num_instrs = uint16_t(m.num_instrs + new_len);
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
    case 5: { // constant MCMC: find all LOADI/LOADF in the whole program,
              // pick one, run short MCMC on its value; fallback to kind 0 if < 1% gain
        int const_idxs[Program::MAX_INSTRS];
        int n_consts = 0;
        for (int i = 0; i < m.num_instrs; i++)
            if (m.instrs[i].op == Op::LOADI || m.instrs[i].op == Op::LOADF)
                const_idxs[n_consts++] = i;

        if (n_consts == 0) goto fallback_kind0;  // no constants → fall through

        {
            int target = const_idxs[std::uniform_int_distribution<int>(0, n_consts - 1)(rng)];
            double base = fitness(m);
            if (!mcmc_constant(m, target, base, rng)) goto fallback_kind0;
        }
        break;

    fallback_kind0:
        if (clen == 0) break;
        {
            Instr& ins = m.instrs[cs + pick_instr(cs, clen, live, hardness, rng)];
            switch (std::uniform_int_distribution<int>(0, 4)(rng)) {
            case 0: ins.op   = Op(op_d(rng));          break;
            case 1: ins.dst  = uint8_t(reg_d(rng));    break;
            case 2: ins.src1 = uint8_t(reg_d(rng));    break;
            case 3: ins.src2 = uint8_t(reg_d(rng));    break;
            case 4: ins.lit  = random_instr(rng).lit;  break;
            }
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
