#include "mutate.hpp"
#include "random.hpp"
#include "dag.hpp"
#include "fitness.hpp"
#include <vector>
#include <cstring>
#include <cmath>

static constexpr float DEAD_WEIGHT = 10.0f;

// Pick an instruction index over the whole program, weighted by liveness + hardness.
static int pick_instr(const Program& m, const bool live[],
                      const Hardness& hardness, std::mt19937& rng) {
    if (m.num_instrs == 1) return 0;
    float w[Program::MAX_INSTRS];
    float total = 0.0f;
    for (int i = 0; i < m.num_instrs; i++) {
        float base = live[i] ? 1.0f : DEAD_WEIGHT;
        w[i]  = base;//  * hardness.weight(i);
        total += w[i];
    }
    float r = std::uniform_real_distribution<float>(0.0f, total)(rng);
    for (int i = 0; i < m.num_instrs; i++) {
        r -= w[i];
        if (r <= 0.0f) return i;
    }
    return int(m.num_instrs) - 1;
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
    prog.instrs[instr_idx].lit.i = int32_t(orig_val);
    return false;
}

// Collect the set of registers that are legal to read at instruction position `pos`.
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

static bool op_supports_imm_src2(Op op) {
    switch (op) {
    case Op::IADD: case Op::ISUB: case Op::IMUL:
    case Op::BAND: case Op::BOR:  case Op::BXOR:
    case Op::LSHL: case Op::LSHR: case Op::ASHL: case Op::ASHR:
    case Op::ILT:  case Op::IEQ:  case Op::ULT:  case Op::UEQ:
        return true;
    default:
        return false;
    }
}

Program mutate(const Program& src, const Hardness& hardness, std::mt19937& rng,
               const ProblemDef& problem, const std::vector<float>& test_inputs) {
    Program m = src;

    if (m.num_instrs == 0) {
        Instr si = random_instr(rng);
        uint8_t legal[Program::NUM_REGS]; int n_legal;
        legal_srcs_at(m, problem.n_inputs, 0, legal, n_legal);
        si.src1 = pick_legal(legal, n_legal, rng);
        si.src2 = pick_legal(legal, n_legal, rng);
        m.instrs[0]  = si;
        m.num_instrs = 1;
        return m;
    }

    // Mutation kind weights over [0,19]:
    //   0..3  -> 0  field mutate         20%
    //   4..5  -> 1  strip dead           10%
    //   6..7  -> 2  MCMC constant        10%
    //   8..9  -> 3  dep-split insert     10%
    //   10..12-> 4  add instruction      15%
    //   13..15-> 5  remove instruction(s)15%
    //   16..19-> 6  replace slice        20%
    int raw  = std::uniform_int_distribution<int>(0, 19)(rng);
    int kind = (raw <= 3) ? 0 : (raw <= 5) ? 1 : (raw <= 7) ? 2 :
               (raw <= 9) ? 3 : (raw <= 12) ? 4 : (raw <= 15) ? 5 : 6;

    std::uniform_int_distribution<int> op_d (0, int(Op::COUNT) - 1);
    std::uniform_int_distribution<int> reg_d(0, Program::NUM_REGS - 1);

    bool live[Program::MAX_INSTRS];
    compute_dag(m, live);

    auto do_field_mutate = [&]() {
        if (m.num_instrs == 0) return;
        int abs_pos = pick_instr(m, live, hardness, rng);
        Instr& ins  = m.instrs[abs_pos];
        uint8_t legal[Program::NUM_REGS]; int n_legal;
        legal_srcs_at(m, problem.n_inputs, abs_pos, legal, n_legal);
        switch (std::uniform_int_distribution<int>(0, 4)(rng)) {
        case 0: ins.op   = Op(op_d(rng));                   break;
        case 1: ins.dst  = uint8_t(reg_d(rng));             break;
        case 2: ins.src1 = pick_legal(legal, n_legal, rng); break;
        case 3:
            if (op_supports_imm_src2(ins.op)) {
                if (ins.src2 == Program::IMM_SRC) {
                    if (rng() & 1) {
                        uint32_t v; memcpy(&v, &ins.lit, 4);
                        uint32_t nv = perturb_lit(v, rng);
                        ins.lit.i = int32_t(nv);
                    } else {
                        ins.src2 = pick_legal(legal, n_legal, rng);
                    }
                } else {
                    if (rng() & 1) {
                        ins.src2 = pick_legal(legal, n_legal, rng);
                    } else {
                        ins.src2  = Program::IMM_SRC;
                        ins.lit.i = int32_t(rng());
                    }
                }
            } else {
                ins.src2 = pick_legal(legal, n_legal, rng);
            }
            break;
        case 4: ins.lit = random_instr(rng).lit; break;
        }
    };

    switch (kind) {
    case 0: { // mutate one instruction field, biased toward dead/soft instrs
        do_field_mutate();
        break;
    }
    case 1: { // remove dead instructions (1-3 passes, weighted toward dead)
        int n = std::uniform_int_distribution<int>(1, 3)(rng);
        for (int pass = 0; pass < n; pass++) {
            if (m.num_instrs == 0) break;
            compute_dag(m, live);

            float weights[Program::MAX_INSTRS];
            float total_w = 0.0f;
            for (int i = 0; i < m.num_instrs; i++) {
                weights[i] = live[i] ? 1.0f : 8.0f;
                total_w += weights[i];
            }
            float r2 = std::uniform_real_distribution<float>(0.0f, total_w)(rng);
            int idx = int(m.num_instrs) - 1;
            for (int i = 0; i < m.num_instrs; i++) {
                r2 -= weights[i]; if (r2 <= 0.0f) { idx = i; break; }
            }

            const Instr& rem = m.instrs[idx];
            uint8_t R = rem.dst % Program::NUM_REGS;
            bool is_literal = (rem.op == Op::LOADI || rem.op == Op::LOADF);

            if (!is_literal) {
                bool is_unary = (rem.op == Op::BNOT || rem.op == Op::LNOT  ||
                                 rem.op == Op::INEG || rem.op == Op::FNEG  ||
                                 rem.op == Op::ITF  || rem.op == Op::FTI   ||
                                 rem.op == Op::MOV);
                uint8_t sub_reg = is_unary
                    ? (rem.src1 % Program::NUM_REGS)
                    : ((rng() & 1) ? (rem.src1 % Program::NUM_REGS)
                                   : (rem.src2 % Program::NUM_REGS));
                for (int i = idx + 1; i < m.num_instrs; i++) {
                    if (m.instrs[i].dst % Program::NUM_REGS == R) break;
                    if (m.instrs[i].src1 % Program::NUM_REGS == R) m.instrs[i].src1 = sub_reg;
                    if (m.instrs[i].src2 % Program::NUM_REGS == R) m.instrs[i].src2 = sub_reg;
                }
            }

            int tail = int(m.num_instrs) - idx - 1;
            memmove(m.instrs + idx, m.instrs + idx + 1, tail * sizeof(Instr));
            m.num_instrs--;

            if (m.num_instrs == 0) {
                Instr seed_instr = random_instr(rng);
                uint8_t legal[Program::NUM_REGS]; int n_legal;
                legal_srcs_at(m, problem.n_inputs, 0, legal, n_legal);
                seed_instr.src1 = pick_legal(legal, n_legal, rng);
                seed_instr.src2 = pick_legal(legal, n_legal, rng);
                m.instrs[0]  = seed_instr;
                m.num_instrs = 1;
            }
        }
        break;
    }
    case 2: { // MCMC constant: pick a LOADI/LOADF, run short MCMC on its value
        int const_idxs[Program::MAX_INSTRS];
        int n_consts = 0;
        for (int i = 0; i < m.num_instrs; i++)
            if (m.instrs[i].op == Op::LOADI || m.instrs[i].op == Op::LOADF)
                const_idxs[n_consts++] = i;

        if (n_consts == 0) { do_field_mutate(); break; }

        int target = const_idxs[std::uniform_int_distribution<int>(0, n_consts - 1)(rng)];
        double base = fitness(m, problem, test_inputs);
        if (!mcmc_constant(m, target, base, rng, problem, test_inputs)) do_field_mutate();
        break;
    }
    case 3: { // intercept a live dep-edge: insert instruction with fresh dst register
        if (m.num_instrs >= Program::MAX_INSTRS) break;

        int live_idxs[Program::MAX_INSTRS];
        int nlive = 0;
        for (int i = 0; i < m.num_instrs; i++)
            if (live[i]) live_idxs[nlive++] = i;
        if (nlive == 0) break;

        int P = live_idxs[std::uniform_int_distribution<int>(0, nlive - 1)(rng)];
        uint8_t R = m.instrs[P].dst % Program::NUM_REGS;

        bool defined[Program::NUM_REGS] = {};
        for (int r = 0; r < problem.n_inputs && r < Program::NUM_REGS; r++) defined[r] = true;
        for (int i = 0; i <= P; i++) defined[m.instrs[i].dst % Program::NUM_REGS] = true;
        uint8_t legal[Program::NUM_REGS]; int n_legal = 0;
        for (int r = 0; r < Program::NUM_REGS; r++)
            if (defined[r]) legal[n_legal++] = uint8_t(r);

        bool used_as_dst[Program::NUM_REGS] = {};
        for (int i = 0; i < m.num_instrs; i++)
            used_as_dst[m.instrs[i].dst % Program::NUM_REGS] = true;
        uint8_t fresh[Program::NUM_REGS]; int n_fresh = 0;
        for (int r = 0; r < Program::NUM_REGS; r++)
            if (!used_as_dst[r]) fresh[n_fresh++] = uint8_t(r);

        uint8_t new_dst = (n_fresh > 0)
            ? fresh[std::uniform_int_distribution<int>(0, n_fresh - 1)(rng)]
            : R;

        Instr ni = random_instr(rng);
        ni.src1  = R;
        ni.src2  = pick_legal(legal, n_legal, rng);
        ni.dst   = new_dst;

        int insert_at = P + 1;
        int tail = int(m.num_instrs) - insert_at;
        memmove(m.instrs + insert_at + 1, m.instrs + insert_at, tail * sizeof(Instr));
        m.instrs[insert_at] = ni;
        m.num_instrs++;

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
    case 4: { // add instruction at a random position
        if (m.num_instrs >= Program::MAX_INSTRS) break;

        int pos = std::uniform_int_distribution<int>(0, int(m.num_instrs))(rng);
        int tail = int(m.num_instrs) - pos;
        memmove(m.instrs + pos + 1, m.instrs + pos, tail * sizeof(Instr));

        Instr ni = random_instr(rng);
        uint8_t legal[Program::NUM_REGS]; int n_legal;
        legal_srcs_at(m, problem.n_inputs, pos, legal, n_legal);
        ni.src1 = pick_legal(legal, n_legal, rng);
        ni.src2 = pick_legal(legal, n_legal, rng);
        if (pos > 0 && live[pos - 1])
            ni.src1 = m.instrs[pos - 1].dst % Program::NUM_REGS;
        m.instrs[pos] = ni;
        m.num_instrs++;
        break;
    }
    case 5: { // remove 1-3 contiguous instructions
        int n = std::uniform_int_distribution<int>(1, 3)(rng);
        n = std::min(n, int(m.num_instrs) - 1);  // keep at least 1
        if (n <= 0) break;
        int start = std::uniform_int_distribution<int>(0, int(m.num_instrs) - n)(rng);
        int tail  = int(m.num_instrs) - start - n;
        memmove(m.instrs + start, m.instrs + start + n, tail * sizeof(Instr));
        m.num_instrs = uint16_t(m.num_instrs - n);
        if (m.num_instrs == 0) {
            m.instrs[0]  = random_instr(rng);
            m.num_instrs = 1;
        }
        break;
    }
    case 6: { // replace a random slice with fresh random instructions
        if (m.num_instrs == 0) break;
        static constexpr int MAX_SLICE = 8;
        int start   = std::uniform_int_distribution<int>(0, int(m.num_instrs) - 1)(rng);
        int old_len = std::uniform_int_distribution<int>(1, std::min(MAX_SLICE, int(m.num_instrs) - start))(rng);
        int new_len = std::uniform_int_distribution<int>(1, MAX_SLICE)(rng);
        int delta   = new_len - old_len;
        if (int(m.num_instrs) + delta > Program::MAX_INSTRS) break;
        if (int(m.num_instrs) + delta < 1) break;

        int tail = int(m.num_instrs) - start - old_len;
        memmove(m.instrs + start + new_len, m.instrs + start + old_len, tail * sizeof(Instr));
        for (int i = 0; i < new_len; i++) {
            m.instrs[start + i] = random_instr(rng);
            uint8_t legal[Program::NUM_REGS]; int n_legal;
            legal_srcs_at(m, problem.n_inputs, start + i, legal, n_legal);
            m.instrs[start + i].src1 = pick_legal(legal, n_legal, rng);
            m.instrs[start + i].src2 = pick_legal(legal, n_legal, rng);
        }
        m.num_instrs = uint16_t(int(m.num_instrs) + delta);
        break;
    }
    }
    return m;
}

// Build a child from prefix a[0..cut_a) + suffix b[cut_b..end), fixing undefined srcs.
static Program make_positional_child(const Program& a, const Program& b,
                                     int cut_a, int cut_b) {
    int plen = cut_a;
    int slen = b.num_instrs - cut_b;
    if (plen + slen > Program::MAX_INSTRS) slen = Program::MAX_INSTRS - plen;

    Program child = {};
    memcpy(child.instrs,        a.instrs,         plen * sizeof(Instr));
    memcpy(child.instrs + plen, b.instrs + cut_b, slen * sizeof(Instr));
    child.num_instrs = uint16_t(plen + slen);

    bool defined[Program::NUM_REGS] = {};
    defined[0] = true;
    for (int i = 0; i < plen; i++)
        defined[child.instrs[i].dst % Program::NUM_REGS] = true;
    uint8_t fallback = (plen > 0) ? uint8_t(child.instrs[plen-1].dst % Program::NUM_REGS) : 0;

    for (int i = plen; i < child.num_instrs; i++) {
        Instr& ins = child.instrs[i];
        bool is_leaf = (ins.op == Op::LOADI || ins.op == Op::LOADF);
        if (!is_leaf) {
            if (!defined[ins.src1 % Program::NUM_REGS]) ins.src1 = fallback;
            if (ins.src2 != Program::IMM_SRC &&
                !defined[ins.src2 % Program::NUM_REGS]) ins.src2 = fallback;
        }
        defined[ins.dst % Program::NUM_REGS] = true;
    }
    return child;
}

// Positional crossover with hardness-biased cut selection.
//
// Two complementary biases:
//  1. Cut point sampling: weight cut_a by hardness of the instruction just BEFORE
//     the cut (prefer to cut after a hard instruction, capturing it in the prefix).
//     Weight cut_b by hardness of the instruction AT cut_b (prefer to start B's suffix
//     right before a hard instruction, capturing it in the suffix).
//  2. Candidate scoring: generate N_CANDS cuts (mix of biased + uniform), score each
//     by total hardness of live inherited instructions, return the best-scoring child.
Program crossover_positional(const Program& a, const Hardness& ha,
                              const Program& b, const Hardness& hb,
                              std::mt19937& rng) {
    if (a.num_instrs == 0) return b;
    if (b.num_instrs == 0) return a;

    const int na = a.num_instrs;
    const int nb = b.num_instrs;

    // Score a candidate child: sum hardness of its live instructions,
    // sourcing scores from the parent that contributed each instruction.
    auto score_child = [&](const Program& child, int plen, int cut_b) -> float {
        bool live[Program::MAX_INSTRS];
        compute_dag(child, live);
        float s = 0.0f;
        for (int i = 0; i < plen; i++)
            if (live[i]) s += ha.scores[i];
        int slen = child.num_instrs - plen;
        for (int i = 0; i < slen; i++)
            if (live[plen + i]) s += hb.scores[cut_b + i];
        return s;
    };

    // Hardness-biased cut_a: weight[i] = hardness of instr i-1 (just before cut) + epsilon.
    // This prefers cuts that land right after a high-hardness instruction.
    auto sample_cut_a = [&]() -> int {
        float weights[Program::MAX_INSTRS + 1];
        float total = 0.0f;
        for (int i = 0; i <= na; i++) {
            weights[i] = (i > 0 ? ha.scores[i-1] : 0.0f) + 0.5f;
            total += weights[i];
        }
        float r = std::uniform_real_distribution<float>(0.0f, total)(rng);
        for (int i = 0; i <= na; i++) { r -= weights[i]; if (r <= 0.0f) return i; }
        return na;
    };

    // Hardness-biased cut_b: weight[j] = hardness of instr j (start of suffix) + epsilon.
    // This prefers starting B's suffix right before a high-hardness instruction.
    auto sample_cut_b = [&]() -> int {
        float weights[Program::MAX_INSTRS + 1];
        float total = 0.0f;
        for (int j = 0; j <= nb; j++) {
            weights[j] = (j < nb ? hb.scores[j] : 0.0f) + 0.5f;
            total += weights[j];
        }
        float r = std::uniform_real_distribution<float>(0.0f, total)(rng);
        for (int j = 0; j <= nb; j++) { r -= weights[j]; if (r <= 0.0f) return j; }
        return nb;
    };

    static constexpr int N_CANDS = 5;
    Program best_child = {};
    float   best_score = -1.0f;
    bool    have_best  = false;

    for (int k = 0; k < N_CANDS; k++) {
        int cut_a, cut_b;
        if (k < 3) {
            // Hardness-biased candidates
            cut_a = sample_cut_a();
            cut_b = sample_cut_b();
        } else {
            // Uniform random candidates for diversity
            cut_a = std::uniform_int_distribution<int>(0, na)(rng);
            cut_b = std::uniform_int_distribution<int>(0, nb)(rng);
        }

        int plen = cut_a;
        int slen = nb - cut_b;
        if (plen + slen <= 0 || plen + slen > Program::MAX_INSTRS) continue;

        Program cand  = make_positional_child(a, b, cut_a, cut_b);
        float   score = score_child(cand, plen, cut_b);
        if (!have_best || score > best_score) {
            best_score = score;
            best_child = cand;
            have_best  = true;
        }
    }

    if (!have_best) return (na >= nb) ? a : b;
    return best_child;
}

// Innovation-aligned crossover (NEAT-style).
// The fitter parent's structure is the backbone. At each instruction, if the
// weaker parent carries the same innovation number, 40% chance to swap in its fields.
Program crossover(const Program& a, double fa, const Program& b, double fb, std::mt19937& rng) {
    if (a.num_instrs == 0) return b;
    if (b.num_instrs == 0) return a;

    const Program& fitter = (fa <= fb) ? a : b;
    const Program& weaker = (fa <= fb) ? b : a;

    Program child = fitter;

    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    int nw = weaker.num_instrs;
    for (int i = 0; i < child.num_instrs; i++) {
        uint32_t innov = child.instrs[i].innov;
        for (int j = 0; j < nw; j++) {
            if (weaker.instrs[j].innov == innov) {
                if (coin(rng) < 0.4f) {
                    Instr tmp       = weaker.instrs[j];
                    tmp.innov       = innov;
                    child.instrs[i] = tmp;
                }
                break;
            }
        }
    }

    return child;
}
