#include "mutate.hpp"
#include "random.hpp"
#include "dag.hpp"
#include "fitness.hpp"
#include "innovation.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>

// ── Helpers ───────────────────────────────────────────────────────────────────

static constexpr float INACTIVE_BIAS = 3.0f;

static bool op_is_unary(Op op) {
    return op == Op::BNOT || op == Op::LNOT || op == Op::INEG ||
           op == Op::FNEG || op == Op::ITF  || op == Op::FTI  ||
           op == Op::MOV;
}

static bool op_is_terminal(Op op) {
    return op == Op::LOADI || op == Op::LOADF;
}

static bool op_supports_imm(Op op) {
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

// Weighted pick among function nodes [lo, hi).
// Inactive nodes get INACTIVE_BIAS × their hardness weight.
static int pick_node(const Program& m, int lo, int hi,
                     const bool live[], const Hardness& hardness,
                     std::mt19937& rng) {
    if (hi <= lo) return lo;
    if (hi - lo == 1) return lo;

    float w[Program::MAX_NODES];
    float total = 0.0f;
    for (int i = lo; i < hi; i++) {
        float base = live[i] ? 1.0f : INACTIVE_BIAS;
        w[i - lo] = base * hardness.weight(i);
        total += w[i - lo];
    }
    if (total <= 0.0f)
        return lo + std::uniform_int_distribution<int>(0, hi - lo - 1)(rng);

    float r = std::uniform_real_distribution<float>(0.0f, total)(rng);
    for (int i = lo; i < hi; i++) {
        r -= w[i - lo];
        if (r <= 0.0f) return i;
    }
    return hi - 1;
}

// Perturb a raw 32-bit literal with a small random change.
static uint32_t perturb_lit(uint32_t v, std::mt19937& rng) {
    switch (std::uniform_int_distribution<int>(0, 4)(rng)) {
    case 0: return v ^ (1u << (rng() & 31u));
    case 1: { uint32_t d = 1u << (rng() & 15u);
              return (rng() & 1) ? v + d : v - d; }
    case 2: return (rng() & 1) ? (v << 1) : (v >> 1);
    case 3: return (v & 0xFFFFFF00u) | (rng() & 0xFFu);
    default:return (v & 0x00FFFFFFu) | ((rng() & 0xFFu) << 24);
    }
}

// MCMC search over the literal of one LOADI/LOADF node.
// Returns true and keeps the best value found if it improves base_fitness.
// Otherwise restores the original value and returns false.
static bool mcmc_lit_node(Program& prog, int idx, double base_fitness,
                           std::mt19937& rng, const ProblemDef& problem,
                           const std::vector<float>& test_inputs) {
    constexpr int    STEPS   = 200;
    constexpr double T_START = 2.0;
    constexpr double T_END   = 1e-3;

    std::uniform_real_distribution<double> u01(0.0, 1.0);

    uint32_t orig_val;
    __builtin_memcpy(&orig_val, &prog.nodes[idx].lit, 4);

    uint32_t cur_val = orig_val, best_val = orig_val;
    double   cur_fit = base_fitness, best_fit = base_fitness;

    for (int step = 0; step < STEPS; step++) {
        double t    = double(step) / STEPS;
        double temp = T_START * std::pow(T_END / T_START, t);

        uint32_t cand = perturb_lit(cur_val, rng);
        prog.nodes[idx].lit.i = int32_t(cand);
        double cand_fit = fitness(prog, problem, test_inputs);

        double delta = cand_fit - cur_fit;
        if (delta < 0.0 || u01(rng) < std::exp(-delta / temp)) {
            cur_val = cand; cur_fit = cand_fit;
        }
        if (cur_fit < best_fit) { best_fit = cur_fit; best_val = cur_val; }
    }

    prog.nodes[idx].lit.i = int32_t(best_val);
    if (best_fit >= base_fitness) {
        prog.nodes[idx].lit.i = int32_t(orig_val);
        return false;
    }
    return true;
}

// ── mutate ────────────────────────────────────────────────────────────────────

Program mutate(const Program& src, const Hardness& hardness,
               int kind, std::mt19937& rng,
               const ProblemDef& problem,
               const std::vector<float>& test_inputs) {
    Program m = src;

    // Bootstrap: if there are no function nodes, just add one and return.
    if (m.n_nodes <= m.n_inputs) {
        add_random_node(m, rng);
        m.output_node = uint16_t(m.n_nodes - 1);
        return m;
    }

    bool live[Program::MAX_NODES];
    compute_live(m, live);

    const int fn_lo = m.n_inputs;      // first function node index
    const int fn_hi = int(m.n_nodes);  // one past last function node index

    std::uniform_int_distribution<int> op_d(0, int(Op::COUNT) - 1);

    switch (kind) {

    case 0: { // change_op — change a node's operation
        int idx = pick_node(m, fn_lo, fn_hi, live, hardness, rng);
        m.nodes[idx].op = Op(op_d(rng));
        // Fix src2 for ops that don't use it
        if (op_is_terminal(m.nodes[idx].op) || op_is_unary(m.nodes[idx].op))
            m.nodes[idx].src2 = 0;
        break;
    }

    case 1: { // rewire_src — redirect a src edge to a different upstream node
        // Collect function nodes that aren't pure terminals (they have src edges)
        int candidates[Program::MAX_NODES], n_cand = 0;
        for (int i = fn_lo; i < fn_hi; i++)
            if (!op_is_terminal(m.nodes[i].op))
                candidates[n_cand++] = i;
        if (n_cand == 0) break;

        int idx = candidates[std::uniform_int_distribution<int>(0, n_cand - 1)(rng)];
        Node& nd = m.nodes[idx];

        bool is_unary    = op_is_unary(nd.op);
        bool has_reg_src2 = !is_unary && nd.src2 != Program::IMM_SRC;
        bool do_src2      = has_reg_src2 && (rng() & 1);

        int n_avail = idx; // valid indices: [0, idx)
        if (n_avail <= 0) break;
        uint16_t new_src = uint16_t(std::uniform_int_distribution<int>(0, n_avail - 1)(rng));

        if (do_src2) nd.src2 = new_src;
        else         nd.src1 = new_src;
        break;
    }

    case 2: { // change_literal — perturb a LOADI/LOADF value
        int consts[Program::MAX_NODES], n_consts = 0;
        for (int i = fn_lo; i < fn_hi; i++)
            if (op_is_terminal(m.nodes[i].op))
                consts[n_consts++] = i;
        if (n_consts == 0) {
            // Fall back: change_op on a random function node
            int idx = pick_node(m, fn_lo, fn_hi, live, hardness, rng);
            m.nodes[idx].op = Op(op_d(rng));
            break;
        }
        int tgt = consts[std::uniform_int_distribution<int>(0, n_consts - 1)(rng)];
        uint32_t v; __builtin_memcpy(&v, &m.nodes[tgt].lit, 4);
        m.nodes[tgt].lit.i = int32_t(perturb_lit(v, rng));
        break;
    }

    case 3: { // mcmc_literal — MCMC search over a literal node's value
        int consts[Program::MAX_NODES], n_consts = 0;
        for (int i = fn_lo; i < fn_hi; i++)
            if (op_is_terminal(m.nodes[i].op))
                consts[n_consts++] = i;
        if (n_consts == 0) {
            int idx = pick_node(m, fn_lo, fn_hi, live, hardness, rng);
            m.nodes[idx].op = Op(op_d(rng));
            break;
        }
        int tgt  = consts[std::uniform_int_distribution<int>(0, n_consts - 1)(rng)];
        double base = fitness(m, problem, test_inputs);
        mcmc_lit_node(m, tgt, base, rng, problem, test_inputs);
        break;
    }

    case 4: { // insert_node — append a new random function node
        if (m.n_nodes >= Program::MAX_NODES) {
            // Fall back: change_op
            int idx = pick_node(m, fn_lo, fn_hi, live, hardness, rng);
            m.nodes[idx].op = Op(op_d(rng));
            break;
        }
        int new_idx = m.n_nodes;
        m.nodes[new_idx] = random_node(new_idx > 0 ? new_idx : 1, rng);
        m.nodes[new_idx].innov = next_innovation();
        m.n_nodes++;
        // 50% chance to make it the new output
        if (rng() & 1) m.output_node = uint16_t(new_idx);
        break;
    }

    case 5: { // remove_node — make a live function node inactive
        // Candidates: live function nodes that are not the output node
        int candidates[Program::MAX_NODES], n_cand = 0;
        for (int i = fn_lo; i < fn_hi; i++)
            if (live[i] && int(i) != int(m.output_node))
                candidates[n_cand++] = i;
        if (n_cand == 0) break;

        int idx = candidates[std::uniform_int_distribution<int>(0, n_cand - 1)(rng)];
        const Node& rem = m.nodes[idx];

        // Substitution: use rem's src1 (< idx, always valid)
        uint16_t sub = 0;
        if (!op_is_terminal(rem.op)) {
            sub = rem.src1;
        } else {
            // Terminal with no src — substitute the nearest prior node
            sub = uint16_t(idx > fn_lo ? idx - 1 : 0);
        }

        // Redirect all nodes that directly reference idx
        for (int j = idx + 1; j < fn_hi; j++) {
            if (m.nodes[j].src1 == uint16_t(idx)) m.nodes[j].src1 = sub;
            if (m.nodes[j].src2 != Program::IMM_SRC &&
                m.nodes[j].src2 == uint16_t(idx))   m.nodes[j].src2 = sub;
        }
        if (m.output_node == uint16_t(idx)) m.output_node = sub;
        // idx is now unreferenced → inactive
        break;
    }

    case 6: { // change_output — move output_node to a different node
        if (fn_hi <= fn_lo) break;
        int new_out = fn_lo + std::uniform_int_distribution<int>(0, fn_hi - fn_lo - 1)(rng);
        m.output_node = uint16_t(new_out);
        break;
    }

    case 7: { // redirect_to_inactive — activate an inactive node
        int inactive[Program::MAX_NODES], n_inactive = 0;
        for (int i = fn_lo; i < fn_hi; i++)
            if (!live[i]) inactive[n_inactive++] = i;

        if (n_inactive == 0) {
            // Nothing inactive: fall back to change_op
            int idx = pick_node(m, fn_lo, fn_hi, live, hardness, rng);
            m.nodes[idx].op = Op(op_d(rng));
            break;
        }

        int inact_idx = inactive[std::uniform_int_distribution<int>(0, n_inactive - 1)(rng)];

        // Find live nodes at higher indices that can reference inact_idx
        int consumers[Program::MAX_NODES], n_cons = 0;
        for (int i = inact_idx + 1; i < fn_hi; i++)
            if (live[i]) consumers[n_cons++] = i;

        if (n_cons == 0) {
            // Only the output node can act as consumer
            if (int(m.output_node) > inact_idx)
                m.output_node = uint16_t(inact_idx);
            break;
        }

        int cons_idx = consumers[std::uniform_int_distribution<int>(0, n_cons - 1)(rng)];
        Node& nd = m.nodes[cons_idx];

        bool is_unary    = op_is_unary(nd.op) || op_is_terminal(nd.op);
        bool has_reg_src2 = !is_unary && nd.src2 != Program::IMM_SRC;

        if (has_reg_src2 && (rng() & 1)) nd.src2 = uint16_t(inact_idx);
        else                              nd.src1 = uint16_t(inact_idx);
        break;
    }

    case 8: { // add_constant — append a new LOADI/LOADF terminal node
        if (m.n_nodes >= Program::MAX_NODES) break;
        int new_idx = m.n_nodes;
        Node& nd = m.nodes[new_idx];
        nd.op    = (rng() & 1) ? Op::LOADI : Op::LOADF;
        nd.lit.i = int32_t(std::uniform_int_distribution<uint32_t>()(rng));
        nd.src1  = 0;
        nd.src2  = 0;
        nd.innov = next_innovation();
        m.n_nodes++;
        // Starts inactive; redirect_to_inactive or rewire_src will activate it later
        break;
    }

    default: break;
    }

    return m;
}

// ── crossover ─────────────────────────────────────────────────────────────────
//
// Positional crossover: fitter parent's structure (n_nodes, output_node) is the
// backbone.  At each shared node index, 40% chance to take that node's fields
// from the weaker parent.  DAG validity is preserved because both parents
// maintain src < index at every position.

Program crossover(const Program& a, double fa,
                  const Program& b, double fb,
                  std::mt19937& rng) {
    if (a.n_nodes == 0) return b;
    if (b.n_nodes == 0) return a;

    const Program& fitter = (fa <= fb) ? a : b;
    const Program& weaker = (fa <= fb) ? b : a;

    Program child = fitter;

    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    int n_swap = std::min(int(fitter.n_nodes), int(weaker.n_nodes));

    for (int i = fitter.n_inputs; i < n_swap; i++) {
        if (coin(rng) < 0.4f) {
            child.nodes[i] = weaker.nodes[i];
            // Clamp src indices to be valid at this position in the child
            if (child.nodes[i].src1 >= uint16_t(i))
                child.nodes[i].src1 = uint16_t(i > 0 ? i - 1 : 0);
            if (child.nodes[i].src2 != Program::IMM_SRC &&
                child.nodes[i].src2 >= uint16_t(i))
                child.nodes[i].src2 = uint16_t(i > 0 ? i - 1 : 0);
        }
    }

    return child;
}
