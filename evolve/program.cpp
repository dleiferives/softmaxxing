#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cstring>

// ── Types ─────────────────────────────────────────────────────────────────────

enum class RegType : uint8_t { INT, FLOAT };

union RegVal {
    int32_t  i;
    uint32_t u;
    float    f;
};

struct Reg {
    RegType type = RegType::FLOAT;
    RegVal  val  = {};
};

// ── Operations ────────────────────────────────────────────────────────────────

enum class Op : uint8_t {
    IADD, ISUB, IMUL,
    FADD, FSUB, FMUL,
    BAND, BOR, BXOR,
    LSHL, LSHR,
    ASHL, ASHR,
    ILT,  IEQ,
    ULT,  UEQ,
    FLT,  FEQ,
    BNOT, LNOT, INEG, FNEG,
    ITF, FTI,
    LOADI, LOADF,
    MOV,
    COUNT
};

// ── Instruction ───────────────────────────────────────────────────────────────

struct Instr {
    Op      op   = Op::MOV;
    uint8_t dst  = 0;
    uint8_t src1 = 0;
    uint8_t src2 = 0;
    union { int32_t i; float f; } lit = {};
};

// ── Program ───────────────────────────────────────────────────────────────────
//
// Flat layout:  instrs[0 .. num_instrs-1]  run in order.
// Chromosome boundaries live in chrom_lens[0 .. num_chroms-1].
// Execution ignores chromosomes entirely — they're only for genetic ops.

struct Program {
    static constexpr int NUM_REGS        = 16;
    static constexpr int MAX_CHROMOSOMES = 64;
    static constexpr int MAX_CHROM_LEN   = 8;
    static constexpr int MAX_INSTRS      = MAX_CHROMOSOMES * MAX_CHROM_LEN; // 512

    Instr    instrs[MAX_INSTRS]          = {};
    uint8_t  chrom_lens[MAX_CHROMOSOMES] = {};
    uint8_t  num_chroms                  = 0;
    uint16_t num_instrs                  = 0;

    // O(num_chroms) but only called by genetic ops, not execution
    int chrom_start(int ci) const {
        int s = 0;
        for (int j = 0; j < ci; j++) s += chrom_lens[j];
        return s;
    }

    int total_instrs() const { return num_instrs; }
};

// ── Execution ─────────────────────────────────────────────────────────────────

float execute(const Program& prog, float input) {
    Reg regs[Program::NUM_REGS] = {};
    regs[0].type  = RegType::FLOAT;
    regs[0].val.f = input;

    const Instr* ins = prog.instrs;
    const Instr* end = ins + prog.num_instrs;

    for (; ins != end; ++ins) {
        const int di     = ins->dst  % Program::NUM_REGS;
        const RegVal& a  = regs[ins->src1 % Program::NUM_REGS].val;
        const RegVal& b  = regs[ins->src2 % Program::NUM_REGS].val;
        Reg& d           = regs[di];

        switch (ins->op) {
        case Op::IADD:  d.type=RegType::INT;   d.val.i = a.i + b.i;           break;
        case Op::ISUB:  d.type=RegType::INT;   d.val.i = a.i - b.i;           break;
        case Op::IMUL:  d.type=RegType::INT;   d.val.i = a.i * b.i;           break;
        case Op::FADD:  d.type=RegType::FLOAT; d.val.f = a.f + b.f;           break;
        case Op::FSUB:  d.type=RegType::FLOAT; d.val.f = a.f - b.f;           break;
        case Op::FMUL:  d.type=RegType::FLOAT; d.val.f = a.f * b.f;           break;
        case Op::BAND:  d.type=RegType::INT;   d.val.u = a.u & b.u;           break;
        case Op::BOR:   d.type=RegType::INT;   d.val.u = a.u | b.u;           break;
        case Op::BXOR:  d.type=RegType::INT;   d.val.u = a.u ^ b.u;           break;
        case Op::LSHL:  d.type=RegType::INT;   d.val.u = a.u << (b.u & 31u);  break;
        case Op::LSHR:  d.type=RegType::INT;   d.val.u = a.u >> (b.u & 31u);  break;
        case Op::ASHL:  d.type=RegType::INT;   d.val.i = a.i << (b.u & 31u);  break;
        case Op::ASHR:  d.type=RegType::INT;   d.val.i = a.i >> (b.u & 31u);  break;
        case Op::ILT:   d.type=RegType::INT;   d.val.i = a.i <  b.i ? 1 : 0; break;
        case Op::IEQ:   d.type=RegType::INT;   d.val.i = a.i == b.i ? 1 : 0; break;
        case Op::ULT:   d.type=RegType::INT;   d.val.i = a.u <  b.u ? 1 : 0; break;
        case Op::UEQ:   d.type=RegType::INT;   d.val.i = a.u == b.u ? 1 : 0; break;
        case Op::FLT:   d.type=RegType::INT;   d.val.i = a.f <  b.f ? 1 : 0; break;
        case Op::FEQ:   d.type=RegType::INT;   d.val.i = a.f == b.f ? 1 : 0; break;
        case Op::BNOT:  d.type=RegType::INT;   d.val.u = ~a.u;                break;
        case Op::LNOT:  d.type=RegType::INT;   d.val.i = a.i == 0 ? 1 : 0;   break;
        case Op::INEG:  d.type=RegType::INT;   d.val.i = -a.i;                break;
        case Op::FNEG:  d.type=RegType::FLOAT; d.val.f = -a.f;                break;
        case Op::ITF:   d.type=RegType::FLOAT; d.val.u = a.u;                 break;
        case Op::FTI:   d.type=RegType::INT;   d.val.u = a.u;                 break;
        case Op::LOADI: d.type=RegType::INT;   d.val.i = ins->lit.i;          break;
        case Op::LOADF: d.type=RegType::FLOAT; d.val.f = ins->lit.f;          break;
        case Op::MOV:   d = regs[ins->src1 % Program::NUM_REGS];              break;
        default: break;
        }
    }
    return regs[0].val.f;
}

// ── Fitness ───────────────────────────────────────────────────────────────────

static std::vector<float> make_test_inputs() {
    std::vector<float> v(100);
    for (int i = 0; i < 100; i++)
        v[i] = std::pow(10.0f, -2.0f + 4.0f * float(i) / 99.0f);
    return v;
}
static const std::vector<float> TEST_INPUTS = make_test_inputs();

double fitness(const Program& prog) {
    double err = 0.0;
    for (float x : TEST_INPUTS) {
        float got    = execute(prog, x);
        float target = 1.0f / std::sqrt(x);
        if (!std::isfinite(got)) { err += 1e6; continue; }
        double re = (double(got) - double(target)) / double(target);
        err += re * re;
    }
    return err / TEST_INPUTS.size();
}

// ── Random primitives ─────────────────────────────────────────────────────────

static const int32_t MAGIC_LITS[] = {
    0x5f3759df, 0x5f375a86, 0x5f400000,
    0x3f000000,  // 0.5f
    0x3f800000,  // 1.0f
    0x40000000,  // 2.0f
    0x3fc00000,  // 1.5f
};
static constexpr int N_MAGIC = sizeof(MAGIC_LITS) / sizeof(MAGIC_LITS[0]);

static Instr random_instr(std::mt19937& rng) {
    static std::uniform_int_distribution<int> op_d (0, int(Op::COUNT) - 1);
    static std::uniform_int_distribution<int> reg_d(0, Program::NUM_REGS - 1);
    static std::uniform_int_distribution<int> lk_d (0, 3);
    static std::uniform_int_distribution<int> mg_d (0, N_MAGIC - 1);

    Instr ins;
    ins.op   = Op(op_d(rng));
    ins.dst  = uint8_t(reg_d(rng));
    ins.src1 = uint8_t(reg_d(rng));
    ins.src2 = uint8_t(reg_d(rng));
    ins.lit.i = (lk_d(rng) == 0)
                    ? MAGIC_LITS[mg_d(rng)]
                    : int32_t(std::uniform_int_distribution<uint32_t>()(rng));
    return ins;
}

// Append a random chromosome to prog; returns false if at capacity
static bool append_random_chromosome(Program& prog, std::mt19937& rng) {
    if (prog.num_chroms >= Program::MAX_CHROMOSOMES) return false;
    std::uniform_int_distribution<int> len_d(1, Program::MAX_CHROM_LEN);
    int len = len_d(rng);
    if (prog.num_instrs + len > Program::MAX_INSTRS) return false;

    int ci = prog.num_chroms++;
    prog.chrom_lens[ci] = uint8_t(len);
    for (int i = 0; i < len; i++)
        prog.instrs[prog.num_instrs++] = random_instr(rng);
    return true;
}

static Program random_program(std::mt19937& rng) {
    Program prog = {};
    std::uniform_int_distribution<int> nc_d(1, 8); // start small
    int nc = nc_d(rng);
    for (int i = 0; i < nc; i++)
        append_random_chromosome(prog, rng);
    return prog;
}

// ── Mutation ──────────────────────────────────────────────────────────────────

static Program mutate(const Program& src, std::mt19937& rng) {
    Program m = src;
    if (m.num_chroms == 0) { append_random_chromosome(m, rng); return m; }

    std::uniform_int_distribution<int> kind_d(0, 4);
    std::uniform_int_distribution<int> ci_d  (0, m.num_chroms - 1);
    std::uniform_int_distribution<int> op_d  (0, int(Op::COUNT) - 1);
    std::uniform_int_distribution<int> reg_d (0, Program::NUM_REGS - 1);

    int kind = kind_d(rng);
    int ci   = ci_d(rng);
    int cs   = m.chrom_start(ci);
    int clen = m.chrom_lens[ci];

    switch (kind) {
    case 0: { // mutate one instruction field inside chromosome ci
        if (clen == 0) break;
        std::uniform_int_distribution<int> ii_d(0, clen - 1);
        Instr& ins = m.instrs[cs + ii_d(rng)];
        switch (std::uniform_int_distribution<int>(0, 4)(rng)) {
        case 0: ins.op   = Op(op_d(rng));         break;
        case 1: ins.dst  = uint8_t(reg_d(rng));   break;
        case 2: ins.src1 = uint8_t(reg_d(rng));   break;
        case 3: ins.src2 = uint8_t(reg_d(rng));   break;
        case 4: ins.lit  = random_instr(rng).lit; break;
        }
        break;
    }
    case 1: { // swap two chromosomes (reorder)
        if (m.num_chroms < 2) break;
        int cj = ci_d(rng);
        while (cj == ci) cj = ci_d(rng);
        int cjs = m.chrom_start(cj);
        int cjlen = m.chrom_lens[cj];
        // swap the instruction blocks and lengths
        // easiest: rebuild by copying into a temp buffer
        Program tmp = m;
        // copy ci's instrs to cj's slot and vice versa
        int new_cs  = m.chrom_start(cj < ci ? cj : ci);   // lower one first
        int lo = (ci < cj) ? ci : cj;
        int hi = (ci < cj) ? cj : ci;
        int lo_s = m.chrom_start(lo), lo_l = m.chrom_lens[lo];
        int hi_s = m.chrom_start(hi), hi_l = m.chrom_lens[hi];

        // rebuild from scratch into tmp
        tmp.num_instrs = 0;
        for (int k = 0; k < m.num_chroms; k++) {
            int ks = m.chrom_start(k);
            int kl = m.chrom_lens[k];
            const Instr* src_ptr;
            int           src_len;
            if (k == lo)      { src_ptr = m.instrs + hi_s; src_len = hi_l; }
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
        // remove old, insert new of random length
        int new_len = std::uniform_int_distribution<int>(1, Program::MAX_CHROM_LEN)(rng);
        int delta   = new_len - clen;
        if (m.num_instrs + delta > Program::MAX_INSTRS) break;

        // shift instructions after ci
        int tail = int(m.num_instrs) - cs - clen;
        memmove(m.instrs + cs + new_len, m.instrs + cs + clen,
                tail * sizeof(Instr));
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

        int ins_at = cs + clen; // instruction offset to insert at
        int tail   = int(m.num_instrs) - ins_at;
        memmove(m.instrs + ins_at + new_len, m.instrs + ins_at,
                tail * sizeof(Instr));
        for (int i = 0; i < new_len; i++)
            m.instrs[ins_at + i] = random_instr(rng);
        // shift chrom_lens array
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
    }
    return m;
}

// ── Crossover ─────────────────────────────────────────────────────────────────

// Single-point crossover at a chromosome boundary
static Program crossover(const Program& a, const Program& b, std::mt19937& rng) {
    if (a.num_chroms == 0) return b;
    if (b.num_chroms == 0) return a;

    std::uniform_int_distribution<int> pa(0, a.num_chroms);
    std::uniform_int_distribution<int> pb(0, b.num_chroms);
    int cut_a = pa(rng);  // take first cut_a chroms from a
    int cut_b = pb(rng);  // take last (num_chroms - cut_b) chroms from b

    Program child = {};

    // copy first cut_a chromosomes from a
    int a_instrs = a.chrom_start(cut_a);
    memcpy(child.instrs, a.instrs, a_instrs * sizeof(Instr));
    memcpy(child.chrom_lens, a.chrom_lens, cut_a * sizeof(uint8_t));
    child.num_chroms  = uint8_t(cut_a);
    child.num_instrs  = uint16_t(a_instrs);

    // append remaining chromosomes from b starting at cut_b
    int b_start  = b.chrom_start(cut_b);
    int b_instrs = b.num_instrs - b_start;
    int b_chroms = b.num_chroms - cut_b;

    int total_instrs = child.num_instrs + b_instrs;
    int total_chroms = child.num_chroms + b_chroms;

    if (total_instrs <= Program::MAX_INSTRS && total_chroms <= Program::MAX_CHROMOSOMES) {
        memcpy(child.instrs    + child.num_instrs,  b.instrs    + b_start,  b_instrs * sizeof(Instr));
        memcpy(child.chrom_lens + child.num_chroms, b.chrom_lens + cut_b,   b_chroms * sizeof(uint8_t));
        child.num_instrs = uint16_t(total_instrs);
        child.num_chroms = uint8_t(total_chroms);
    }

    if (child.num_chroms == 0)
        append_random_chromosome(child, rng);

    return child;
}

// ── Population / GA ───────────────────────────────────────────────────────────

struct Individual {
    Program prog;
    double  fit = std::numeric_limits<double>::max();
};

struct Population {
    static constexpr int    SIZE       = 128;
    static constexpr int    ELITE      = 8;
    static constexpr int    TOURNAMENT = 5;
    static constexpr double MUT_RATE   = 0.7;

    Individual indivs[SIZE];

    void init(std::mt19937& rng) {
        for (auto& ind : indivs) {
            ind.prog = random_program(rng);
            ind.fit  = fitness(ind.prog);
        }
        sort_pop();
    }

    void sort_pop() {
        std::sort(indivs, indivs + SIZE,
                  [](const Individual& a, const Individual& b){ return a.fit < b.fit; });
    }

    int select(std::mt19937& rng) const {
        std::uniform_int_distribution<int> d(0, SIZE - 1);
        int best = d(rng);
        for (int i = 1; i < TOURNAMENT; i++) {
            int c = d(rng);
            if (indivs[c].fit < indivs[best].fit) best = c;
        }
        return best;
    }

    void step(std::mt19937& rng) {
        // generate SIZE - ELITE new children into the bottom slots
        std::uniform_real_distribution<double> coin(0.0, 1.0);

        Individual next[SIZE];
        for (int i = 0; i < ELITE; i++) next[i] = indivs[i]; // elites

        for (int i = ELITE; i < SIZE; i++) {
            int p = select(rng);
            Program child;
            if (coin(rng) < MUT_RATE) {
                child = mutate(indivs[p].prog, rng);
            } else {
                int q = select(rng);
                child = crossover(indivs[p].prog, indivs[q].prog, rng);
            }
            next[i] = { child, fitness(child) };
        }

        memcpy(indivs, next, sizeof(indivs));
        sort_pop();
    }

    const Individual& best() const { return indivs[0]; }
};

// ── Pretty print ──────────────────────────────────────────────────────────────

static const char* op_str(Op op) {
    switch (op) {
    case Op::IADD:  return "IADD";  case Op::ISUB:  return "ISUB";
    case Op::IMUL:  return "IMUL";  case Op::FADD:  return "FADD";
    case Op::FSUB:  return "FSUB";  case Op::FMUL:  return "FMUL";
    case Op::BAND:  return "BAND";  case Op::BOR:   return "BOR";
    case Op::BXOR:  return "BXOR";  case Op::LSHL:  return "LSHL";
    case Op::LSHR:  return "LSHR";  case Op::ASHL:  return "ASHL";
    case Op::ASHR:  return "ASHR";  case Op::ILT:   return "ILT";
    case Op::IEQ:   return "IEQ";   case Op::ULT:   return "ULT";
    case Op::UEQ:   return "UEQ";   case Op::FLT:   return "FLT";
    case Op::FEQ:   return "FEQ";   case Op::BNOT:  return "BNOT";
    case Op::LNOT:  return "LNOT";  case Op::INEG:  return "INEG";
    case Op::FNEG:  return "FNEG";  case Op::ITF:   return "ITF";
    case Op::FTI:   return "FTI";   case Op::LOADI: return "LOADI";
    case Op::LOADF: return "LOADF"; case Op::MOV:   return "MOV";
    default: return "???";
    }
}

void print_program(const Program& prog) {
    for (int ci = 0; ci < prog.num_chroms; ci++) {
        std::cout << "  [chrom " << ci << "]\n";
        int cs = prog.chrom_start(ci);
        for (int ii = 0; ii < prog.chrom_lens[ci]; ii++) {
            const Instr& ins = prog.instrs[cs + ii];
            std::cout << "    " << op_str(ins.op)
                      << "  r" << int(ins.dst)
                      << "  r" << int(ins.src1)
                      << "  r" << int(ins.src2);
            if (ins.op == Op::LOADI)
                std::cout << "  0x" << std::hex << ins.lit.i << std::dec;
            if (ins.op == Op::LOADF)
                std::cout << "  " << ins.lit.f;
            std::cout << "\n";
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::mt19937 rng(std::random_device{}());

    std::cout << "Initialising population (size=" << Population::SIZE << ")...\n";

    static Population pop;  // static to avoid stack overflow (large struct)
    pop.init(rng);

    for (int gen = 1; ; gen++) {
        pop.step(rng);

        if (gen % 100 == 0) {
            const auto& b = pop.best();
            std::cout << "gen=" << gen
                      << "  fit=" << b.fit
                      << "  chroms=" << int(b.prog.num_chroms)
                      << "  instrs=" << b.prog.num_instrs
                      << "\n";
        }

        if (gen % 1000 == 0) {
            std::cout << "\n--- gen " << gen << " best ---\n";
            print_program(pop.best().prog);
            std::cout << "Sample outputs:\n";
            for (float x : {0.25f, 1.0f, 4.0f, 9.0f, 16.0f, 100.0f}) {
                float got    = execute(pop.best().prog, x);
                float target = 1.0f / std::sqrt(x);
                float err    = std::abs(got - target) / target * 100.0f;
                std::cout << "  x=" << x
                          << "  got=" << got
                          << "  target=" << target
                          << "  err=" << err << "%\n";
            }
            std::cout << "\n";
        }
    }
}
