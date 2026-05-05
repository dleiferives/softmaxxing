# Implementation Plan

## Goals

1. **SSA representation** — eliminate register-naming aliasing; shrink search space
2. **Batch JIT** — evaluate all population members for one input in one compiled blob; decouple from single-program path
3. **Decouple gen/eval** — generate all candidates, then batch-compile and batch-evaluate
4. **Constant MCMC only** — no random lit perturbation in normal mutation; instead MCMC-solve all constants when triggered
5. **Two-tier evaluation** — fast small test set for all candidates; full test set only for survivors
6. **Header reorganisation** — all `.hpp` files move to `include/`

## Non-Goals

- Removing the novelty/frontier system (it's the main thing keeping search from stalling)
- Removing `speciation.cpp` (`compute_fingerprint` is called by the novelty system)
- Removing the LRU eval cache (still useful — avoids re-evaluating identical programs)
- Removing the hot-burst / global-stagnation mechanism

---

## 1. New Data Representation

### 1.1 Slot Layout (the core SSA idea)

Fixed constants (compile-time, in `program.hpp`):

```
N_MAX_INPUTS = 4      // max inputs any problem can use; pads unused ones with 0
MAX_INSTRS   = 24     // max instructions per program
TOTAL_SLOTS  = N_MAX_INPUTS + MAX_INSTRS = 28
```

Slot semantics:
- Slots `0 .. N_MAX_INPUTS-1`: input values (read-only during execution; slot 0 = x for current 1-input problem; unused slots hold 0.0f)
- Slots `N_MAX_INPUTS .. N_MAX_INPUTS+MAX_INSTRS-1`: instruction outputs (instruction `i` writes to slot `N_MAX_INPUTS + i`)
- **Program output** = value in slot `N_MAX_INPUTS + num_instrs - 1` (always the last instruction)

Key invariant: **input slots are never written by any instruction**. This is what makes batch JIT scratch-space reuse safe without re-zeroing between programs.

### 1.2 New `Instr` struct (8 bytes, down from 12)

```cpp
struct Instr {
    Op      op;      // 1 byte — unchanged
    uint8_t src1;    // slot index in [0, N_MAX_INPUTS + position)
    uint8_t src2;    // slot index same range, or IMM_SRC = 0xFF
    uint8_t _pad;    // reserved
    uint32_t lit;    // immediate bits (reinterpret as int32 or float)
};
// Dropped: dst (implicit = N_MAX_INPUTS + i), innov (NEAT crossover removed)
```

Valid src values for instruction at position `i`:
- `[0, N_MAX_INPUTS + i)` — any input slot or any earlier instruction's output
- `IMM_SRC (0xFF)` for src2 only — uses `lit` as immediate

This makes mutation trivially correct: `rng() % (N_MAX_INPUTS + i)` always produces a valid source. No more `legal_srcs_at()`.

### 1.3 New `Program` struct

```cpp
struct Program {
    static constexpr int     N_MAX_INPUTS = 4;
    static constexpr int     MAX_INSTRS   = 24;
    static constexpr int     TOTAL_SLOTS  = N_MAX_INPUTS + MAX_INSTRS;
    static constexpr uint8_t IMM_SRC      = 0xFF;

    Instr    instrs[MAX_INSTRS] = {};
    uint16_t num_instrs         = 0;
};
```

### 1.4 Types Cleanup (`types.hpp`)

**Remove**: `RegType`, `Reg` — these tracked int/float type tags per register. In SSA, the type of each slot is fully determined by the op that wrote it (known at JIT-compile time; implicit in the interpreter). Remove entirely.

**Keep**: `Op` enum unchanged (all ops still valid), `Instr` redefined as above.

### 1.5 What Gets Removed

| Item | Where | Reason |
|---|---|---|
| `Instr::dst` | types.hpp | Implicit: slot = `N_MAX_INPUTS + i` |
| `Instr::innov` | types.hpp | NEAT crossover dropped |
| `RegType`, `Reg` | types.hpp | Type implicit from op |
| `hardening.hpp/cpp` | src/ | `recompute()` commented out everywhere; scores always zero |
| `innovation.hpp/cpp` | src/ | Only served innov field and NEAT crossover |
| `crossover()` (NEAT) | mutate.cpp | Replaced by positional crossover |
| `crossover_positional()` with hardness | mutate.cpp | Replaced by simpler positional crossover |
| `legal_srcs_at()` | mutate.cpp | Not needed in SSA |
| `Hardness hardness` in `Individual` | population.hpp | Hardening removed |
| `fitness_and_cases()` batch path | fitness.cpp | Moves to batch JIT in population |

---

## 2. File Structure

```
assmcmcbly/
  include/           ← ALL .hpp files move here
    types.hpp        ← modified (remove RegType, Reg; redefine Instr)
    program.hpp      ← modified (N_MAX_INPUTS, TOTAL_SLOTS; no dst/innov)
    dag.hpp          ← unchanged interface, simpler implementation
    execute.hpp      ← interface unchanged externally
    jit.hpp          ← unchanged (single-program JIT stays)
    batch_jit.hpp    ← NEW
    fitness.hpp      ← modified (remove inline globals, simplify)
    mutate.hpp       ← modified (remove Hardness param, remove crossover_positional)
    population.hpp   ← modified (remove Hardness from Individual; add two-tier consts)
    problem.hpp      ← unchanged
    random.hpp       ← unchanged
    print.hpp        ← unchanged interface
    codegen.hpp      ← unchanged interface
    speciation.hpp   ← unchanged
    innovation.hpp   ← DELETED
    hardening.hpp    ← DELETED
  src/
    types.cpp        ← (does not exist currently; not needed)
    dag.cpp          ← rewritten (simpler with SSA)
    execute.cpp      ← rewritten (val array instead of register file)
    jit.cpp          ← rewritten (SSA slot layout; keep single-program path)
    batch_jit.cpp    ← NEW
    fitness.cpp      ← simplified (MCMC path only; batch eval moves to population)
    mutate.cpp       ← rewritten (no hardness, no NEAT, new positional crossover)
    population.cpp   ← modified (decoupled gen/eval, two-tier, batch JIT)
    problem.cpp      ← unchanged
    random.cpp       ← updated (new SSA random_program; keep random_instr)
    print.cpp        ← updated (no dst field in output)
    codegen.cpp      ← updated (SSA slot variables in emitted C)
    speciation.cpp   ← unchanged (compute_fingerprint just calls execute)
    innovation.cpp   ← DELETED
    hardening.cpp    ← DELETED
  main.cpp           ← update includes ("foo.hpp" not "src/foo.hpp")
  Makefile           ← -I include; remove hardening/innovation from LIB_SRCS
  tests/
    jit_vs_vm.cpp    ← rewrite for new execute API and SSA programs
```

---

## 3. Module-by-Module Changes

### 3.1 `dag.cpp` — Simplified

SSA makes dead-code analysis unambiguous: a live instruction is one that transitively contributes to the output (last instruction's slot).

```cpp
int compute_dag(const Program& prog, bool live[Program::MAX_INSTRS]) {
    const int n    = prog.num_instrs;
    const int base = Program::N_MAX_INPUTS;
    memset(live, 0, n * sizeof(bool));
    if (n == 0) return 0;

    bool needed[Program::TOTAL_SLOTS] = {};
    needed[base + n - 1] = true;  // last instruction is always the output

    for (int i = n - 1; i >= 0; i--) {
        if (!needed[base + i]) continue;
        live[i] = true;
        const Instr& ins = prog.instrs[i];
        // All ops use src1 except LOADI/LOADF
        switch (ins.op) {
        case Op::LOADI: case Op::LOADF:
            break;  // no register sources
        default:
            needed[ins.src1] = true;
            if (ins.src2 != Program::IMM_SRC)
                needed[ins.src2] = true;
            break;
        }
    }
    int count = 0;
    for (int i = 0; i < n; i++) count += live[i];
    return count;
}
```

No "clear needed[dst] after marking" trick needed — SSA slot indices are unique per instruction, so there's no aliasing.

### 3.2 `execute.cpp` — Val Array

```cpp
void execute(const Program& prog, const float* inputs, int n_in,
             float* outputs, int n_out) {
    float vals[Program::TOTAL_SLOTS] = {};
    for (int j = 0; j < n_in && j < Program::N_MAX_INPUTS; j++)
        vals[j] = inputs[j];

    bool live[Program::MAX_INSTRS];
    compute_dag(prog, live);

    const int base = Program::N_MAX_INPUTS;
    for (int i = 0; i < prog.num_instrs; i++) {
        if (!live[i]) continue;
        const Instr& ins = prog.instrs[i];
        float a = vals[ins.src1];
        union { uint32_t u; float f; int32_t i; } imm_b;
        imm_b.u = ins.lit;
        float b = (ins.src2 == Program::IMM_SRC)
                  ? imm_b.f   // note: float ops use imm as float; int ops use as int
                  : vals[ins.src2];
        // ... same switch on ins.op as before, but writing to vals[base + i] ...
        vals[base + i] = result;
    }
    for (int k = 0; k < n_out; k++)
        outputs[k] = (prog.num_instrs > 0) ? vals[base + prog.num_instrs - 1] : 0.0f;
}
```

**Issue with IMM_SRC interpretation**: currently `ins.lit.i` for int ops and `ins.lit.f` for float ops. The union field names need to be consistent. `Instr::lit` stays as `uint32_t`; the execute loop bit-casts per op type. Same as today, just the field is now `uint32_t lit` instead of `union {int32_t i; float f;} lit`.

### 3.3 `jit.cpp` — SSA Slot Layout (single-program path kept)

The single-program JIT (`jit_compile`, returning `JitProgram` with `float fn(float)`) stays for MCMC use. Stack layout changes:

**Old**: `[rsp + reg*4]` for 16 registers (64 bytes of stack)
**New**: `[rsp + slot*4]` for 28 slots (112 bytes → use 128 bytes for alignment)

Prologue changes:
- `sub rsp, 128` (was 64)
- Zero 128 bytes with xmm1 (8 × `movups [rsp+N], xmm1` at 16-byte intervals, or just clear the 28 used slots)
- `movss [rsp+0], xmm0` to write input into slot 0 (same as before for slot 0)
- Slots 1-3 are zeroed by the zero-init above

Epilogue changes:
- `movss xmm0, [rsp + (N_MAX_INPUTS + last_live_instr) * 4]` — read from last live instruction's slot (not always slot 0)
- `add rsp, 128`; `ret`

**Pitfall**: the existing `mem_rsp(reg, d)` helper uses `uint8_t d` for disp8. All 28 slots × 4 = offsets 0..108, all fit in uint8_t range (< 128). No change needed to addressing mode.

Instruction emission in `emit_instr`:
- `d = (N_MAX_INPUTS + i) * 4` — destination slot (passed in by caller; `emit_instr` needs the slot index, not instruction-relative)
- `s1 = ins.src1 * 4`, `s2 = ins.src2 * 4` — source slots
- Remove `ins.dst % NUM_REGS` — no dst field
- Arithmetic is identical; just addressing changes

**Keep the single-program arena (64KB)** — unchanged. Add a separate larger arena for the batch JIT.

### 3.4 `batch_jit.hpp/cpp` — NEW

```cpp
// include/batch_jit.hpp
#pragma once
#include "program.hpp"

using BatchFn = void (*)(float x, float* outputs);

struct BatchJIT {
    void*   mem  = nullptr;   // mmap region (nullptr = uses static arena)
    size_t  size = 0;
    BatchFn fn   = nullptr;
    int     n_progs = 0;

    BatchJIT() = default;
    ~BatchJIT();
    BatchJIT(BatchJIT&&) noexcept;
    BatchJIT& operator=(BatchJIT&&) noexcept;
    BatchJIT(const BatchJIT&) = delete;
    BatchJIT& operator=(const BatchJIT&) = delete;
};

// Compile programs[0..n-1] into a single batch function:
//   fn(x, outputs)  writes outputs[i] = prog_i(x) for i in 0..n-1
// Must call before evaluating; recompile whenever programs change.
BatchJIT compile_batch(const Program* progs, int n);
```

**Batch JIT structure:**

```
Function signature (SysV AMD64):
  void batch_eval(float x, float* outputs)
  xmm0 = x,  rdi = outputs

Prologue:
  sub rsp, 128              ; 28 slots × 4 bytes = 112, rounded to 128
  movss [rsp+0], xmm0       ; slot 0 = x (NEVER overwritten by instructions)
  xorps xmm0, xmm0
  movss [rsp+4],  xmm0      ; slot 1 = 0
  movss [rsp+8],  xmm0      ; slot 2 = 0
  movss [rsp+12], xmm0      ; slot 3 = 0
  ; slots 4..27 are NOT pre-zeroed (safe: live instrs only read live instrs' slots)

For each program p = 0..n-1:
  ; Emit only LIVE instructions for program p:
  ; Each instr writes to [rsp + (N_MAX_INPUTS + i) * 4]
  ; Sources read from [rsp + src * 4]
  [... live instruction sequence ...]
  ; Write output (last live instruction's slot):
  movss xmm0, [rsp + (N_MAX_INPUTS + last_live_i) * 4]
  movss [rdi + p*4], xmm0   ; NOTE: disp may exceed disp8; use disp32

Epilogue:
  add rsp, 128
  ret
```

**Addressing for `[rdi + p*4]`**: with up to 128 programs, max disp = 127×4 = 508 bytes. Exceeds disp8 (max 127). Need ModRM with disp32:
```
movss [rdi + imm32], xmm0:
  F3 0F 11 87 <imm32 as 4 bytes LE>
  ModRM = 10 000 111 = 0x87 (Mod=10=disp32, Reg=0=xmm0, RM=7=rdi)
```
Add a `store_xmm0_rdi_disp32(int32_t disp)` helper to the emitter struct `E`.

**Arena sizing**: separate 512KB static arena for batch JIT. Worst case: 128 programs × 24 live instrs × ~24 bytes/instr + 50-byte prologue/epilogue = ~73KB. 512KB gives plenty of headroom.

**Why slots 4+ don't need zeroing between programs**: In SSA, a live instruction at position `i` in program P only has src values in `[0, N_MAX_INPUTS + i)`. If src refers to slot `N_MAX_INPUTS + k` (k < i), then instruction k is a dependency of instruction i, hence also live in program P (by DAG transitivity). So every live instruction reads only slots written by earlier live instructions in the same program block. Dead instructions are not emitted at all. No stale data from a previous program can be observed.

### 3.5 `fitness.cpp` — MCMC-Only Path

The batch evaluation loop moves out of fitness.cpp and into population.cpp. What remains in fitness.cpp:

```cpp
// Single-program eval for MCMC use.
// Uses single-program JIT (or interpreter as fallback).
// n_cases: how many test cases to use (pass N_SMALL or N_CASES).
double fitness_single(const Program& prog,
                      const ProblemDef& problem,
                      const std::vector<float>& test_inputs,
                      int n_cases,
                      bool penalize_length = true);

// Same but also fills case_err[n_cases].
double fitness_and_cases_single(const Program& prog,
                                float* case_err,
                                const ProblemDef& problem,
                                const std::vector<float>& test_inputs,
                                int n_cases,
                                bool penalize_length = true);
```

Remove the `fitness_duration` / `fitness_calls` inline globals from `fitness.hpp` — they're confusing and will be tracked differently once the batch JIT is in place. Add `fitness_calls` as a local metric inside population.cpp if still wanted.

### 3.6 `mutate.cpp` — Simplified

**Remove:**
- `Hardness` parameter from `mutate()` (was passed but `.weight()` was commented out)
- `crossover_positional()` — replaced (see below)
- `crossover()` NEAT — replaced
- `legal_srcs_at()` — not needed in SSA
- `pick_instr()` hardness parameter — simplify to just liveness-weighted

**Simplify `pick_instr`**: uniform over instructions, but dead instructions get higher weight so evolution can clean up dead code.

**Mutation kinds**: keep the same high-level types but adapt for SSA:

| Kind | Change |
|---|---|
| Field mutate | Change op, src1, src2, or lit. Sources picked from `[0, N_MAX_INPUTS + pos)` — trivially correct |
| Strip dead | Pick instruction weighted toward dead; remove it; fixup refs (see pitfall below) |
| MCMC constants | **Changed** — see §3.7 |
| Dep-split insert | Insert new instruction; shift later refs (see pitfall below) |
| Add instruction | Insert at random position; shift later refs |
| Remove instructions | Remove 1-3 contiguous; shift later refs |
| Replace slice | Replace slice with fresh random instructions |

**Pitfall — SSA insert/remove index fixup:**

When an instruction is inserted at position `k`:
- Instructions at positions `> k` now have their implicit output slot incremented
- Any later instruction's src that references a slot `>= N_MAX_INPUTS + k` must be incremented by 1

When instruction at position `k` is removed:
- Find all later instructions with src `== N_MAX_INPUTS + k` — remap to some valid alternative (src1 of the removed instruction, or 0)
- Decrement all later src references `> N_MAX_INPUTS + k` by 1

This fixup is O(MAX_INSTRS) — cheap.

**New positional crossover** (replaces both old crossover functions):

```cpp
Program crossover(const Program& a, const Program& b, std::mt19937& rng);
```

Take prefix of `a` (instrs `0..cut_a-1`) + suffix of `b` (instrs `cut_b..nb-1`). Fix up `b`'s suffix sources:
- src < N_MAX_INPUTS: keep (input slot, always valid)
- src >= N_MAX_INPUTS and `src - N_MAX_INPUTS < cut_b`: references B's prefix (not in child) → remap to `N_MAX_INPUTS + cut_a - 1` (last slot of A's prefix) or slot 0 if `cut_a == 0`
- src >= N_MAX_INPUTS + cut_b: in B's suffix; shift: `new_src = src - cut_b + cut_a`

Generate 3-5 random (cut_a, cut_b) pairs, pick the one that produces a non-trivially-small program. No hardness scoring.

### 3.7 Constant MCMC — New Design

**Rule**: `lit` fields are never randomly perturbed in normal mutation. The only way constants change is:
1. Mutation kind "MCMC constants" triggers a full solve of all LOADI/LOADF lits in the program
2. When any instruction with LOADI/LOADF is added (insert or replace), immediately trigger MCMC on ALL constants in that program before returning from `mutate()`

**MCMC procedure** (`solve_constants` in mutate.cpp):
```
solve_constants(Program& prog, rng, problem, test_inputs, n_cases=N_SMALL):
  find all constant instructions (LOADI, LOADF)
  if none: return
  
  // Coordinate descent: iterate over constants in random order
  for round in 0..2:
    shuffle constant indices
    for each constant instruction k:
      run short MCMC (200 steps, simulated annealing) on prog.instrs[k].lit
      evaluate with fitness_single(..., N_SMALL, penalize_length=false)
      keep best value found
```

The outer `mutate()` function calls `solve_constants()` after any mutation that adds a LOADI/LOADF. The "MCMC constants" mutation kind calls `solve_constants()` unconditionally (replacing a full mutation).

**Pitfall**: `solve_constants` calls `fitness_single` which uses the single-program JIT. This means MCMC is ~200× N_SMALL evaluations per program per trigger. At N_SMALL=50: 10,000 interpreter calls. At ~200ns each = 2ms per MCMC trigger. Acceptable since it's infrequent and produces well-tuned constants.

**Do NOT call `solve_constants` inside the batch JIT loop** — it uses the single-program path and happens during *generation* (before batch eval), so the flow is:
1. Generate child (mutate returns a program with constants already MCMC'd)
2. Add to candidate list
3. Batch-compile candidates
4. Batch-evaluate candidates

### 3.8 `population.hpp/cpp` — Decoupled Gen/Eval + Two-Tier

**Remove from `Individual`**: `Hardness hardness`

**New constants**:
```cpp
static constexpr int N_SMALL     = 50;   // tier-1 test cases (all candidates)
static constexpr int N_CASES     = 500;  // tier-2 test cases (survivors only)
static constexpr int TIER2_FRAC  = 2;    // top 1/TIER2_FRAC per island go to tier 2
```

**New `step()` flow**:

```
step(rng):
  1. Global stagnation + hot-burst update (unchanged logic)
  2. Novelty/frontier hysteresis update (unchanged)
  3. For each island:
     a. Generate candidates:
        - Elitism: champion stays
        - Extra migration if island stagnant (unchanged)
        - Fill rest: mutate or crossover with finalize_child (novelty check)
        → collect as list of Programs (not yet evaluated)
  4. Compile batch JIT for all candidate programs across all islands
  5. Tier-1 eval: run batch_eval on N_SMALL test cases → case_err_small, fit_small per candidate
  6. For each island: select top ISLAND_SIZE / TIER2_FRAC by fit_small → tier-2 candidates
  7. Compile batch JIT for tier-2 candidates (subset of the full batch)
     OR: evaluate tier-2 using full test set in the same batch (just run more cases)
  8. Tier-2 eval: run batch_eval on N_CASES test cases → final case_err, fit per tier-2 individual
  9. Champions and selected survivors get full fitness; rest get tier-1 fitness
  10. Selection / replacement per island using final fitness
  11. sort_island, LRU cache update, curriculum advancement (unchanged)
```

**Simplification**: Since the batch JIT processes all islands at once, candidate arrays are flat:
```cpp
Program  candidates[N_ISLANDS * ISLAND_SIZE];   // all new programs
double   fit_small[N_ISLANDS * ISLAND_SIZE];     // tier-1 fitness
float    case_err_small[N_ISLANDS * ISLAND_SIZE][N_SMALL];
```

**LRU cache** stays but becomes an LRU of `(program_hash → {fit, case_err[N_CASES]})`. Tier-2 results are cached; tier-1 results are NOT cached (too cheap to be worth it). Lookup before tier-2 eval.

**Migrate** simplified — no positional crossover with hardness:
```cpp
void migrate(rng):
  for each island ii:
    pick random other island jj
    immigrant = islands[ii].indivs[0]   // best of source
    cross = crossover(immigrant.prog, islands[jj].indivs[0].prog, rng)
    // eval cross with full test set (single-program JIT, not batch — only 1 program)
    eval cross → replace worst slot in jj
    clone immigrant → replace second-worst slot in jj
    sort_island(jj)
```

**Pitfall — LRU cache and two-tier eval**: The cache stores full (N_CASES) fitness. On a cache hit during tier-2, use the cached full fitness. During tier-1, never check cache (it's faster to just run the small eval than hash + lookup).

**Pitfall — `current_test_inputs` size**: Currently it's `N_CASES * n_inputs` floats for the full eval. With two-tier, keep two: `small_test_inputs` (N_SMALL × n_inputs) and `full_test_inputs` (N_CASES × n_inputs). Both regenerated each generation (random samples — same as today).

### 3.9 `print.cpp` and `codegen.cpp`

**print.cpp**: `print_program` currently shows `op dst src1 src2`. Remove `dst` column (implicit). Show slot indices as `s0`, `s1`, etc.:
```
  FMUL  s6  ← s4 * s5    (instr 2: output=slot 6, src1=slot 4, src2=slot 5)
  LOADI s4  0x3f800000    (float 1.0f)
```

**codegen.cpp** (`emit_solution`): Generate named slot variables for clarity:
```c
float jit_eval(float x) {
    float s0 = x;
    // s1=s2=s3 unused inputs, implicitly 0.0
    float s4 = 1.0f;             // LOADF
    float s5 = s0 * s0;          // FMUL s5 = s0 * s0
    float s6 = s5 + s4;          // FADD s6 = s5 + s4
    return s6;
}
```

No `dst` to extract; the output name is `s[N_MAX_INPUTS + last_live_instr]`. Much cleaner than the current register-aliasing approach.

### 3.10 `speciation.cpp` — Minimal Update

`compute_fingerprint` calls `execute(p, xs, problem.n_inputs, out_val, 1)`. After execute.cpp is rewritten, this call site is compatible if the signature remains `execute(prog, inputs, n_in, outputs, n_out)`. No logic changes needed.

### 3.11 `random.cpp`

**`random_instr(rng)`**: remove `innov` field assignment. src1 and src2 initialised to 0 (safe: callers fix them up contextually after the call). The `pos`-dependent source range must be fixed by the caller (mutate already does this).

**`random_program(rng)`**: rewrite to generate a valid SSA program:
- Pick random length 1..8
- For instruction at position `i`, pick `src1` from `[0, N_MAX_INPUTS + i)` and `src2` similarly or IMM_SRC
- No `dst` field to set

The test `tests/jit_vs_vm.cpp` uses `random_program`. It needs a full rewrite for the new execute API anyway (see §4).

### 3.12 `innovation.hpp/cpp` — Deleted

Remove files. Remove `#include "innovation.hpp"` from `random.cpp` and `population.cpp`. Remove `si.innov = next_innovation()` from `population.cpp::init()`.

### 3.13 `hardening.hpp/cpp` — Deleted

Remove files. Remove all `#include "hardening.hpp"`. Remove `Hardness hardness` from `Individual`. Remove commented-out `isl.indivs[0].hardness.recompute(...)` block.

### 3.14 Seed Program in `population.cpp::init()`

Currently seeds with LOADI 0 (returns integer 0 — a terrible starting point). Better seed with a MOV of slot 0 (returns the input, giving f(x) = x as starting point):
```cpp
Program seed = {};
Instr& si = seed.instrs[0];
si.op   = Op::MOV;
si.src1 = 0;       // slot 0 = input x
si.src2 = 0;       // ignored for MOV
si.lit  = 0;
seed.num_instrs = 1;
```

### 3.15 `Makefile` Updates

```makefile
BASE_FLAGS := -std=c++17 -Wall -Wextra -I include   # was -I src

LIB_SRCS := src/problem.cpp src/execute.cpp src/jit.cpp src/fitness.cpp \
            src/dag.cpp src/random.cpp src/mutate.cpp \
            src/population.cpp src/print.cpp src/speciation.cpp \
            src/codegen.cpp src/batch_jit.cpp
            # removed: src/hardening.cpp src/innovation.cpp
```

### 3.16 `main.cpp` Updates

Change all includes:
```cpp
// Before:
#include "src/population.hpp"
// After:
#include "population.hpp"
```

Remove references to `fitness_duration` / `fitness_calls` globals (moved or removed).

---

## 4. Known Issues and Pitfalls

### 4.1 IMM_SRC and Integer vs Float Literals

With `Instr::lit` as `uint32_t`, the interpreter must bit-cast to the right type based on op:
- Integer ops (IADD, ISUB, IMUL, BAND, BOR, BXOR, LSHL, LSHR, ASHL, ASHR, ILT, IEQ, ULT, UEQ): `(int32_t)lit`
- LOADI: `(int32_t)lit`
- LOADF: `bit_cast<float>(lit)`

Float ops (FADD, FSUB, FMUL, FLT, FEQ) do not support IMM_SRC — src2 must be a slot. If a mutation sets src2=IMM_SRC on a float op, treat `lit` as float bits (`bit_cast<float>(lit)`). The old code avoided this entirely; the new code should either prohibit it or handle it consistently. Recommendation: prohibit IMM_SRC on float binary ops in `mutate()` (same as current `op_supports_imm_src2` logic, just renamed).

### 4.2 `num_instrs == 0` Edge Case

Program with 0 instructions has no output slot. The execute function returns 0.0f for this case. Mutations must ensure at least 1 instruction is always present (currently enforced; keep this invariant).

### 4.3 Last Instruction May Be an Integer Op

The program output reads from the last instruction's slot as a float. If the last instruction has an integer op (e.g., IADD), the bits are reinterpreted as float — which may produce garbage. This already happens today (reg[0] is returned as float regardless of type tag). The fitness function's `!isfinite(got)` check catches the worst cases.

To improve: weight the mutation's op selection to prefer float ops for the last instruction, or add a penalty for returning from an integer op. This is a future optimization, not a blocker.

### 4.4 Batch JIT and Novelty Fingerprinting Interaction

The novelty system calls `compute_fingerprint` → `execute` (interpreter, not JIT) for N_BEH=5 samples per candidate. This happens during `finalize_child` inside the generation phase (before batch eval). Since it uses the interpreter and only 5 samples, this is cheap (~1μs per program). No change needed to this path.

### 4.5 Batch JIT Re-Compilation Trigger

The batch JIT must be recompiled whenever the candidate set changes. In the decoupled gen/eval flow, this happens exactly once per generation (after all candidates are generated). No partial recompile needed.

**However**: migrate() evaluates a single program immediately (for the crossover result). This uses the single-program JIT path, not the batch JIT. Migrate happens after selection/replacement, so it doesn't interfere with the batch eval loop.

### 4.6 `compute_fingerprint` and Two-Tier Eval

`compute_fingerprint` is called during generation (inside `finalize_child`) before evaluation. It uses `execute()` (interpreter), not the JIT, so it's independent of the two-tier eval. No change needed.

### 4.7 LRU Cache Key After Representation Change

The program hash in `population.cpp::program_hash()` iterates over `num_instrs` and hashes `op, dst, src1, src2, lit, innov`. After removing `dst` and `innov`, the hash mixes `op, src1, src2, lit` per instruction. Update the hash function accordingly.

### 4.8 Stack Slot Displacement in JIT — Disp8 vs Disp32

All SSA slot offsets (`slot * 4`) for slots 0..27 give offsets 0..108, all fitting in `uint8_t` (< 128). The existing `mem_rsp` helper with `uint8_t d` remains valid. **No change needed for slot→stack addressing.**

Output stores into `[rdi + p*4]` require disp32 since `p` can be up to 127 (giving disp 508 > 127). Add a new helper `store_xmm0_outputs(int p)` to the emitter:
```cpp
void store_xmm0_outputs(int p) {
    // movss [rdi + p*4], xmm0
    // F3 0F 11 87 <disp32>
    b(0xF3); b(0x0F); b(0x11); b(0x87);
    int32_t disp = p * 4;
    w32(uint32_t(disp));
}
```

### 4.9 Test File `tests/jit_vs_vm.cpp`

This file is already broken (calls `execute(prog, inputs[i])` — old single-arg API). Must be rewritten to:
- Generate valid SSA programs using new `random_program()`
- Call `execute(prog, inputs, 1, &out, 1)` for interpreter
- Compile with `jit_compile(prog)` and call `jit.fn(x)` for JIT
- Compare outputs

### 4.10 Two-Tier Eval and Stagnation Metrics

The `global_best_fit` and `island.best_fit` comparisons must use tier-2 fitness (full N_CASES evaluation) to be meaningful. The champion that drives stagnation tracking is always fully evaluated. Tier-1 fitness is used only for within-generation selection of who gets tier-2 evaluated.

---

## 5. Implementation Order

Dependencies flow top-to-bottom. Each step is independently testable before proceeding.

```
Step 1: types.hpp, program.hpp
        ← New Instr (no dst, no innov), Program with N_MAX_INPUTS/TOTAL_SLOTS
        ← Remove RegType, Reg
        ← Delete innovation.hpp, hardening.hpp

Step 2: dag.cpp (rewrite for SSA)
        ← Test: manually craft SSA programs, verify live sets

Step 3: execute.cpp (rewrite, val array)
        ← Test: interpreter gives correct results on simple SSA programs

Step 4: random.cpp (new random_program for SSA; remove innov from random_instr)
        ← Needed by test and mutate

Step 5: jit.cpp (rewrite single-program JIT for SSA slot layout)
        ← Test: jit vs interpreter agree on 2000 random programs (rewrite jit_vs_vm.cpp)

Step 6: fitness.cpp (trim to fitness_single / fitness_and_cases_single)
        ← Remove batch eval path; remove inline globals

Step 7: mutate.cpp (rewrite: no hardness, no NEAT, SSA-correct insert/remove,
                    new positional crossover, MCMC constants)
        ← Test: mutated programs always have valid src indices; MCMC improves constants

Step 8: batch_jit.cpp (new: compile_batch, BatchJIT)
        ← Test: batch_eval agrees with N individual jit_compile + jit.fn calls
                 for all programs and a grid of inputs

Step 9: speciation.cpp (update execute call if signature changed — likely trivial)

Step 10: population.hpp/cpp (remove Hardness; decouple gen/eval; two-tier eval;
                              use batch JIT; simplify migrate)
         ← Delete hardening.cpp, innovation.cpp

Step 11: print.cpp, codegen.cpp (update for no-dst, SSA slot labels)

Step 12: main.cpp (update includes, remove fitness_duration/calls usage)

Step 13: Makefile (-I include, remove deleted files from LIB_SRCS)

Step 14: Move all .hpp to include/; verify full build; run jit_vs_vm test
```

---

## 6. Quick Reference: What Each File Becomes

| File | Status | Key Change |
|---|---|---|
| `include/types.hpp` | Modified | Remove RegType, Reg; new Instr (no dst/innov) |
| `include/program.hpp` | Modified | N_MAX_INPUTS, TOTAL_SLOTS; no NUM_REGS |
| `include/dag.hpp` | Unchanged interface | — |
| `include/execute.hpp` | Unchanged interface | — |
| `include/jit.hpp` | Unchanged | — |
| `include/batch_jit.hpp` | **NEW** | BatchJIT, compile_batch |
| `include/fitness.hpp` | Modified | fitness_single replaces fitness_and_cases |
| `include/mutate.hpp` | Modified | No Hardness param; crossover(a,b,rng) |
| `include/population.hpp` | Modified | No Hardness in Individual; two-tier consts |
| `include/hardening.hpp` | **DELETED** | — |
| `include/innovation.hpp` | **DELETED** | — |
| `src/dag.cpp` | Rewritten | SSA backward pass |
| `src/execute.cpp` | Rewritten | Val array, no register file |
| `src/jit.cpp` | Rewritten | SSA slot layout; keep single-prog path |
| `src/batch_jit.cpp` | **NEW** | Batch compilation and evaluation |
| `src/fitness.cpp` | Simplified | MCMC-only path; single-prog eval |
| `src/mutate.cpp` | Rewritten | No hardness; SSA fixup; positional crossover; MCMC trigger |
| `src/population.cpp` | Modified | Decoupled gen/eval; batch JIT; two-tier |
| `src/speciation.cpp` | Trivial update | execute call site |
| `src/random.cpp` | Updated | No innov; new random_program |
| `src/print.cpp` | Updated | No dst; slot labels |
| `src/codegen.cpp` | Updated | SSA slot variable names in emitted C |
| `src/hardening.cpp` | **DELETED** | — |
| `src/innovation.cpp` | **DELETED** | — |
| `tests/jit_vs_vm.cpp` | Rewritten | New execute API; SSA programs |
| `main.cpp` | Updated | Include paths; remove fitness globals |
| `Makefile` | Updated | -I include; removed files |
