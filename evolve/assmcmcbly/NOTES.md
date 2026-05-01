# assmcmcbly — project notes

## What this is

A genetic algorithm that evolves small assembly-like programs to rediscover the
Quake III fast inverse square root trick (`1/sqrt(x)`) from scratch — without
being told the algorithm, without magic constants, purely from selection pressure.

The target function is `f(x) = 1/sqrt(x)` for positive floats.  
The evolved programs run on a tiny virtual register machine.

---

## The virtual machine

**Registers:** 16 general-purpose registers (`r0`–`r15`), each tagged as either
`INT` (int32) or `FLOAT` (float32). They share the same 32 bits — the tag just
tracks the current interpretation.

**Input/output:**  
- `r0` is initialised with the float input before execution  
- `r0` after execution is the return value (read as float regardless of tag)

**Instruction set:**

| Category | Ops |
|---|---|
| Signed int arithmetic | `IADD ISUB IMUL` |
| Float arithmetic | `FADD FSUB FMUL` |
| Bitwise | `BAND BOR BXOR BNOT` |
| Logical shift (zero-fill) | `LSHL LSHR` |
| Arithmetic shift (sign-extend) | `ASHL ASHR` |
| Signed int compare → 0/1 | `ILT IEQ` |
| Unsigned int compare → 0/1 | `ULT UEQ` |
| Float compare → 0/1 | `FLT FEQ` |
| Unary | `LNOT INEG FNEG` |
| Reinterpret bits (tag flip only) | `ITF FTI` |
| Load literal | `LOADI LOADF` |
| Copy | `MOV` |

`ITF`/`FTI` are the key ops — they reinterpret the raw 32 bits as the other
type without changing them. This is exactly what the Quake trick does:
`i = *(long*)&y` then `y = *(float*)&i`.

---

## Program representation

Programs are a flat array of instructions (`instrs[0..num_instrs-1]`) executed
in order. No branches, no loops — straight-line SSA-style code.

Chromosomes are a logical subdivision of the instruction array:

```
instrs: [ chrom 0 | chrom 1 | ... | chrom N ]
         ^--- chrom_lens[0] ---^
```

- Up to 64 chromosomes, each up to 8 instructions (512 instructions max)
- Execution ignores chromosome boundaries — it's a flat walk
- Chromosomes exist only for genetic operators (crossover slices at chromosome
  boundaries, not mid-instruction)
- All storage is fixed-size flat arrays — no heap allocation in the hot path

Key struct fields:
```cpp
Instr    instrs[512]   // flat instruction array
uint8_t  chrom_lens[64] // length of each chromosome
uint8_t  num_chroms     // active chromosome count
uint16_t num_instrs     // total instruction count (= sum of chrom_lens)
```

---

## Fitness function (`fitness.cpp`)

Mean squared relative error over 100 log-spaced test inputs in `[0.01, 100]`:

```
msre = mean( ((got - target) / target)^2 )
```

Plus a dead-code penalty:
```
fitness = msre * (1 + 0.1 * (dead_instrs / total_instrs))
```

where dead instructions are those not in the DAG leading to `r0`'s final value
(computed by backward liveness analysis in `dag.cpp`).

Lower fitness = better. Perfect = 0.

---

## Genetic algorithm (`population.cpp`)

**Population size:** 128 individuals  
**Elitism:** top 8 survive unchanged each generation  
**Selection:** tournament of size 5  
**Each generation:** 70% mutation, 30% crossover

### Mutation types (`mutate.cpp`)

| Kind | What it does |
|---|---|
| 0 | Mutate one field (op/dst/src1/src2/lit) of one instruction |
| 1 | Swap two chromosomes (reorder) |
| 2 | Replace an entire chromosome with a random one |
| 3 | Insert a new random chromosome |
| 4 | Remove a chromosome |
| 5 | **Constant MCMC** — see below |

Instruction selection within a chromosome is **weighted**:
- Dead instructions (not in DAG) are 10× more likely to be mutated
- Hard instructions (high hardness score) are proportionally less likely
- Combined weight: `(dead ? 10 : 1) * hardness.weight(i)`

### Constant MCMC mutation (kind 5)

When selected:
1. Collect all `LOADI`/`LOADF` instructions in the program
2. Pick one randomly
3. Run 200 steps of simulated annealing (T: 2.0 → 0.001) touching only that
   literal's 32 bits. Perturbations: bit flip, add/subtract power-of-two,
   shift by 1, randomise upper/lower byte
4. If best found improves fitness by ≥ 1% → keep it
5. Otherwise → restore original and fall back to kind 0 (normal field mutation)

This is how the search finds magic constants like `0x5f3759df` — the neighbourhood
around them is exactly the kind of bit-level perturbations this explores.

### Crossover (`mutate.cpp`)

Single-point crossover at chromosome boundaries:
- Take first `cut_a` chromosomes from parent A
- Append last `(num_chroms - cut_b)` chromosomes from parent B

### Hardening (`hardening.cpp`)

Every 100 generations, for each of the 8 elite individuals:
- For each instruction: temporarily replace it with `MOV r15 r15` (a no-op)
  and re-evaluate fitness
- `hardness[i] = max(0, ablated_fitness - base_fitness)`
- High hardness = instruction is load-bearing = less likely to be mutated

Hardness is reset to 0 for newly created/replaced instructions. This means
critical instructions that the search has discovered accumulate protection over
time.

---

## File structure

```
assmcmcbly/
  types.hpp       — RegType, RegVal, Reg, Op, Instr
  program.hpp     — Program struct (flat arrays + chrom_start())
  execute.cpp/hpp — VM interpreter loop
  fitness.cpp/hpp — MSRE + dead-code penalty
  dag.cpp/hpp     — Backward liveness analysis (which instrs reach r0)
  random.cpp/hpp  — random_instr, random_program, append_random_chromosome
  mutate.cpp/hpp  — mutate() (6 kinds), crossover()
  hardening.cpp/hpp — Hardness struct + ablation recompute
  population.cpp/hpp — Individual, Population (init/step/select)
  print.cpp/hpp   — op_str(), print_program()
  main.cpp        — init population, loop forever printing progress
  Makefile        — incremental build, target = ./assmcmcbly
  NOTES.md        — this file
```

Build: `make -j` from this directory, or `just evolve` from the repo root.

---

## What's working / current state

- Full GA loop running indefinitely, printing every 100 gens
- Fitness converges — was getting to ~0.03 MSRE in early runs with the old
  single-file MCMC version; GA version should do better with more search diversity
- The constant MCMC mutation is the most likely path to rediscovering the magic
  constant, since it does targeted bit-level search around whatever value a
  LOADI instruction currently holds

---

## Ideas / things to try next

- **Parallelism:** run multiple independent populations (islands) in threads,
  periodically migrate the best individual between islands
- **Seeding:** inject a known-good partial program (just the `FTI → ISUB → ITF`
  skeleton) as one of the initial population members to guide early search
- **Adaptive mutation rates:** track improvement rate per mutation kind and
  weight toward kinds that have been productive recently
- **Larger test set / curriculum:** start with fewer/easier test points and
  expand as fitness improves, to avoid overfitting to the 100-point sample
- **Two-point crossover:** currently single-point; two-point would allow
  extracting "middle sections" of chromosomes from a parent
- **Print live-only:** when printing the best program, skip dead instructions
  or mark them visually so the output is easier to read
- **Save/load best:** checkpoint the best individual to disk so a run can be
  resumed or inspected without restarting
