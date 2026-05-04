# E-Graphs, Approximate Equality Saturation, and Program Search

Notes on whether / how e-graph techniques could help the `assmcmcbly` project
(evolving assembly-like programs to rediscover the Quake III fast inverse
square root trick).

---

## 1. What Equality Saturation Actually Is

An **e-graph** is a data structure that compactly represents an entire
*equivalence class* of expressions simultaneously. Equality saturation works
by:

1. Insert a starting expression into the e-graph
2. Repeatedly fire rewrite rules (`a + b → b + a`, `x * 2 → x << 1`, etc.)
   to merge new expressions into the same equivalence class
3. Once saturated (no new merges possible), extract the cheapest member of the
   equivalence class under some cost function

The key property: merging is monotonic — once two nodes are deemed equal they
stay equal. This makes the search efficient but also means **rules must be
exactly correct**. Approximate equalities break this.

**What it is not:** a synthesis tool in the generative sense. An e-graph starts
from a known expression and explores its equivalence class. It cannot generate
a program from scratch that satisfies a specification. This is the core
mismatch with our use case.

---

## 2. State of the Art: What Exists

### Optimization tools (not synthesis)

**egg** (POPL 2021) — the foundational e-graph library. Fast, extensible,
Rust. Used as the backend for Herbie, Cranelift, etc.
- Paper: https://arxiv.org/abs/2004.03082
- Repo: https://github.com/egraphs-good/egg

**Herbie** — rewrites floating-point expressions to improve numerical accuracy.
Uses e-graphs internally with approximate rewrite rules (guaranteed-bounded
error, not arbitrary approximation). Works on smooth scalar expressions. Cannot
handle bit-reinterpretation (FTI/ITF) because there is no algebraic rewrite
path from `1/sqrt(x)` to the Quake trick.
- Site: https://herbie.uwplse.org/

**egglog** — e-graphs + Datalog. Allows relational reasoning alongside
rewriting. More expressive than egg alone.
- Repo: https://github.com/egraphs-good/egglog

### Synthesis tools (closer to what we want)

**MegaLibm** (POPL 2024) — synthesizes math library implementations (`sin`,
`exp`, `log`, etc.) from scratch within a DSL. Uses e-graphs to: generate
candidate identities, prove equalities, simplify, solve inductive equations.
Can produce from-scratch `sin` implementations.
**Hard limit:** cannot discover unexpected identities on its own — the `asin`
range-reduction trick in the authors' own paper required manual insertion via
a `NamedHole`. FTI/ITF reinterpretation is completely outside its model.
- Paper: https://arxiv.org/abs/2311.01515

**Ruler** (OOPSLA 2021) — synthesizes *rewrite rules* rather than programs.
Given a grammar and interpreter, enumerates terms up to some size, clusters by
observed equality on sample inputs, extracts a minimal rule set. Replaced
Herbie's hand-written rational-number rules and found a bug in them. Could
mine rules for our ISA if given a grammar — but gives you rules, not the final
program.
- Paper: https://www.mwillsey.com/papers/ruler

**Guided Equality Saturation** (POPL 2024) — human supplies intermediate
"sketch" goals; system fills the gap automatically. Reduces 60 GB / 1 hour
problems to <1 GB / seconds. Useful if you already know the shape of the
solution (e.g. "I know the program passes through an int-domain intermediate").
Defeats the purpose of open-ended discovery.
- Paper: https://steuwer.info/files/publications/2024/POPL-Guided-Equality-Saturation.pdf

**eggp** (GECCO 2024 / arxiv 2501.17848) — **most directly relevant.**
Combines e-graphs with genetic programming for symbolic regression. The
e-graph stores every previously visited expression. Before accepting a
proposed mutant or crossover child, it checks equivalence; if the result is
semantically equivalent to anything already seen, it is rejected. Prevents
re-evaluation of semantically identical programs. Competitive with PySR and
Operon on small expression benchmarks.
- Paper: https://arxiv.org/abs/2501.17848
- Repo: https://github.com/folivetti/eggp

### Stochastic assembly synthesis (closest in spirit to assmcmcbly)

**STOKE** (ASPLOS 2013) — MCMC random search over x86-64 assembly with a
continuous cost function: weighted sum of correctness error (bitwise distance
on sampled inputs) and performance (instruction count). No e-graphs; stochastic
search + good cost function is the key.

**STOKE floating-point** (PLDI 2014) — extends STOKE to floating-point
programs with tunable precision. Synthesizes and optimizes real x86 code
including FP kernels. Demonstrates that bit-level synthesis of FP programs is
tractable with MCMC when the cost function is continuous.
- ASPLOS 2013: https://theory.stanford.edu/~aiken/publications/papers/asplos13.pdf
- PLDI 2014:   https://theory.stanford.edu/~aiken/publications/papers/pldi14a.pdf

---

## 3. Why Classical E-Graphs Don't Directly Apply Here

The Quake trick requires three qualitative domain crossings:

1. `FTI(x)` — reinterpret float bits as integer
2. Integer arithmetic (subtract from magic constant, arithmetic right-shift)
3. `ITF(result)` — reinterpret integer bits as float

Each crossing looks like pure noise from an algebraic perspective. There is no
chain of algebraic rewrite rules that leads from `1.0 / sqrt(x)` to
`ITF(MAGIC - (FTI(x) >> 1))`. You would have to *already know the trick* and
encode it as an explicit rule — which defeats the purpose of discovery.

Additionally, the key algebraic identity underlying the trick is only
*approximate*: `FTI(x) ≈ 2²³ · log₂(x) + 2²³ · 127`. This holds to ~0.04%
for normalized floats (the mantissa linear approximation introduces that error).
E-graphs require exact equality for merging to be sound.

---

## 4. Approximate Rewrite Rules: What They Would Look Like

For this codebase, you'd need three semantic domains:

```
F(v)  — value in float domain
I(v)  — raw 32-bit representation (integer domain)  
L(v)  — float-as-approx-log: L(x) ≈ 2²³ · log₂(x) + 2²³ · 127
```

### Core bridging rules (ε ≈ 0.04% each)

```
I(F(x)) ≈[ε] L(x)                        // FTI: float bits ≈ scaled log
F(I(n))  = n                              // ITF: exact by definition
```

### Composition rules in the log domain

```
F(L(x) >> 1)      ≈[ε'] sqrt(x) · K₁    // right-shift halves the exponent
F(C - L(x))       ≈[ε'] C_f / x          // subtract = divide in log domain
F((C - L(x)) >> 1) ≈[ε''] C_k / sqrt(x) // compose the above two
```

where `K₁ = 2^(-127/2)`, `C_f = ITF(C)`, `C_k` depends on `C` and
`K₁` combined.

### Bit-addition = float scaling

```
F(I(F(v)) + k)  ≈[ε] v · 2^(k / 2²³)
```

Adding `k` to the raw float bits shifts the exponent field, scaling the float
value by `2^(k/2²³)`. Used in Newton correction steps.

### Concrete example: the evolved program

The program in this run computes:

```
K = C - (C << 10) mod 2³²   // ≈ 2552561888
r7 = K - FTI(x)             // ≈ L(K_f / x) where K_f = ITF(K)
r10 = r7 >> 1                // ≈ L(sqrt(K_f / x))
r0 = r9.f * ITF(r10)         // ≈ r9.f · sqrt(K_f / x) ≈ M / sqrt(x)
```

The `ISUB r8 r0 r4` + `FADD r0 r8 r0` correction adds ~60% of `r0` to itself,
producing ≈ `1.607 · M / sqrt(x) + M / sqrt(x) ≈ 0.923 / sqrt(x)`. The
remaining ~1.45% error comes from accumulated mantissa approximation through
three domain crossings.

---

## 5. The Fundamental Problem: Approximate E-Graphs

The rules above are not exact — they have error ε that compounds. For an
approximate e-graph to work:

1. **Attach an error interval** `[lo, hi]` to each e-node (measured over the
   test input range `[lo_x, hi_x]`)
2. **Only fire a rule** if the composed error stays under a threshold
3. **Use interval arithmetic** to propagate bounds:
   `compose(ε₁, ε₂) ≤ (1 + ε₁)(1 + ε₂) - 1 ≈ ε₁ + ε₂` for small ε

The core difficulty: **ε-equivalence is not transitive**. If A ≈ B and B ≈ C,
it does not follow that A ≈ C with the same ε. Classical e-graph merging
requires transitivity to be sound. An approximate e-graph must either:

- Use conservative error tracking (propagate worst-case bounds) — safe but
  misses many useful identities
- Use clustering instead of merging (group nodes by behavior bucket) — loses
  the compact representation advantage

Herbie sidesteps this by only encoding rewrite rules that are *guaranteed*
within a user-specified ULP error bound — not nodes that happen to be
numerically close on test inputs. This is principled but requires knowing the
rules in advance.

---

## 6. What Would Actually Help This Project

### Immediately applicable: eggp-style equivalence filter

Maintain an e-graph over the program population. Before evaluating a proposed
mutant or crossover child, check if it is semantically equivalent (via algebraic
rewrites over the ISA's exact rules) to any existing individual. If so, skip
evaluation. This avoids wasting fitness calls on semantically identical programs.

**Applicable rules** (all exact, no approximation needed):

```
IADD(a, b)  = IADD(b, a)                    // commutativity
FADD(a, b)  = FADD(b, a)
IADD(a, 0)  = a                              // identity
FADD(a, 0)  = a
LSHL(LSHR(a, n), n) = BAND(a, ~((1<<n)-1)) // shift-mask
ISUB(a, a)  = 0
MOV(a)      = a
BAND(a, 0xFFFFFFFF) = a
LOADI(0) → zero register
```

### Intermediate value novelty (no existing tool)

The current novelty search fingerprints programs by their *final* output
distribution. Extending this to *intermediate register state* at the midpoint
of execution would detect qualitatively different algorithms earlier:

- A program that has `FTI(x)` in a register partway through is doing something
  structurally different from one that never crosses the float/int boundary
- The behavior descriptor would be a vector of (register_index, step_fraction,
  output_distribution) tuples
- MAP-Elites over this richer descriptor would maintain diversity over
  qualitatively different computation paths

This is the "behavioral fragment indexing" idea — no published tool does it for
register-machine synthesis.

### Fragment library with domain-type tags

Tag each live instruction in the best individuals with its semantic domain:
`FLOAT_DOMAIN`, `INT_DOMAIN`, `LOG_APPROX_DOMAIN`. Crossover that preserves
domain type at cut points is more likely to produce valid compositions than
uniform positional crossover.

The specific transitions that matter:
- `ITF` / `FTI` are domain crossings — instructions immediately after them are
  in a different semantic world
- Arithmetic after a `FTI` is in log-space — an `ISUB` here is computing a
  float division; an `ASHR` is computing a float square root

---

## 7. The Multi-Crossing Problem

The evolved program crosses the float↔int boundary three times:

1. `IADD r0 r1 r0` — treats float `r0` as integer (implicit FTI)
2. `FMUL r0 r9 r10` — treats the integer result `r10` as float (implicit ITF)
3. `ISUB r8 r0 r4` — treats the float result `r0` as integer again

Each crossing compounds the mantissa approximation error (~0.04%). After three
crossings: `(1.0004)³ - 1 ≈ 0.12%` theoretical minimum error, before any
arithmetic error. The 1.45% observed error comes from the arithmetic between
crossings diverging from the log-linear model.

An e-graph cost function for this domain should penalize domain-crossing count,
not just instruction count. Programs with fewer FTI/ITF transitions have more
predictable behavior and are easier to optimize with algebraic rules.

---

## 8. Key Papers

| Paper | Relevance |
|---|---|
| [egg (POPL 2021)](https://arxiv.org/abs/2004.03082) | Foundational e-graph library |
| [eggp (arxiv 2501.17848)](https://arxiv.org/abs/2501.17848) | GP + e-graph dedup, most directly applicable |
| [Ruler (OOPSLA 2021)](https://www.mwillsey.com/papers/ruler) | Mining rewrite rules from examples |
| [MegaLibm (POPL 2024)](https://arxiv.org/abs/2311.01515) | From-scratch math function synthesis |
| [Guided EqSat (POPL 2024)](https://steuwer.info/files/publications/2024/POPL-Guided-Equality-Saturation.pdf) | Sketch-guided e-graph scaling |
| [STOKE (ASPLOS 2013)](https://theory.stanford.edu/~aiken/publications/papers/asplos13.pdf) | MCMC assembly synthesis |
| [STOKE FP (PLDI 2014)](https://theory.stanford.edu/~aiken/publications/papers/pldi14a.pdf) | MCMC synthesis of FP programs |
| [Herbie](https://herbie.uwplse.org/papers.html) | FP accuracy via bounded rewrites |

---

## 9. Open Questions / Things to Try

- **Ruler on our ISA**: give Ruler the register machine grammar + interpreter,
  let it mine exact algebraic rules for IADD, ISUB, BAND, shifts, etc. Then
  feed those rules to egg for deduplication.

- **Approximate rule mining**: extend the Ruler approach to mine rules with
  measured error bounds on sample inputs. Two program fragments that produce
  outputs within ε of each other on all test points go into the same behavioral
  bucket.

- **Intermediate register fingerprinting**: snapshot all 16 registers at 25%,
  50%, 75% of program execution. Use these as MAP-Elites behavior descriptors
  instead of just final output. Likely to discover the `FTI → arithmetic →
  ITF` structure more reliably.

- **Domain-typed crossover**: annotate each register at each program point with
  its inferred semantic domain (float, int, log-approx). Only allow crossover
  cuts at points where both parents have the same domain tag in the output
  register. Reduces the chance of producing nonsense compositions.

- **Cost function with crossing penalty**: add a term to fitness for each
  implicit FTI/ITF crossing (treating float bits as int or vice versa without
  an explicit conversion instruction). This rewards structurally clean programs
  and may guide evolution toward the explicit `FTI → magic_sub → ITF` pattern.
