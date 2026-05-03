# TODO

## NEAT-style instruction introduction

Every structurally-inserted instruction gets a global innovation number stamped on it. Crossover aligns by innovation number instead of position so programs of different sizes recombine cleanly. Growth mutations: add-instruction (insert at random position, bias src toward nearby live output), split-chromosome (cut one chrom into two at internal point), merge-chromosome (join two adjacent chroms if combined len fits). Speciation groups individuals by genomic distance (excess+disjoint gene count + weight diff); individuals compete within species so fresh structure isn't immediately killed by fitter compact programs. Fitness sharing divides raw fitness by species size so big species don't crowd out small exploratory ones.

## Adversarial / minimax

Co-evolving test input generator whose job is to maximize the error of the current best program. Fixed test inputs let a constant ~0.193 be "good enough" on average — an adversary immediately finds the extreme x values where it fails. Blend adversarial points with the fixed set for fitness evaluation. This directly breaks premature convergence to constants.

## Behavioral fingerprinting

Run a chromosome (or any contiguous instruction slice) on a fixed set of probe inputs and record the resulting register state tuple. Hash those tuples into a 64-bit fingerprint — same fingerprint means same behavior regardless of which ops produced it.

Three levels of resolution. Register-state signature: run on N probes, hash the full (r0..r15) output tuple per probe, exact match. Bucketed float signature: quantize each output float to a few significant bits before hashing, giving fuzzy equivalence so slices that do "approximately the same thing" cluster together. Output vector distance: collect only r0 across all 100 existing test inputs as a length-100 float vector, compare by Euclidean distance — expensive but captures the full behavioral shape including how much it varies with x.

Where this plugs in. Novelty pressure: reward programs whose behavioral signature is far from anything seen before, directly killing the constant-output trap. Crossover alignment: prefer recombining chromosomes with complementary signatures (one does the x-dependent transform, one does the scaling). Deduplication: if two individuals in the same species share a fingerprint they are redundant, replace one with a fresh mutation.

The register-state signature is the right starting point — cheap, fits the existing execution model, and immediately identifies the constant-output plague.

## Dreaming

Two flavors. Fast pre-screen: evaluate mutation candidates on ~10 points before committing to full 100-point fitness eval, skip obvious duds early. Deep local search: periodically run MCMC on entire instruction sequences (not just literal constants) for elite individuals, find good neighborhoods before handing back to evolution.

## Runtime optimization

For population that is going to be tested. Put their instruction strips next to each other. Make them load from mem at the start the value to use -> store r0 to index in array that corresponds to them. So we don't have to constantly be switching in and out of program scope yk. Should increase performance.
