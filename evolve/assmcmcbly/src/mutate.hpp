#pragma once
#include "program.hpp"
#include "hardening.hpp"
#include "problem.hpp"
#include <random>
#include <vector>

// Apply mutation operator `kind` (0–8) to src.
// kind values:
//   0  change_op            change a node's operation
//   1  rewire_src           redirect a src edge to a different upstream node
//   2  change_literal       perturb a LOADI/LOADF node's value
//   3  mcmc_literal         MCMC search over a literal node's value
//   4  insert_node          append a new random function node
//   5  remove_node          make a live node inactive by redirecting its consumers
//   6  change_output        move output_node to a different node
//   7  redirect_to_inactive activate an inactive node by wiring a consumer to it
//   8  add_constant         append a new LOADI/LOADF constant node
Program mutate(const Program& src, const Hardness& hardness,
               int kind, std::mt19937& rng,
               const ProblemDef& problem,
               const std::vector<float>& test_inputs);

// Innovation-aligned positional crossover.
// Fitter parent's structure (n_nodes, output_node) dominates.
// At each node index shared by both parents, 40% chance to take that node
// from the weaker parent.
Program crossover(const Program& a, double fa,
                  const Program& b, double fb,
                  std::mt19937& rng);
