#pragma once
#include <cstdint>

// Global innovation counter.  Every structurally new instruction (one created
// by a growth mutation or during random initialisation) gets a unique number.
// Field-level mutations (changing op/dst/src/lit on an existing instruction)
// leave the innovation number unchanged so alignment-based crossover still works.

uint32_t next_innovation();
void     reset_innovation(uint32_t start = 1);
uint32_t current_innovation();
