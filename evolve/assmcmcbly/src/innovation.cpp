#include "innovation.hpp"

static uint32_t g_innov = 1;

uint32_t next_innovation()    { return g_innov++; }
void     reset_innovation(uint32_t start) { g_innov = start; }
uint32_t current_innovation() { return g_innov; }
