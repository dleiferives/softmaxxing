#include "execute.hpp"
#include "jit.hpp"
#include "random.hpp"
#include "print.hpp"
#include <cstdio>
#include <cstring>
#include <random>

static constexpr int N_PROGRAMS    = 2000;
static constexpr int N_INPUTS      = 32;
static constexpr int RNG_SEED      = 42;

// Inputs: spread of positive floats + edge cases
static float make_inputs(float out[N_INPUTS]) {
    // 24 log-spaced positive floats
    for (int i = 0; i < 24; ++i)
        out[i] = 1e-4f * std::pow(1e8f, float(i) / 23.0f);
    // edge cases
    out[24] =  0.0f;
    out[25] = -0.0f;
    out[26] =  1.0f;
    out[27] = -1.0f;
    out[28] =  1e38f;
    out[29] = -1e38f;
    out[30] =  std::numeric_limits<float>::infinity();
    out[31] = -std::numeric_limits<float>::infinity();
    return 0;
}

int main() {
    std::mt19937 rng(RNG_SEED);
    float inputs[N_INPUTS];
    make_inputs(inputs);

    int failures = 0;

    for (int p = 0; p < N_PROGRAMS; ++p) {
        Program prog = random_program(rng);
        JitProgram jit = jit_compile(prog);

        for (int i = 0; i < N_INPUTS; ++i) {
            float vm_out  = execute(prog, inputs[i]);
            float jit_out = jit.fn(inputs[i]);

            uint32_t vm_bits, jit_bits;
            memcpy(&vm_bits,  &vm_out,  4);
            memcpy(&jit_bits, &jit_out, 4);

            if (vm_bits != jit_bits) {
                ++failures;
                printf("FAIL prog=%d input=%d (%.6e)\n", p, i, (double)inputs[i]);
                printf("     vm =0x%08x (%.8e)\n", vm_bits,  (double)vm_out);
                printf("     jit=0x%08x (%.8e)\n", jit_bits, (double)jit_out);
                if (failures == 1) {
                    printf("  Program:\n");
                    print_program(prog);
                }
                if (failures >= 10) {
                    printf("Too many failures, stopping.\n");
                    return 1;
                }
            }
        }
    }

    if (failures == 0)
        printf("OK  %d programs x %d inputs — JIT and VM agree on all outputs\n",
               N_PROGRAMS, N_INPUTS);

    return failures > 0 ? 1 : 0;
}
