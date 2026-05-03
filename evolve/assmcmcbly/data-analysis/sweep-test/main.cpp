#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cfloat>

// pull in the evolved candidate — path injected at compile time via -DTARGET_CPP
#ifndef TARGET_CPP
#  error "Pass -DTARGET_CPP='\"path/to/candidate.cpp\"' on the compiler command line"
#endif
#include TARGET_CPP

static float ref_invsqrt(float x) {
    return 1.0f / sqrtf(x);
}

int main() {
    printf("x,ref,jit,vm,jit_err,vm_err\n");

    // sweep positive normalised floats on a log scale: 2^-24 .. 2^24 in ~500k steps
    const float lo = 1.0f / (1 << 24);
    const float hi = (float)(1 << 24);
    const int   N  = 500000;

    for (int i = 0; i <= N; ++i) {
        float t = (float)i / (float)N;
        // log-linear interpolation
        float x = lo * powf(hi / lo, t);

        float r   = ref_invsqrt(x);
        float jit = jit_eval(x);
        float vm  = vm_eval(x);

        float jit_err = (r != 0.0f) ? (jit - r) / r : jit - r;
        float vm_err  = (r != 0.0f) ? (vm  - r) / r : vm  - r;

        printf("%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
               (double)x, (double)r, (double)jit, (double)vm,
               (double)jit_err, (double)vm_err);
    }
}
