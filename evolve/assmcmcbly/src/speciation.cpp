#include "speciation.hpp"
#include "execute.hpp"
#include <cmath>
#include <algorithm>

static const float SAMPLE_X[N_BEH] = {0.01f, 0.1f, 1.0f, 10.0f, 100.0f};

void compute_fingerprint(const Program& p, float out[N_BEH]) {
    for (int i = 0; i < N_BEH; i++) {
        float o = execute(p, SAMPLE_X[i]);
        out[i] = std::isfinite(o) ? std::clamp(o, -1e6f, 1e6f) : 1e6f;
    }
}

double behavioral_distance(const float fa[N_BEH], const float fb[N_BEH]) {
    double dist = 0.0;
    for (int i = 0; i < N_BEH; i++)
        dist += std::abs(double(fa[i]) - double(fb[i]));
    return dist / N_BEH;
}
