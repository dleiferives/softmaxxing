#include <cmath>
#include <chrono>
#include <cstdio>
#include "data.h"

inline double predict(double m, double s, double x) {
    double m_new = m > x ? m : x;
    return std::exp(x - m_new) + s * std::exp(m - m_new);
}

int main() {
    constexpr std::size_t outer = 2000;
    const std::size_t n = N_ROWS;
    const std::size_t total_calls = outer * n;

    for (std::size_t j = 0; j < n; ++j) {
        auto d = data[j];
        volatile double sink = predict(d.m, d.s, d.x);
        (void)sink;
    }

    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < outer; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            auto d = data[j];
            volatile double sink = predict(d.m, d.s, d.x);
            (void)sink;
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    double ns_per_call = ns / (double)total_calls;
    double mcps = 1.0e9 / ns_per_call / 1.0e6;

    std::printf("total_calls=%zu  ns_per_call=%.3f  mcps=%.4f\n",
                total_calls, ns_per_call, mcps);
    return 0;
}
