#!/usr/bin/env python3
"""
Convert PySR hall-of-fame CSV → one micro-benchmark per equation.
CLI:  python csv2bench.py  <directory-containing-csv-files>
Each executable prints:
  #calls  <total μs spent in predict()>
"""
import csv, re, sys, pandas as pd
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit("Usage: python csv2bench.py  <directory-with-csvs>")
ROOT = Path(sys.argv[1]).expanduser().resolve()
CSV_FILE  = ROOT / "hall_of_fame.csv"
DATA_FILE = ROOT / "softmax_incremental.csv"
MAKEFILE  = ROOT / "Makefile"

# ------------------------------------------------------------------
def cppize(expr: str) -> str:
    expr = re.sub(r"\babs\b",  "std::abs",  expr)
    expr = re.sub(r"\bexp\b",  "std::exp",  expr)
    expr = re.sub(r"\blog\b",  "std::log",  expr)
    expr = re.sub(r"\bsqrt\b", "std::sqrt", expr)
    return expr

def emit_cpp(idx: int, eq: str, rows: int) -> Path:
    cpp = ROOT / f"eq_{idx:02d}.cpp"
    code = f"""
#include <cmath>
#include <chrono>
#include <cstdio>

constexpr struct {{ double m, s, x; }} data[{rows}] = {embed_data()};

inline double sr_predict(double m, double s, double x) {{
    (void)m;
    return {cppize(eq)};
}}

int main() {{
    constexpr std::size_t outer = 100'000;
    const std::size_t n = {rows};
    const std::size_t total_calls = outer * n;

    // warm-up
    for (std::size_t j = 0; j < n; ++j) {{
        auto [m, s, x] = data[j];
        volatile double sink = sr_predict(m, s, x);
        (void)sink;
    }}

    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < outer; ++i) {{
        for (std::size_t j = 0; j < n; ++j) {{
            auto [m, s, x] = data[j];
            volatile double sink = sr_predict(m, s, x);
            (void)sink;
        }}
    }}
    auto t1 = std::chrono::steady_clock::now();

    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double us_per_call = us / (total_calls / 1000);
    std::printf("%zu  %.2f: %f us/1k calls \\n", total_calls, us, us_per_call);
    return 0;
}}
"""
    cpp.write_text(code)
    return cpp

def embed_data() -> str:
    df = pd.read_csv(DATA_FILE, usecols=["m", "s", "x"])
    out = ["{"]
    for _, r in df.iterrows():
        out.append(f" {{{r.m},{r.s},{r.x}}},")
    out[-1] = out[-1].rstrip(",")
    out.append("}")
    return "\n".join(out)

def main():
    df = pd.read_csv(CSV_FILE)
    targets = []
    with open(MAKEFILE, "w") as mk:
        mk.write("CXXFLAGS = -O3 -march=native -std=c++17\n")
        mk.write("all:")
        for idx, row in df.iterrows():
            idx += 1
            cpp = emit_cpp(idx, row.Equation, len(pd.read_csv(DATA_FILE)))
            targets.append(cpp.stem)
            mk.write(f" {cpp.stem}")
        mk.write("\n\n")
        for t in targets:
            mk.write(f"{t}: {t}.cpp\n")
            mk.write(f"\t$(CXX) $(CXXFLAGS) -o $@ $<\n")
        mk.write("\nclean:\n\trm -f eq_??\n")
    print(f"Generated {len(targets)} benchmarks in {ROOT}")
    print(f"Cd into that folder and run:  make -j")

if __name__ == "__main__":
    main()
