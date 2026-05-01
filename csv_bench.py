#!/usr/bin/env python3
"""
Convert PySR hall-of-fame CSV → one micro-benchmark per equation + baseline softmax.
CLI:  python csv2bench.py  <directory-containing-csv-files>
Each executable prints:
#calls  <total μs spent in predict()>
Data shared via common data.h header.
"""

import csv, re, sys, pandas as pd
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit("Usage: python csv2bench.py  <directory-with-csvs>")

ROOT = Path(sys.argv[1]).expanduser().resolve()
CSV_FILE  = ROOT / "hall_of_fame.csv"
DATA_FILE = ROOT / "softmax_incremental.csv"
DATA_H    = ROOT / "data.h"
MAKEFILE  = ROOT / "Makefile"

# ------------------------------------------------------------------

def emit_data_header(rows: int) -> Path:
    df = pd.read_csv(DATA_FILE, usecols=["m", "s", "x"])
    with open(DATA_H, "w") as f:
        f.write("#ifndef DATA_H\n")
        f.write("#define DATA_H\n\n")
        f.write(f"#include <cstddef>\n\n")
        f.write(f"#define greater(a,b) (a > b)\n\n")
        f.write(f"#define neg(a) (-a)\n\n")
        f.write("constexpr std::size_t N_ROWS = ")
        f.write(f"{rows};\n\n")
        f.write("constexpr struct { double m, s, x; } data[N_ROWS] = {\n")
        for i, (_, r) in enumerate(df.iterrows()):
            f.write(f"  {{ {r.m}, {r.s}, {r.x} }}")
            if i < rows - 1:
                f.write(",")
            f.write("\n")
        f.write("};\n\n")
        f.write("#endif\n")
    return DATA_H

def cppize(expr: str) -> str:
    expr = re.sub(r"\\babs\\b",  "std::abs",  expr)
    expr = re.sub(r"\\bexp\\b",  "std::exp",  expr)
    expr = re.sub(r"\\blog\\b",  "std::log",  expr)
    expr = re.sub(r"\\bsqrt\\b", "std::sqrt", expr)
    return expr

def emit_cpp(idx: int, eq: str) -> Path:
    cpp = ROOT / f"eq_{idx:02d}.cpp"
    code = f"""#include <cmath>
#include <chrono>
#include <cstdio>
#include "data.h"

inline double sr_predict(double m, double s, double x) {{
(void)m;
return {cppize(eq)};
}}

int main() {{
constexpr std::size_t outer = 100'000;
const std::size_t n = N_ROWS;
const std::size_t total_calls = outer * n;
// warm-up
for (std::size_t j = 0; j < n; ++j) {{
auto d = data[j];
volatile double sink = sr_predict(d.m, d.s, d.x);
(void)sink;
}}
auto t0 = std::chrono::steady_clock::now();
for (std::size_t i = 0; i < outer; ++i) {{
for (std::size_t j = 0; j < n; ++j) {{
auto d = data[j];
volatile double sink = sr_predict(d.m, d.s, d.x);
(void)sink;
}}}}
auto t1 = std::chrono::steady_clock::now();
double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
double us_per_call = us / (total_calls / 1000.0);
std::printf("%zu  %.2f: %.2f us/1k calls\\n", total_calls, us, us_per_call);
return 0;
}}"""
    cpp.write_text(code)
    return cpp

def emit_baseline() -> Path:
    baseline = ROOT / "baseline.cpp"
    code = f"""#include <cmath>
#include <chrono>
#include <cstdio>
#include "data.h"


inline double baseline_predict(double m, double s, double x) {{
double max_new = std::max(m, x);
double e_new = std::exp(x - max_new);
double exp_m_new = std::exp(m - max_new);
double exp_sum_new = s * exp_m_new + e_new;
return e_new / exp_sum_new;
}}

int main() {{
constexpr std::size_t outer = 100'000;
const std::size_t n = N_ROWS;
const std::size_t total_calls = outer * n;
// warm-up
for (std::size_t j = 0; j < n; ++j) {{
auto d = data[j];
volatile double sink = baseline_predict(d.m, d.s, d.x);
(void)sink;
}}
auto t0 = std::chrono::steady_clock::now();
for (std::size_t i = 0; i < outer; ++i) {{
for (std::size_t j = 0; j < n; ++j) {{
auto d = data[j];
volatile double sink = baseline_predict(d.m, d.s, d.x);
(void)sink;
}}}}
auto t1 = std::chrono::steady_clock::now();
double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
double us_per_call = us / (total_calls / 1000.0);
std::printf("%zu  %.2f: %.2f us/1k calls\\n", total_calls, us, us_per_call);
return 0;
}}"""
    baseline.write_text(code)
    return baseline

def main():
    df = pd.read_csv(CSV_FILE)
    rows = len(pd.read_csv(DATA_FILE))
    
    # Generate shared data header first
    emit_data_header(rows)
    
    targets = []
    
    # Generate baseline
    baseline = emit_baseline()
    targets.append(baseline.stem)
    
    # Generate equation benchmarks
    for idx, row in df.iterrows():
        idx += 1
        cpp = emit_cpp(idx, row.Equation)
        targets.append(cpp.stem)
    
    # Generate Makefile
    with open(MAKEFILE, "w") as mk:
        mk.write("CXXFLAGS = -O3 -march=native -std=c++17\n")
        mk.write("all:")
        for t in targets:
            mk.write(f" {t}")
        mk.write("\n\n")
        for t in targets:
            mk.write(f"{t}: {t}.cpp data.h\n")
            mk.write("\t$(CXX) $(CXXFLAGS) -o $@ $<\n")
        mk.write("\n")
        mk.write("clean:\n")
        mk.write("\trm -f baseline eq_?? data.h\n")
    
    print(f"Generated {len(targets)} benchmarks + data.h in {ROOT}")
    print(f"  - data.h: shared dataset (N_ROWS={rows})")
    print(f"  - baseline.cpp: standard offline softmax update")
    print(f"  - eq_*.cpp: PySR equations")
    print(f"Cd into that folder and run:  make -j")
    print("Compare baseline vs eq_* times.")

if __name__ == "__main__":
    main()
