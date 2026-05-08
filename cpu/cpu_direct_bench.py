#!/usr/bin/env python3
"""
Convert PySR hall-of-fame CSV → CPU throughput micro-benchmarks.
Mirrors cuda_direct_bench.py but for CPU with proper vectorized throughput.

CLI:  python cpu_direct_bench.py  <directory-with-csvs>  [options]

Options:
  --rows    INT   number of (m, s, x) elements   (default 1048576)
  --outer   INT   timing iterations               (default 2000)
  --warmup  INT   warmup iterations               (default 50)

Generates two binaries per equation:
  eq_XX_f32  — float32  (auto-vectorizes to 8-wide AVX2)
  eq_XX_f64  — float64  (auto-vectorizes to 4-wide AVX2)
"""

import argparse, re
import pandas as pd
from pathlib import Path

HERE      = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent

parser = argparse.ArgumentParser()
parser.add_argument("dir",             help="output directory for generated benchmarks")
parser.add_argument("--csv",    default=None,
                    help="equation CSV (default: auto-detect hall_of_fame.csv or equation_features.csv)")
parser.add_argument("--col",    default=None,
                    help="equation column name (default: auto-detect Equation/equation)")
parser.add_argument("--rows",   type=int, default=1<<20, help="number of data rows (default 1048576)")
parser.add_argument("--outer",  type=int, default=2000,  help="timing iterations (default 2000)")
parser.add_argument("--warmup", type=int, default=50,    help="warmup iterations (default 50)")
args = parser.parse_args()

ROOT     = Path(args.dir).expanduser().resolve()
ROOT.mkdir(parents=True, exist_ok=True)
MAKEFILE = ROOT / "Makefile"

# Auto-detect CSV source
if args.csv:
    CSV_FILE = Path(args.csv).expanduser().resolve()
else:
    candidates = [
        ROOT / "hall_of_fame.csv",
        HERE / "cpu_direct_features.csv",
        HERE / "cpu_equation_features.csv",
        REPO_ROOT / "gpu" / "equation_features.csv",
    ]
    CSV_FILE = next((p for p in candidates if p.exists()), None)
    if CSV_FILE is None:
        import sys; sys.exit("No equation CSV found. Pass --csv <file>")

N_ROWS = args.rows
OUTER  = args.outer
WARMUP = args.warmup


# ── expression → C++ ─────────────────────────────────────────────────────────

def cppize_f64(expr: str) -> str:
    expr = re.sub(r'\babs\b',  'std::abs',  expr)
    expr = re.sub(r'\bexp\b',  'std::exp',  expr)
    expr = re.sub(r'\blog\b',  'std::log',  expr)
    expr = re.sub(r'\bsqrt\b', 'std::sqrt', expr)
    return expr

def cppize_f32(expr: str) -> str:
    # Add f suffix to numeric literals and use float math functions
    def add_f(m):
        s = m.group(0)
        if s.endswith('f'):
            return s
        if '.' not in s and 'e' not in s.lower():
            return s + '.0f'
        return s + 'f'
    expr = re.sub(r'\b\d+(\.\d+)?([eE][+-]?\d+)?\b', add_f, expr)
    expr = re.sub(r'\babs\b',  'std::fabsf', expr)
    expr = re.sub(r'\bexp\b',  'std::expf',  expr)
    expr = re.sub(r'\blog\b',  'std::logf',  expr)
    expr = re.sub(r'\bsqrt\b', 'std::sqrtf', expr)
    return expr


# ── C++ template ──────────────────────────────────────────────────────────────

CPP_TMPL = """\
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#define N_ROWS  {n_rows}
#define OUTER   {outer}
#define WARMUP  {warmup}

// greater() matches PySR semantics (returns 1.0/0.0 as T)
#define greater(a, b) (static_cast<{T}>((a) > (b)))
#define neg(a)        (-(a))

static void gen_data({T}* m, {T}* s, {T}* x, int n) {{
    std::mt19937 rng(42);
    std::uniform_real_distribution<{T}> dm(0.0{suf}, 3.0{suf});
    std::uniform_real_distribution<{T}> ds(0.5{suf}, 5.0{suf});
    std::normal_distribution<{T}>       dx(0.0{suf}, 2.0{suf});
    for (int i = 0; i < n; ++i) {{ m[i] = dm(rng); s[i] = ds(rng); x[i] = dx(rng); }}
}}

inline {T} predict({T} m, {T} s, {T} x) {{
    (void)m; (void)s; (void)x;
    return {expr};
}}

int main() {{
    {T}* dm = ({T}*)malloc(N_ROWS * sizeof({T}));
    {T}* ds = ({T}*)malloc(N_ROWS * sizeof({T}));
    {T}* dx = ({T}*)malloc(N_ROWS * sizeof({T}));
    {T}* out= ({T}*)malloc(N_ROWS * sizeof({T}));
    gen_data(dm, ds, dx, N_ROWS);

    // warm-up
    for (int w = 0; w < WARMUP; ++w) {{
        for (int i = 0; i < N_ROWS; ++i)
            out[i] = predict(dm[i], ds[i], dx[i]);
        // prevent hoisting across warmup iterations
        __asm__ volatile("" : "+m"(*out) : "m"(*dm), "m"(*ds), "m"(*dx) : "memory");
    }}

    // barrier: compiler cannot move loads/stores across this
    __asm__ volatile("" : "+m"(*dm), "+m"(*ds), "+m"(*dx), "+m"(*out) : : "memory");

    auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < OUTER; ++iter) {{
        for (int i = 0; i < N_ROWS; ++i)
            out[i] = predict(dm[i], ds[i], dx[i]);
        __asm__ volatile("" : "+m"(*out) : : "memory");
    }}
    auto t1 = std::chrono::steady_clock::now();

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    long long total = (long long)OUTER * N_ROWS;
    double gcps = (double)total / ns;
    double ns_per = ns / (double)total;

    std::printf("{tag}  total=%lld  ns_per_call=%.3f  gcps=%.4f\\n",
                total, ns_per, gcps);

    free(dm); free(ds); free(dx); free(out);
    return 0;
}}
"""

def make_cpp(stem: str, expr_f64: str, expr_f32: str) -> tuple[str, str]:
    src_f64 = CPP_TMPL.format(
        T="double", suf="", tag=f"{stem}_f64",
        expr=expr_f64, n_rows=N_ROWS, outer=OUTER, warmup=WARMUP,
    )
    src_f32 = CPP_TMPL.format(
        T="float", suf="f", tag=f"{stem}_f32",
        expr=expr_f32, n_rows=N_ROWS, outer=OUTER, warmup=WARMUP,
    )
    return src_f64, src_f32


def write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)


def emit_eq(stem: str, raw_eq: str) -> None:
    f64_src, f32_src = make_cpp(stem, cppize_f64(raw_eq), cppize_f32(raw_eq))
    write_if_changed(ROOT / f"{stem}_f64.cpp", f64_src)
    write_if_changed(ROOT / f"{stem}_f32.cpp", f32_src)


def emit_baseline() -> None:
    baseline_f64 = "(x > m) ? (s * std::exp(m - x) + 1.0) : (s * std::exp(m - m) + std::exp(x - m))"
    baseline_f64 = "std::exp(x - std::fmax(m, x)) + s * std::exp(m - std::fmax(m, x))"
    baseline_f32 = "std::expf(x - std::fmaxf(m, x)) + s * std::expf(m - std::fmaxf(m, x))"
    f64_src, f32_src = make_cpp("baseline", baseline_f64, baseline_f32)
    write_if_changed(ROOT / "baseline_f64.cpp", f64_src)
    write_if_changed(ROOT / "baseline_f32.cpp", f32_src)


# ── Makefile ──────────────────────────────────────────────────────────────────

SUMMARY_SCRIPT = """\
#!/usr/bin/env python3
import sys, re
lines = sys.stdin.read().strip().split('\\n')
pat = re.compile(r'^(\\S+)\\s+total=\\d+\\s+ns_per_call=([\\d.]+)\\s+gcps=([\\d.]+)')
results = {}
for l in lines:
    m = pat.match(l)
    if m:
        results[m.group(1)] = (float(m.group(2)), float(m.group(3)))
base_f64 = results.get('baseline_f64', (None, None))[1]
base_f32 = results.get('baseline_f32', (None, None))[1]
print(f"{'Name':<22} {'ns/call':>8} {'Gcalls/s':>10} {'vs baseline':>12}")
print('-' * 57)
for name, (ns, gcps) in sorted(results.items()):
    base = base_f32 if name.endswith('_f32') else base_f64
    ratio = f'{gcps/base:.2f}x' if base else '  n/a'
    marker = ' <-- BEST' if gcps == max(v[1] for v in results.values()) else ''
    print(f'{name:<22} {ns:>8.3f} {gcps:>10.4f} {ratio:>12}{marker}')
"""


def emit_makefile(stems: list[str]) -> None:
    all_targets = [f"{s}_{t}" for s in stems for t in ("f64", "f32")]

    lines = [
        "CXX      = g++\n",
        "CXXFLAGS = -O3 -march=native -std=c++17 -funroll-loops -ffast-math\n",
        "all: " + " ".join(all_targets) + "\n\n",
    ]
    for t in all_targets:
        lines.append(f"{t}: {t}.cpp\n")
        lines.append(f"\t$(CXX) $(CXXFLAGS) -o $@ $<\n")

    lines.append("\nclean:\n\trm -f " + " ".join(all_targets) + "\n")

    # bench: run all, pipe to summary.py
    run_all = " && ".join(f"./{t}" for t in all_targets)
    lines.append("\nbench: all\n")
    lines.append(f"\t@( {run_all} ) | python3 summary.py\n")

    # bench-raw: unprocessed output for debugging
    lines.append("\nbench-raw: all\n")
    for s in stems:
        lines.append(f"\t@echo '--- {s} ---'\n")
        lines.append(f"\t@./{s}_f64\n")
        lines.append(f"\t@./{s}_f32\n")

    # vec-report: recompile with vectorization diagnostics (via shell script)
    lines.append("\nvec-report:\n")
    lines.append("\t@bash vec_report.sh\n")

    vec_script_lines = ["#!/bin/bash\n", "CXX=g++\n",
                        "FLAGS='-O3 -march=native -std=c++17 -funroll-loops -ffast-math'\n"]
    for s in stems:
        vec_script_lines.append(f'echo "=== {s}_f32 ==="\n')
        vec_script_lines.append(
            f'$CXX $FLAGS -fopt-info-vec-optimized -fopt-info-vec-missed '
            f'-o /dev/null {s}_f32.cpp 2>&1 | grep -v "^$" || true\n'
        )

    write_if_changed(MAKEFILE, "".join(lines))
    write_if_changed(ROOT / "summary.py", SUMMARY_SCRIPT)
    write_if_changed(ROOT / "vec_report.sh", "".join(vec_script_lines))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(CSV_FILE)

    # Auto-detect equation column
    col = args.col
    if col is None:
        for candidate in ("Equation", "equation"):
            if candidate in df.columns:
                col = candidate
                break
    if col is None:
        import sys; sys.exit(f"No equation column found in {CSV_FILE}. Pass --col <name>")

    stems = ["baseline"]
    emit_baseline()

    for idx, row in df.iterrows():
        stem = f"eq_{int(idx)+1:02d}"
        emit_eq(stem, str(row[col]))
        stems.append(stem)

    emit_makefile(stems)

    print(f"Generated {len(stems)*2} CPU benchmarks in {ROOT}")
    print(f"  N_ROWS={N_ROWS}  OUTER={OUTER}  WARMUP={WARMUP}")
    print(f"  *_f64 : double precision (4-wide AVX2)")
    print(f"  *_f32 : float  precision (8-wide AVX2)")
    print(f"Compile:  cd {ROOT} && make -j")
    print(f"Compare:  make bench")


if __name__ == "__main__":
    main()
