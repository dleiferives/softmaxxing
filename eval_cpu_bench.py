#!/usr/bin/env python3
"""
CPU benchmark pipeline for SR softmax equations.

Reads equations from a CSV (default: equation_features.csv), generates C++
microbenchmarks compiled at -O3 -march=native, times them, and appends results
to cpu_bench_results.csv (separate from the GPU bench_results.csv).

Usage:
  python eval_cpu_bench.py
  python eval_cpu_bench.py --source my_hof.csv --col Equation
  python eval_cpu_bench.py --force
"""

import argparse, re, subprocess, sys, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent.resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--source",  default=str(HERE / "equation_features.csv"),
                    help="CSV with equations (default: equation_features.csv)")
parser.add_argument("--col",     default="equation",
                    help="column name for the equation string (default: equation)")
parser.add_argument("--loss-col", default="loss",
                    help="column name for loss (default: loss)")
parser.add_argument("--results", default=str(HERE / "cpu_bench_results.csv"),
                    help="output CSV (default: cpu_bench_results.csv)")
parser.add_argument("--data",    default=str(HERE / "softmax_incremental.csv"),
                    help="dataset CSV with m,s,x,s_new columns")
parser.add_argument("--outer",   type=int, default=2000,
                    help="timing iterations over full dataset (default 2000)")
parser.add_argument("--workers", type=int, default=4,
                    help="parallel compile workers (default 4)")
parser.add_argument("--force",   action="store_true",
                    help="re-run even if equation already cached")
args = parser.parse_args()

RESULTS   = Path(args.results)
DATA_CSV  = Path(args.data)
OUTER     = args.outer


# ── expression → C++ ─────────────────────────────────────────────────────────

def cppize(expr: str) -> str:
    expr = re.sub(r'\babs\b',  'std::abs',  expr)
    expr = re.sub(r'\bexp\b',  'std::exp',  expr)
    expr = re.sub(r'\blog\b',  'std::log',  expr)
    expr = re.sub(r'\bsqrt\b', 'std::sqrt', expr)
    return expr


# ── shared data header ────────────────────────────────────────────────────────

DATA_HEADER_TMPL = """\
#ifndef DATA_H
#define DATA_H
#include <cstddef>

// greater() returns double 1.0/0.0 to match GPU semantics
#define greater(a, b) ((double)((a) > (b)))
#define neg(a)        (-(a))

constexpr std::size_t N_ROWS = {n};

constexpr struct {{ double m, s, x; }} data[N_ROWS] = {{
{rows}
}};

#endif
"""

def write_data_header(build_dir: Path) -> None:
    df = pd.read_csv(DATA_CSV, usecols=["m", "s", "x"])
    rows = []
    for _, r in df.iterrows():
        rows.append(f"  {{ {r.m}, {r.s}, {r.x} }}")
    content = DATA_HEADER_TMPL.format(n=len(df), rows=",\n".join(rows))
    (build_dir / "data.h").write_text(content)


# ── benchmark C++ template ────────────────────────────────────────────────────

CPP_TMPL = """\
#include <cmath>
#include <chrono>
#include <cstdio>
#include "data.h"

inline double predict(double m, double s, double x) {{
    (void)m; (void)s; (void)x;
    return {expr};
}}

int main() {{
    constexpr std::size_t outer = {outer};
    const std::size_t n = N_ROWS;
    const std::size_t total_calls = outer * n;

    // warm-up
    for (std::size_t j = 0; j < n; ++j) {{
        auto d = data[j];
        volatile double sink = predict(d.m, d.s, d.x);
        (void)sink;
    }}

    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < outer; ++i) {{
        for (std::size_t j = 0; j < n; ++j) {{
            auto d = data[j];
            volatile double sink = predict(d.m, d.s, d.x);
            (void)sink;
        }}
    }}
    auto t1 = std::chrono::steady_clock::now();

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    double ns_per_call = ns / (double)total_calls;
    double mcps = 1.0e9 / ns_per_call / 1.0e6;  // million calls/sec

    std::printf("total_calls=%zu  ns_per_call=%.3f  mcps=%.4f\\n",
                total_calls, ns_per_call, mcps);
    return 0;
}}
"""

BASELINE_EXPR = "exp(x - m_new) + s * exp(m - m_new)"

BASELINE_CPP_TMPL = """\
#include <cmath>
#include <chrono>
#include <cstdio>
#include "data.h"

inline double predict(double m, double s, double x) {
    double m_new = m > x ? m : x;
    return std::exp(x - m_new) + s * std::exp(m - m_new);
}

int main() {
    constexpr std::size_t outer = """ + str(OUTER) + """;
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

    std::printf("total_calls=%zu  ns_per_call=%.3f  mcps=%.4f\\n",
                total_calls, ns_per_call, mcps);
    return 0;
}
"""

def write_cpp(build_dir: Path, name: str, expr: str) -> Path:
    p = build_dir / f"{name}.cpp"
    if name == "baseline":
        p.write_text(BASELINE_CPP_TMPL)
    else:
        p.write_text(CPP_TMPL.format(expr=cppize(expr), outer=OUTER))
    return p


# ── Makefile ──────────────────────────────────────────────────────────────────

def write_makefile(build_dir: Path, targets: list[str]) -> None:
    lines = ["CXXFLAGS = -O3 -march=native -std=c++17\n"]
    lines.append("all: " + " ".join(targets) + "\n\n")
    for t in targets:
        lines.append(f"{t}: {t}.cpp data.h\n\t$(CXX) $(CXXFLAGS) -o $@ $<\n")
    lines.append("\nclean:\n\trm -f " + " ".join(targets) + " data.h\n")
    (build_dir / "Makefile").write_text("".join(lines))


# ── output parser ─────────────────────────────────────────────────────────────

OUTPUT_RE = re.compile(
    r"total_calls=(\d+)\s+ns_per_call=([\d.]+)\s+mcps=([\d.]+)"
)

def run_binary(binary: Path) -> dict | None:
    try:
        r = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
    except Exception as e:
        print(f"  [run] {binary.name} ERROR: {e}")
        return None
    if r.returncode != 0:
        print(f"  [run] {binary.name} FAILED:\n{r.stderr.strip()}")
        return None
    m = OUTPUT_RE.search(r.stdout)
    if not m:
        print(f"  [run] {binary.name} unexpected output: {r.stdout!r}")
        return None
    return dict(
        total_calls=int(m.group(1)),
        ns_per_call=float(m.group(2)),
        mcps=float(m.group(3)),
    )


# ── compile one target ────────────────────────────────────────────────────────

def compile_one(build_dir: Path, name: str) -> tuple[str, bool, str]:
    cpp    = build_dir / f"{name}.cpp"
    binary = build_dir / name
    if binary.exists() and binary.stat().st_mtime >= cpp.stat().st_mtime:
        return name, True, "(cached)"
    r = subprocess.run(
        ["g++", "-O3", "-march=native", "-std=c++17", "-o", str(binary), str(cpp)],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        binary.chmod(0o755)
    return name, r.returncode == 0, r.stderr.strip()


# ── main ──────────────────────────────────────────────────────────────────────

def load_cache() -> set[str]:
    if not RESULTS.exists():
        return set()
    df = pd.read_csv(RESULTS, usecols=["equation"])
    return set(df["equation"].dropna().tolist())


def main():
    src = Path(args.source)
    if not src.exists():
        sys.exit(f"Source not found: {src}")

    df_src = pd.read_csv(src)
    if args.col not in df_src.columns:
        sys.exit(f"Column '{args.col}' not in {src}. Available: {list(df_src.columns)}")

    equations = df_src[args.col].dropna().tolist()
    losses    = (df_src[args.loss_col].tolist()
                 if args.loss_col in df_src.columns
                 else [float("nan")] * len(equations))

    cached = load_cache()
    pending = [(eq, loss) for eq, loss in zip(equations, losses)
               if args.force or eq not in cached]

    if not pending:
        print("All equations cached. Use --force to re-run.")
        return

    print(f"Benchmarking {len(pending)} equation(s)  [outer={OUTER}, workers={args.workers}]")

    build_dir = HERE / "cpu_bench_build"
    build_dir.mkdir(exist_ok=True)

    # Write shared data header once
    print("Writing data.h ...", end=" ", flush=True)
    write_data_header(build_dir)
    print("done")

    # Generate C++ sources
    eq_map: dict[str, tuple[str, float]] = {}  # name → (equation, loss)
    targets = ["baseline"]
    write_cpp(build_dir, "baseline", BASELINE_EXPR)
    eq_map["baseline"] = ("baseline (2x exp)", float("nan"))

    for i, (eq, loss) in enumerate(pending):
        name = f"eq_{i+1:03d}"
        write_cpp(build_dir, name, eq)
        eq_map[name] = (eq, loss)
        targets.append(name)

    write_makefile(build_dir, targets)

    # Parallel compile
    print(f"\n── Compiling {len(targets)} target(s) ──")
    compiled_ok: set[str] = set()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compile_one, build_dir, t): t for t in targets}
        done = 0
        for future in as_completed(futures):
            name, ok, msg = future.result()
            done += 1
            status = msg if msg == "(cached)" else ("ok" if ok else f"FAILED: {msg}")
            print(f"  [{done:>{len(str(len(targets)))}}/{len(targets)}] {name:<14} {status}",
                  flush=True)
            if ok:
                compiled_ok.add(name)

    # Run binaries sequentially
    print(f"\n── Running {len(compiled_ok)} benchmark(s) ──")
    new_rows: list[dict] = []
    for name in targets:
        if name not in compiled_ok:
            continue
        eq, loss = eq_map[name]
        print(f"  {name:<14} {eq[:60]:<60}", end=" ", flush=True)
        timing = run_binary(build_dir / name)
        if timing is None:
            continue
        print(f"→ {timing['ns_per_call']:.1f} ns/call  {timing['mcps']:.1f} Mcps")
        row = dict(
            name=name,
            equation=eq,
            loss=loss,
            outer=OUTER,
            n_rows=sum(1 for _ in open(DATA_CSV)) - 1,
            **timing,
        )
        new_rows.append(row)
        chunk = pd.DataFrame([row])
        write_header = not RESULTS.exists()
        chunk.to_csv(RESULTS, mode="a", index=False, header=write_header)
        cached.add(eq)

    print(f"\nDone. {len(new_rows)} result(s) written to {RESULTS}")


if __name__ == "__main__":
    main()
