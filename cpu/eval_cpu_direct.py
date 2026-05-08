#!/usr/bin/env python3
"""
CPU throughput benchmark pipeline - scans outputs/ HOF CSVs, times every
unique equation with f32 and f64 throughput benchmarks.

Mirrors gpu/eval_bench.py but measures Gcalls/s instead of G-updates/s.

Usage:
  python eval_cpu_direct.py              # scan outputs/ for all hof_*.csv
  python eval_cpu_direct.py --source equation_features.csv
  python eval_cpu_direct.py --force      # re-run even if cached
"""

import argparse, hashlib, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

HERE      = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent

parser = argparse.ArgumentParser()
parser.add_argument("--source",  default=None,
                    help="single CSV with equations (default: scan outputs/)")
parser.add_argument("--col",     default=None,
                    help="equation column (default: auto-detect)")
parser.add_argument("--loss-col", default=None)
parser.add_argument("--results", default=str(HERE / "cpu_direct_results.csv"))
parser.add_argument("--build",   default=str(HERE.parent / "cpu_bench_build"),
                    help="build directory for C++ benchmarks")
parser.add_argument("--rows",    type=int, default=1<<20)
parser.add_argument("--outer",   type=int, default=500)
parser.add_argument("--warmup",  type=int, default=20)
parser.add_argument("--workers", type=int, default=6)
parser.add_argument("--force",   action="store_true")
args = parser.parse_args()

RESULTS   = Path(args.results)
BUILD_DIR = Path(args.build)
BUILD_DIR.mkdir(parents=True, exist_ok=True)

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
    def add_f(m):
        s = m.group(0)
        if s.endswith('f'): return s
        if '.' not in s and 'e' not in s.lower(): return s + '.0f'
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
#include <random>

#define N_ROWS  {n_rows}
#define OUTER   {outer}
#define WARMUP  {warmup}

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

    for (int w = 0; w < WARMUP; ++w) {{
        for (int i = 0; i < N_ROWS; ++i)
            out[i] = predict(dm[i], ds[i], dx[i]);
        __asm__ volatile("" : "+m"(*out) : "m"(*dm), "m"(*ds), "m"(*dx) : "memory");
    }}

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

    std::printf("{tag}  total=%lld  ns_per_call=%.4f  gcps=%.6f\\n",
                total, ns_per, gcps);

    free(dm); free(ds); free(dx); free(out);
    return 0;
}}
"""

BASELINE_F64 = "std::exp(x - std::fmax(m, x)) + s * std::exp(m - std::fmax(m, x))"
BASELINE_F32 = "std::expf(x - std::fmaxf(m, x)) + s * std::expf(m - std::fmaxf(m, x))"

OUTPUT_RE = re.compile(
    r'(\S+)\s+total=\d+\s+ns_per_call=([\d.]+)\s+gcps=([\d.]+)'
)


def eq_hash(eq: str) -> str:
    return hashlib.md5(eq.encode()).hexdigest()[:10]


def write_cpp(name: str, expr_f64: str, expr_f32: str) -> tuple[Path, Path]:
    src_f64 = CPP_TMPL.format(
        T="double", suf="", tag=f"{name}_f64",
        expr=expr_f64, n_rows=N_ROWS, outer=OUTER, warmup=WARMUP,
    )
    src_f32 = CPP_TMPL.format(
        T="float", suf="f", tag=f"{name}_f32",
        expr=expr_f32, n_rows=N_ROWS, outer=OUTER, warmup=WARMUP,
    )
    p64 = BUILD_DIR / f"{name}_f64.cpp"
    p32 = BUILD_DIR / f"{name}_f32.cpp"
    if not p64.exists() or p64.read_text() != src_f64: p64.write_text(src_f64)
    if not p32.exists() or p32.read_text() != src_f32: p32.write_text(src_f32)
    return p64, p32


def compile_one(name: str, suffix: str) -> tuple[str, bool, str]:
    cpp    = BUILD_DIR / f"{name}_{suffix}.cpp"
    binary = BUILD_DIR / f"{name}_{suffix}"
    if binary.exists() and binary.stat().st_mtime >= cpp.stat().st_mtime:
        return f"{name}_{suffix}", True, "(cached)"
    r = subprocess.run(
        ["g++", "-O3", "-march=native", "-std=c++17", "-funroll-loops", "-ffast-math",
         "-o", str(binary), str(cpp)],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        binary.chmod(0o755)
    return f"{name}_{suffix}", r.returncode == 0, r.stderr.strip()


def run_binary(name: str, suffix: str) -> float | None:
    binary = BUILD_DIR / f"{name}_{suffix}"
    try:
        r = subprocess.run([str(binary)], capture_output=True, text=True, timeout=300)
    except Exception as e:
        print(f"  [run] {name}_{suffix} ERROR: {e}")
        return None
    if r.returncode != 0:
        print(f"  [run] {name}_{suffix} FAILED:\n{r.stderr.strip()}")
        return None
    m = OUTPUT_RE.search(r.stdout)
    if not m:
        print(f"  [run] {name}_{suffix} bad output: {r.stdout!r}")
        return None
    return float(m.group(3))  # gcps


def load_cache() -> set[str]:
    if not RESULTS.exists():
        return set()
    df = pd.read_csv(RESULTS, usecols=["equation"])
    return set(df["equation"].dropna().tolist())


def find_hof_csvs() -> list[Path]:
    outputs = REPO_ROOT / "outputs"
    if not outputs.exists():
        return []
    csvs = sorted(outputs.rglob("hof_*.csv"))
    return csvs


def collect_equations(csvs: list[Path]) -> list[tuple[str, float, float]]:
    """Returns list of (equation, complexity, loss), deduplicated."""
    seen = set()
    rows = []
    for csv in csvs:
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        col = next((c for c in ("equation", "Equation") if c in df.columns), None)
        if col is None:
            continue
        loss_col  = next((c for c in ("loss", "Loss") if c in df.columns), None)
        cplx_col  = next((c for c in ("complexity", "Complexity") if c in df.columns), None)
        for _, row in df.iterrows():
            eq = str(row[col]).strip()
            # skip label entries and anything that's not a parseable expression
            if not eq or eq in seen or eq.startswith("baseline"):
                continue
            if not any(c in eq for c in ('+', '-', '*', '>')):
                continue
            seen.add(eq)
            loss = float(row[loss_col]) if loss_col else float("nan")
            cplx = float(row[cplx_col]) if cplx_col else float("nan")
            rows.append((eq, cplx, loss))
    return rows


def main():
    # ── gather equations ──────────────────────────────────────────────────────
    if args.source:
        src = Path(args.source)
        if not src.exists():
            sys.exit(f"Source not found: {src}")
        equations = collect_equations([src])
    else:
        csvs = find_hof_csvs()
        if not csvs:
            sys.exit(f"No hof_*.csv found under {REPO_ROOT / 'outputs'}. "
                     f"Run an SR script first or pass --source.")
        print(f"Found {len(csvs)} HOF CSV(s) under outputs/")
        equations = collect_equations(csvs)

    print(f"Total unique equations: {len(equations)}")

    cached = load_cache()
    # always bench the baseline
    pending_eqs = [(eq, c, l) for eq, c, l in equations
                   if args.force or eq not in cached]

    print(f"New equations to bench: {len(pending_eqs)}")
    if not pending_eqs:
        print("All cached. Use --force to re-run.")
        return

    # ── write baseline ────────────────────────────────────────────────────────
    baseline_name = "baseline"
    write_cpp(baseline_name, BASELINE_F64, BASELINE_F32)

    # ── write equation C++ files ──────────────────────────────────────────────
    eq_names: dict[str, tuple[str, float, float]] = {}  # name → (eq, cplx, loss)
    compile_targets: list[tuple[str, str]] = [(baseline_name, "f32"), (baseline_name, "f64")]

    for eq, cplx, loss in pending_eqs:
        name = f"eq_{eq_hash(eq)}"
        write_cpp(name, cppize_f64(eq), cppize_f32(eq))
        eq_names[name] = (eq, cplx, loss)
        compile_targets.append((name, "f32"))
        compile_targets.append((name, "f64"))

    # ── parallel compile ──────────────────────────────────────────────────────
    print(f"\n── Compiling {len(compile_targets)} target(s) with {args.workers} workers ──")
    compiled_ok: set[tuple[str, str]] = set()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compile_one, name, suf): (name, suf)
                   for name, suf in compile_targets}
        done = 0
        for future in as_completed(futures):
            tgt, ok, err = future.result()
            done += 1
            status = err if err == "(cached)" else ("ok" if ok else f"FAILED: {err[:60]}")
            print(f"  [{done:>3}/{len(compile_targets)}] {tgt:<22} {status}", flush=True)
            if ok:
                name, suf = futures[future]
                compiled_ok.add((name, suf))

    # ── get baseline gcps ─────────────────────────────────────────────────────
    base_f32 = run_binary(baseline_name, "f32") or float("nan")
    base_f64 = run_binary(baseline_name, "f64") or float("nan")
    print(f"\n  baseline  f32={base_f32:.4f} Gcalls/s  f64={base_f64:.4f} Gcalls/s")

    # ── run equations ─────────────────────────────────────────────────────────
    print(f"\n── Running {len(eq_names)} equation(s) ──")
    new_rows = []

    for name, (eq, cplx, loss) in eq_names.items():
        gcps_f32 = gcps_f64 = float("nan")

        if (name, "f32") in compiled_ok:
            gcps_f32 = run_binary(name, "f32") or float("nan")
        if (name, "f64") in compiled_ok:
            gcps_f64 = run_binary(name, "f64") or float("nan")

        speedup_f32 = gcps_f32 / base_f32 if base_f32 else float("nan")
        speedup_f64 = gcps_f64 / base_f64 if base_f64 else float("nan")

        print(f"  {name}  f32={gcps_f32:.3f}G ({speedup_f32:.2f}x)  "
              f"f64={gcps_f64:.3f}G ({speedup_f64:.2f}x)  "
              f"loss={loss:.5g}  {eq[:50]}")

        row = dict(
            equation=eq, complexity=cplx, loss=loss,
            f32_gcps=gcps_f32, f64_gcps=gcps_f64,
            f32_speedup=speedup_f32, f64_speedup=speedup_f64,
            baseline_f32_gcps=base_f32, baseline_f64_gcps=base_f64,
            n_rows=N_ROWS, outer=OUTER,
        )
        new_rows.append(row)
        chunk = pd.DataFrame([row])
        write_header = not RESULTS.exists()
        chunk.to_csv(RESULTS, mode="a", index=False, header=write_header)

    print(f"\nDone. {len(new_rows)} result(s) → {RESULTS}")
    if new_rows:
        best = max(new_rows, key=lambda r: r["f32_gcps"])
        print(f"Best f32: {best['f32_gcps']:.4f} Gcalls/s ({best['f32_speedup']:.2f}x baseline)")
        print(f"  eq: {best['equation'][:80]}")


if __name__ == "__main__":
    main()
