#!/usr/bin/env python3
"""
Compile and evaluate CUDA softmax benchmarks, storing results to a CSV.

Usage:
  python eval_bench.py <dir>           # single output directory
  python eval_bench.py --all           # all subdirs under outputs/ with hall_of_fame.csv
  python eval_bench.py --all --force   # rerun even if already cached

Results appended to bench_results.csv (change with --results).
Cache key is the equation string — if an equation has already been timed it is
skipped regardless of which run directory it came from. Use --force to override.

Compilation is parallelised across 4 workers (--workers to change).
"""

import argparse, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pathlib import Path

HERE      = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Compile + run CUDA softmax benchmarks")
grp = parser.add_mutually_exclusive_group(required=True)
grp.add_argument("dir",   nargs="?", help="single output directory to process")
grp.add_argument("--all", action="store_true", help="process every subdir of outputs/ with hall_of_fame.csv")

parser.add_argument("--results", default=str(HERE / "bench_results.csv"),
                    help="CSV to append results to (default: bench_results.csv)")
parser.add_argument("--force",   action="store_true", help="rerun even if already cached")
parser.add_argument("--workers", type=int, default=4, help="parallel nvcc workers (default 4)")

parser.add_argument("--batch",  type=int, default=32,   help="batch size (default 32)")
parser.add_argument("--heads",  type=int, default=32,   help="number of attention heads (default 32)")
parser.add_argument("--seq",    type=int, default=2048, help="sequence length (default 2048)")
parser.add_argument("--outer",  type=int, default=200,  help="timing iterations (default 200)")
parser.add_argument("--warmup", type=int, default=10,   help="warmup iterations (default 10)")

args = parser.parse_args()

RESULTS  = Path(args.results)
CUDA_GEN = HERE / "cuda_bench.py"

NVCC_FLAGS = ["-O3", "-arch=native", "--use_fast_math", "-lcurand"]

# ── helpers ───────────────────────────────────────────────────────────────────

OUTPUT_RE = re.compile(
    r"total_updates=(\d+)\s+time=([\d.]+)\s+ms\s+throughput=([\d.]+)\s+G-updates/s\s+([\d.]+)\s+us/seq"
)

def find_dirs() -> list[Path]:
    outputs = REPO_ROOT / "outputs"
    if not outputs.exists():
        sys.exit(f"No outputs/ directory found under {REPO_ROOT}")
    return sorted(p for p in outputs.iterdir() if p.is_dir() and (p / "hall_of_fame.csv").exists())


def load_cache() -> set[str]:
    if not RESULTS.exists():
        return set()
    df = pd.read_csv(RESULTS, usecols=["equation"])
    return set(df["equation"].dropna().tolist())


BASELINE_EQ = "baseline (2x __expf)"

def build_eq_map(run_dir: Path) -> dict[str, tuple[str, float, float]]:
    """kernel-name → (equation, complexity, loss)"""
    hof = pd.read_csv(run_dir / "hall_of_fame.csv")
    eq_map: dict[str, tuple[str, float, float]] = {
        "baseline": (BASELINE_EQ, float("nan"), float("nan"))
    }
    for i, row in hof.iterrows():
        eq_map[f"eq_{int(i)+1:02d}"] = (row["Equation"], row["Complexity"], row["Loss"])
    return eq_map


# ── phase 1: generate .cu files ───────────────────────────────────────────────

def gen_cu_files(run_dir: Path) -> bool:
    """Run cuda_bench.py to (re)generate all .cu files in run_dir."""
    cmd = [
        sys.executable, str(CUDA_GEN), str(run_dir),
        "--batch",  str(args.batch),
        "--heads",  str(args.heads),
        "--seq",    str(args.seq),
        "--outer",  str(args.outer),
        "--warmup", str(args.warmup),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [gen] FAILED {run_dir.name}:\n{r.stderr.strip()}")
        return False
    return True


# ── phase 2: parallel compilation ─────────────────────────────────────────────

def compile_one(run_dir: Path, target: str) -> tuple[Path, str, bool, str]:
    """Compile a single .cu → binary with nvcc. Returns (run_dir, target, ok, stderr).
    Skips compilation if the binary already exists and is newer than the .cu file."""
    cu     = run_dir / f"{target}.cu"
    binary = run_dir / target
    if binary.exists() and binary.stat().st_mtime >= cu.stat().st_mtime:
        return run_dir, target, True, "(cached)"
    cmd = ["nvcc", *NVCC_FLAGS, "-o", str(binary), str(cu)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        binary.chmod(0o755)
    return run_dir, target, r.returncode == 0, r.stderr.strip()


# ── phase 3: run binaries ─────────────────────────────────────────────────────

def run_binary(binary: Path) -> dict | None:
    try:
        r = subprocess.run([str(binary)], capture_output=True, text=True)
    except Exception as e:
        print(f"  [run] {binary.name} ERROR: {e}")
        return None
    if r.returncode != 0:
        print(f"  [run] {binary.name} FAILED: {r.stderr.strip()}")
        return None
    m = OUTPUT_RE.search(r.stdout)
    if not m:
        print(f"  [run] {binary.name} unexpected output: {r.stdout!r}")
        return None
    return dict(
        total_updates=int(m.group(1)),
        time_ms=float(m.group(2)),
        throughput_gups=float(m.group(3)),
        us_per_seq=float(m.group(4)),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    dirs = find_dirs() if args.all else [Path(args.dir).expanduser().resolve()]
    dirs = [d for d in dirs if (d / "hall_of_fame.csv").exists()]
    if not dirs:
        sys.exit("No directories with hall_of_fame.csv found.")

    cached_eqs = load_cache()

    # Build per-dir work lists, deduplicating by equation string across dirs.
    # First directory that contains a given equation string wins; later dirs skip it.
    eq_maps: dict[Path, dict] = {}
    pending: dict[Path, list[str]] = {}   # dir → kernel names that need compiling + running
    seen_eqs = set(cached_eqs)

    for d in dirs:
        eq_map = build_eq_map(d)
        eq_maps[d] = eq_map
        need = []
        for name, (eq, _, _) in eq_map.items():
            if args.force or eq not in seen_eqs:
                need.append(name)
                seen_eqs.add(eq)
        if need:
            pending[d] = need

    if not pending:
        print("All equations cached — nothing to do. Use --force to rerun.")
        return

    total_jobs = sum(len(v) for v in pending.values())
    print(f"Found {total_jobs} kernel(s) to compile across {len(pending)} dir(s)  "
          f"[{args.workers} workers]")

    # Phase 1: generate .cu files for dirs that have pending work (parallel)
    print("\n── Phase 1: generating .cu files ──")
    failed_gen: set[Path] = set()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(gen_cu_files, d): d for d in pending}
        for future in as_completed(futures):
            d = futures[future]
            ok = future.result()
            print(f"  {d.name} {'ok' if ok else 'FAILED'}", flush=True)
            if not ok:
                failed_gen.add(d)
    for d in failed_gen:
        del pending[d]

    # Phase 2: compile all pending targets in parallel
    print(f"\n── Phase 2: compiling {total_jobs} kernel(s) with {args.workers} workers ──")
    compile_jobs = [(d, name) for d, names in pending.items() for name in names]
    compiled_ok: set[tuple[Path, str]] = set()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compile_one, d, name): (d, name) for d, name in compile_jobs}
        done = 0
        for future in as_completed(futures):
            d, name = futures[future]
            _, _, ok, err = future.result()
            done += 1
            status = err if err == "(cached)" else ("ok" if ok else "FAILED")
            print(f"  [{done:>3}/{total_jobs}] {d.name}/{name:<12} {status}"
                  + (f"\n    {err}" if not ok and err != "(cached)" else ""), flush=True)
            if ok:
                compiled_ok.add((d, name))

    # Phase 3: run binaries and collect results
    print("\n── Phase 3: running benchmarks ──")
    all_new: list[dict] = []

    for d in pending:
        names_for_dir = [name for name in pending[d] if (d, name) in compiled_ok]
        if not names_for_dir:
            continue
        print(f"\n  {d.name}")
        eq_map = eq_maps[d]
        for name in names_for_dir:
            binary = d / name
            eq, complexity, loss = eq_map[name]
            print(f"  [run] {name:<12} {eq[:55]:<55}", end=" ", flush=True)
            timing = run_binary(binary)
            if timing is None:
                continue
            print(f"→ {timing['throughput_gups']:.3f} G-ups/s  {timing['us_per_seq']:.4f} us/seq")
            all_new.append({
                "run_dir":        str(d),
                "kernel":         name,
                "equation":       eq,
                "complexity":     complexity,
                "loss":           loss,
                "batch":          args.batch,
                "heads":          args.heads,
                "seq_len":        args.seq,
                "outer":          args.outer,
                "warmup":         args.warmup,
                **timing,
            })
            # Append immediately so a crash doesn't lose results
            chunk = pd.DataFrame([all_new[-1]])
            write_header = not RESULTS.exists()
            chunk.to_csv(RESULTS, mode="a", index=False, header=write_header)
            cached_eqs.add(eq)

    print(f"\nDone. {len(all_new)} new result(s) written to {RESULTS}")


if __name__ == "__main__":
    main()
