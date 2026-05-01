#!/usr/bin/env python3
"""
Convert PySR hall-of-fame CSV → CUDA micro-benchmarks that directly compare
algorithm speeds on randomly generated (m, s, x) triples.

CLI:  python cuda_direct_bench.py  <directory-with-csvs>  [options]

Options:
  --rows    INT   number of (m, s, x) triples   (default 1048576)
  --rpt     INT   rows per thread for Nt mode    (default 16)
  --outer   INT   timing iterations              (default 1000)
  --warmup  INT   warmup iterations              (default 50)

Generates two binaries per equation:
  eq_XX_1t  — one row per thread
  eq_XX_Nt  — RPT rows per thread (each thread loops over a chunk)
"""

import argparse, re
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate direct CUDA benchmarks from PySR CSV")
parser.add_argument("dir",              help="directory containing hall_of_fame.csv")
parser.add_argument("--rows",   type=int, default=1<<20, help="number of data rows (default 1048576)")
parser.add_argument("--rpt",    type=int, default=16,    help="rows per thread for Nt mode (default 16)")
parser.add_argument("--outer",  type=int, default=1_000, help="timing iterations (default 1000)")
parser.add_argument("--warmup", type=int, default=50,    help="warmup iterations (default 50)")
args = parser.parse_args()

ROOT     = Path(args.dir).expanduser().resolve()
CSV_FILE = ROOT / "hall_of_fame.csv"
MAKEFILE = ROOT / "Makefile"

N_ROWS = args.rows
RPT    = args.rpt
OUTER  = args.outer
WARMUP = args.warmup


# ── expression conversion ─────────────────────────────────────────────────────

def _add_f_suffix(expr: str) -> str:
    def repl(m):
        s = m.group(0)
        if s.endswith('f'):
            return s
        if '.' not in s and 'e' not in s.lower():
            return s + '.0f'
        return s + 'f'
    return re.sub(r'\b\d+(\.\d+)?([eE][+-]?\d+)?\b', repl, expr)

def cudaize(expr: str) -> str:
    expr = re.sub(r'\babs\b',  'fabsf', expr)
    expr = re.sub(r'\bexp\b',  '__expf', expr)
    expr = re.sub(r'\blog\b',  '__logf', expr)
    expr = re.sub(r'\bsqrt\b', 'sqrtf', expr)
    expr = _add_f_suffix(expr)
    return expr


# ── shared preamble ───────────────────────────────────────────────────────────

def preamble() -> str:
    return f"""\
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>

#define N_ROWS {N_ROWS}
#define RPT    {RPT}
#define OUTER  {OUTER}
#define WARMUP {WARMUP}

#define greater(a, b) ((a) > (b) ? 1.0f : 0.0f)
#define neg(a)        (-(a))

__global__ void gen_data(float* __restrict__ d_m,
                         float* __restrict__ d_s,
                         float* __restrict__ d_x,
                         int n, unsigned long long seed) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, i, 0, &rng);
    d_m[i] = curand_uniform(&rng) * 3.0f;
    d_s[i] = curand_uniform(&rng) * 4.5f + 0.5f;
    d_x[i] = curand_normal(&rng) * 2.0f;
}}
"""


# ── timing main (1 row per thread) ───────────────────────────────────────────

TIMING_MAIN_1T = """\
int main() {
    float *d_m, *d_s, *d_x, *d_out;
    cudaMalloc(&d_m,   N_ROWS * sizeof(float));
    cudaMalloc(&d_s,   N_ROWS * sizeof(float));
    cudaMalloc(&d_x,   N_ROWS * sizeof(float));
    cudaMalloc(&d_out, N_ROWS * sizeof(float));

    int tpb = 256;
    int blk = (N_ROWS + tpb - 1) / tpb;

    gen_data<<<blk, tpb>>>(d_m, d_s, d_x, N_ROWS, 42ULL);
    cudaDeviceSynchronize();

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (int i = 0; i < WARMUP; i++)
        bench_kernel<<<blk, tpb>>>(d_m, d_s, d_x, d_out, N_ROWS);
    cudaDeviceSynchronize();

    cudaEventRecord(ev_start);
    for (int i = 0; i < OUTER; i++)
        bench_kernel<<<blk, tpb>>>(d_m, d_s, d_x, d_out, N_ROWS);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    long long total_calls = (long long)OUTER * N_ROWS;
    double gcps      = (double)total_calls / (ms * 1e-3) / 1e9;
    double us_per_1k = (ms * 1e3) / (total_calls / 1000.0);

    printf("1t  total_calls=%lld  time=%.2f ms  throughput=%.3f G-calls/s  %.4f us/1k calls\\n",
           total_calls, ms, gcps, us_per_1k);

    float sink;
    cudaMemcpy(&sink, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    (void)sink;

    cudaFree(d_m); cudaFree(d_s); cudaFree(d_x); cudaFree(d_out);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    return 0;
}
"""


# ── timing main (N rows per thread) ──────────────────────────────────────────

TIMING_MAIN_NT = """\
int main() {
    float *d_m, *d_s, *d_x, *d_out;
    cudaMalloc(&d_m,   N_ROWS * sizeof(float));
    cudaMalloc(&d_s,   N_ROWS * sizeof(float));
    cudaMalloc(&d_x,   N_ROWS * sizeof(float));
    cudaMalloc(&d_out, N_ROWS * sizeof(float));

    // gen_data uses 1 thread/row regardless of bench mode
    int gen_tpb = 256;
    int gen_blk = (N_ROWS + gen_tpb - 1) / gen_tpb;
    gen_data<<<gen_blk, gen_tpb>>>(d_m, d_s, d_x, N_ROWS, 42ULL);
    cudaDeviceSynchronize();

    int tpb     = 256;
    int n_threads = (N_ROWS + RPT - 1) / RPT;
    int blk     = (n_threads + tpb - 1) / tpb;

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (int i = 0; i < WARMUP; i++)
        bench_kernel<<<blk, tpb>>>(d_m, d_s, d_x, d_out, N_ROWS);
    cudaDeviceSynchronize();

    cudaEventRecord(ev_start);
    for (int i = 0; i < OUTER; i++)
        bench_kernel<<<blk, tpb>>>(d_m, d_s, d_x, d_out, N_ROWS);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    long long total_calls = (long long)OUTER * N_ROWS;
    double gcps      = (double)total_calls / (ms * 1e-3) / 1e9;
    double us_per_1k = (ms * 1e3) / (total_calls / 1000.0);

    printf("Nt  total_calls=%lld  time=%.2f ms  throughput=%.3f G-calls/s  %.4f us/1k calls\\n",
           total_calls, ms, gcps, us_per_1k);

    float sink;
    cudaMemcpy(&sink, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    (void)sink;

    cudaFree(d_m); cudaFree(d_s); cudaFree(d_x); cudaFree(d_out);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    return 0;
}
"""


# ── kernel emitters ───────────────────────────────────────────────────────────

def write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)


def emit_1t(stem: str, kernel_body: str) -> Path:
    cu = ROOT / f"{stem}_1t.cu"
    kernel = f"""\
__global__ void bench_kernel(const float* __restrict__ d_m,
                             const float* __restrict__ d_s,
                             const float* __restrict__ d_x,
                             float* __restrict__ out, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float m = d_m[i];
    float s = d_s[i];
    float x = d_x[i];
    out[i] = {kernel_body};
}}
"""
    write_if_changed(cu, preamble() + "\n" + kernel + "\n" + TIMING_MAIN_1T)
    return cu


def emit_nt(stem: str, kernel_body: str) -> Path:
    cu = ROOT / f"{stem}_Nt.cu"
    kernel = f"""\
__global__ void bench_kernel(const float* __restrict__ d_m,
                             const float* __restrict__ d_s,
                             const float* __restrict__ d_x,
                             float* __restrict__ out, int n) {{
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * RPT;
    float acc = 0.0f;
    for (int k = 0; k < RPT; k++) {{
        int i = base + k;
        if (i >= n) break;
        float m = d_m[i];
        float s = d_s[i];
        float x = d_x[i];
        acc = {kernel_body};
    }}
    int out_i = base / RPT;
    if (out_i < (n + RPT - 1) / RPT) out[out_i] = acc;
}}
"""
    write_if_changed(cu, preamble() + "\n" + kernel + "\n" + TIMING_MAIN_NT)
    return cu


def emit_eq(idx: int, eq: str) -> tuple[Path, Path]:
    stem = f"eq_{idx:02d}"
    body = cudaize(eq)
    return emit_1t(stem, body), emit_nt(stem, body)


def emit_baseline() -> tuple[Path, Path]:
    body = "s * __expf(m - fmaxf(m, x)) + __expf(x - fmaxf(m, x))"
    return emit_1t("baseline", body), emit_nt("baseline", body)


# ── Makefile ──────────────────────────────────────────────────────────────────

def emit_makefile(stems: list[str]) -> None:
    targets_1t = [f"{s}_1t" for s in stems]
    targets_nt = [f"{s}_Nt" for s in stems]
    all_targets = targets_1t + targets_nt

    with open(MAKEFILE, "w") as mk:
        mk.write("NVCC      = nvcc\n")
        mk.write("NVCCFLAGS = -O3 -arch=native --use_fast_math -lcurand\n")
        mk.write("all: " + " ".join(all_targets) + "\n\n")

        for t in all_targets:
            mk.write(f"{t}: {t}.cu\n")
            mk.write(f"\t$(NVCC) $(NVCCFLAGS) -o $@ $<\n")

        mk.write("\nclean:\n")
        mk.write("\trm -f " + " ".join(all_targets) + "\n")

        mk.write("\n# Side-by-side comparison: 1t vs Nt for each equation\n")
        mk.write("bench: all\n")
        for s in stems:
            mk.write(f"\t@echo '--- {s} ---'\n")
            mk.write(f"\t@./{s}_1t\n")
            mk.write(f"\t@./{s}_Nt\n")
        mk.write("\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(CSV_FILE)
    stems = ["baseline"]

    emit_baseline()

    for idx, row in df.iterrows():
        stem = f"eq_{int(idx)+1:02d}"
        body = cudaize(row.Equation)
        emit_1t(stem, body)
        emit_nt(stem, body)
        stems.append(stem)

    emit_makefile(stems)

    print(f"Generated {len(stems)*2} CUDA binaries in {ROOT}")
    print(f"  N_ROWS={N_ROWS}  RPT={RPT}  OUTER={OUTER}  WARMUP={WARMUP}")
    print(f"  *_1t : one row per thread")
    print(f"  *_Nt : {RPT} rows per thread")
    print(f"Compile:  cd {ROOT} && make -j")
    print(f"Compare:  make bench")


if __name__ == "__main__":
    main()
