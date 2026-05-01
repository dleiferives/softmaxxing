#!/usr/bin/env python3
"""
Convert PySR hall-of-fame CSV → CUDA micro-benchmarks per equation + baseline.
CLI:  python cuda_bench.py  <directory-with-csvs>  [options]

Options:
  --batch   INT   batch size            (default 32)
  --heads   INT   number of heads       (default 32)
  --seq     INT   sequence length       (default 2048)
  --outer   INT   timing iterations     (default 200)
  --warmup  INT   warmup iterations     (default 10)

Each benchmark simulates realistic attention softmax computation on GPU:
  - N_SEQS = BATCH * HEADS sequences processed in parallel (one thread per seq)
  - Each sequence has SEQ_LEN logits processed with the incremental online softmax
  - m (running max) is tracked exactly; only s (running sum) uses the SR approximation
  - Throughput reported in Giga-updates/second
"""

import argparse, re
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate CUDA softmax benchmarks from PySR CSV")
parser.add_argument("dir",             help="directory containing hall_of_fame.csv")
parser.add_argument("--batch",  type=int, default=32,   help="batch size (default 32)")
parser.add_argument("--heads",  type=int, default=32,   help="number of attention heads (default 32)")
parser.add_argument("--seq",    type=int, default=2048, help="sequence length (default 2048)")
parser.add_argument("--outer",  type=int, default=200,  help="timing iterations (default 200)")
parser.add_argument("--warmup", type=int, default=10,   help="warmup iterations (default 10)")
args = parser.parse_args()

ROOT     = Path(args.dir).expanduser().resolve()
CSV_FILE = ROOT / "hall_of_fame.csv"
MAKEFILE = ROOT / "Makefile"

BATCH   = args.batch
HEADS   = args.heads
SEQ_LEN = args.seq
OUTER   = args.outer
WARMUP  = args.warmup

N_SEQS = BATCH * HEADS   # sequences processed in parallel


# ── expression conversion ─────────────────────────────────────────────────────

def _add_f_suffix(expr: str) -> str:
    """Append f suffix to bare float/int literals so nvcc treats them as float."""
    # Match numbers not already followed by f/e (scientific handled separately)
    # We do a simple regex: digit sequence with optional decimal point and exponent
    def repl(m):
        s = m.group(0)
        if s.endswith('f'):
            return s
        return s + 'f'
    return re.sub(r'\b\d+(\.\d+)?([eE][+-]?\d+)?\b', repl, expr)

def cudaize(expr: str) -> str:
    expr = re.sub(r'\babs\b',  'fabsf', expr)
    expr = re.sub(r'\bexp\b',  '__expf', expr)
    expr = re.sub(r'\blog\b',  '__logf', expr)
    expr = re.sub(r'\bsqrt\b', 'sqrtf', expr)
    expr = _add_f_suffix(expr)
    return expr


# ── shared CUDA header emitted into every .cu file ───────────────────────────

CUDA_PREAMBLE = f"""\
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>

// Realistic attention dimensions
#define BATCH    {BATCH}
#define HEADS    {HEADS}
#define SEQ_LEN  {SEQ_LEN}
#define N_SEQS   (BATCH * HEADS)
#define OUTER    {OUTER}
#define WARMUP   {WARMUP}

// SR helper macros (match PySR operators)
#define greater(a, b) ((a) > (b) ? 1.0f : 0.0f)
#define neg(a)        (-(a))

// ── logit generation kernel ───────────────────────────────────────────────────
// Layout: logits[step * N_SEQS + seq] — transposed so that at each step i,
// adjacent threads read adjacent addresses (coalesced global memory access).
__global__ void gen_logits(float* __restrict__ logits, unsigned long long seed) {{
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= N_SEQS) return;
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, seq, 0, &rng);
    for (int i = 0; i < SEQ_LEN; i++) {{
        logits[(long long)i * N_SEQS + seq] = curand_normal(&rng) * 2.0f;
    }}
}}
"""

TIMING_MAIN = """\
int main() {
    float* d_logits;
    float* d_out;
    cudaMalloc(&d_logits, (long long)N_SEQS * SEQ_LEN * sizeof(float));
    cudaMalloc(&d_out,    (long long)N_SEQS * sizeof(float));

    // Generate random logits once — same data for all benchmark runs
    {
        int tpb = 256;
        int blk = (N_SEQS + tpb - 1) / tpb;
        gen_logits<<<blk, tpb>>>(d_logits, 42ULL);
        cudaDeviceSynchronize();
    }

    int tpb = 256;
    int blk = (N_SEQS + tpb - 1) / tpb;

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // warm-up
    for (int i = 0; i < WARMUP; i++)
        softmax_kernel<<<blk, tpb>>>(d_logits, d_out, N_SEQS, SEQ_LEN);
    cudaDeviceSynchronize();

    cudaEventRecord(ev_start);
    for (int i = 0; i < OUTER; i++)
        softmax_kernel<<<blk, tpb>>>(d_logits, d_out, N_SEQS, SEQ_LEN);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    long long total_updates = (long long)OUTER * N_SEQS * (SEQ_LEN - 1);
    double gups = (double)total_updates / (ms * 1e-3) / 1e9;
    double us_per_seq = (ms * 1e3) / ((double)OUTER * N_SEQS);

    printf("total_updates=%lld  time=%.2f ms  throughput=%.3f G-updates/s  %.4f us/seq\\n",
           total_updates, ms, gups, us_per_seq);

    // Read back one value so the compiler can't eliminate the kernel
    float sink;
    cudaMemcpy(&sink, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    (void)sink;

    cudaFree(d_logits);
    cudaFree(d_out);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    return 0;
}
"""


# ── kernel templates ──────────────────────────────────────────────────────────

def write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)


def emit_cu(idx: int, eq: str) -> Path:
    cu = ROOT / f"eq_{idx:02d}.cu"
    kernel = f"""\
// SR approximation {idx:02d}: {eq}
// Layout: logits[step * n_seqs + seq] — coalesced reads across warp at each step.
// m is tracked exactly; s is updated via the SR approximation.
__global__ void softmax_kernel(
        const float* __restrict__ logits,
        float* __restrict__ out,
        int n_seqs, int seq_len) {{
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= n_seqs) return;

    float m = __ldg(&logits[seq]);   // step 0, all threads read consecutive addresses
    float s = 1.0f;                  // exp(logits[0] - m) = 1

    #pragma unroll 4
    for (int i = 1; i < seq_len; i++) {{
        float x = __ldg(&logits[(long long)i * n_seqs + seq]);
        s = {cudaize(eq)};
        m = fmaxf(m, x);
    }}
    out[seq] = s;
}}
"""
    write_if_changed(cu, CUDA_PREAMBLE + "\n" + kernel + "\n" + TIMING_MAIN)
    return cu


def emit_baseline() -> Path:
    cu = ROOT / "baseline.cu"
    kernel = """\
// Baseline: standard online softmax (2 __expf calls when x > m, 1 otherwise).
// Layout: logits[step * n_seqs + seq] — coalesced reads across warp at each step.
__global__ void softmax_kernel(
        const float* __restrict__ logits,
        float* __restrict__ out,
        int n_seqs, int seq_len) {
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= n_seqs) return;

    float m = __ldg(&logits[seq]);   // step 0
    float s = 1.0f;

    #pragma unroll 4
    for (int i = 1; i < seq_len; i++) {
        float x    = __ldg(&logits[(long long)i * n_seqs + seq]);
        float m_new = fmaxf(m, x);
        float e_new = __expf(x - m_new);
        s = s * __expf(m - m_new) + e_new;
        m = m_new;
    }
    out[seq] = s;
}
"""
    write_if_changed(cu, CUDA_PREAMBLE + "\n" + kernel + "\n" + TIMING_MAIN)
    return cu


# ── Makefile ──────────────────────────────────────────────────────────────────

def emit_makefile(targets: list[str]) -> None:
    with open(MAKEFILE, "w") as mk:
        mk.write("NVCC      = nvcc\n")
        mk.write("NVCCFLAGS = -O3 -arch=native --use_fast_math -lcurand\n")
        mk.write("all:")
        for t in targets:
            mk.write(f" {t}")
        mk.write("\n\n")
        for t in targets:
            mk.write(f"{t}: {t}.cu\n")
            mk.write("\t$(NVCC) $(NVCCFLAGS) -o $@ $<\n")
        mk.write("\nclean:\n")
        mk.write("\trm -f baseline eq_??")
        mk.write("\n")
        mk.write("\n# Quick comparison: run all benchmarks and collect results\n")
        mk.write("bench: all\n")
        mk.write("\t@echo '--- baseline ---' && ./baseline\n")
        for t in targets:
            if t != "baseline":
                mk.write(f"\t@echo '--- {t} ---' && ./{t}\n")
        mk.write("\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df   = pd.read_csv(CSV_FILE)
    targets = []

    baseline = emit_baseline()
    targets.append(baseline.stem)

    for idx, row in df.iterrows():
        cu = emit_cu(int(idx) + 1, row.Equation)
        targets.append(cu.stem)

    emit_makefile(targets)

    n_eq = len(targets) - 1
    print(f"Generated {len(targets)} CUDA benchmarks in {ROOT}")
    print(f"  BATCH={BATCH}  HEADS={HEADS}  SEQ_LEN={SEQ_LEN}  N_SEQS={N_SEQS}")
    print(f"  baseline.cu  : standard online softmax (2x expf per update)")
    print(f"  eq_*.cu      : {n_eq} SR approximations")
    print(f"Compile:  cd {ROOT} && make -j")
    print(f"Compare:  make bench")


if __name__ == "__main__":
    main()
