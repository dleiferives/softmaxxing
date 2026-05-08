
# ── GPU ───────────────────────────────────────────────────────────────────────

cuda-direct dir py-args="":
	cp softmax_incremental.csv {{dir}}
	python3 gpu/cuda_direct_bench.py {{dir}} {{py-args}}
	cd {{dir}}; make -j; make bench

run-gpu-eval:
	python3 gpu/eval_bench.py --all --seq 8128 --outer 500 --heads 128 --workers 12

run-gpu-analyze:
	python3 gpu/analyze_bench.py

# Full GPU loop: bench all outputs → analyze → (re-run SR with updated features)
gpu-loop:
	just run-gpu-eval
	just run-gpu-analyze


# ── CPU ───────────────────────────────────────────────────────────────────────

# Quick bench a directory of equations (for manual exploration / spot-checks)
cpu-direct dir py-args="":
	python3 cpu/cpu_direct_bench.py {{dir}} {{py-args}}
	cd {{dir}}; make -j; make bench

cpu-vec dir:
	python3 cpu/cpu_direct_bench.py {{dir}}
	cd {{dir}}; bash vec_report.sh 2>&1 | grep -E '(===|optimized|missed.*vectorized)' | head -60

# Benchmark every equation from all outputs/ HOF CSVs
run-cpu-eval:
	python3 cpu/eval_cpu_direct.py --workers 8

# Correlate CPU bench results with equation features → cpu/cpu_direct_features.csv
run-cpu-analyze:
	python3 cpu/analyze_cpu_direct.py

# Full CPU loop: bench all outputs → analyze correlations
cpu-loop:
	just run-cpu-eval
	just run-cpu-analyze


# ── SR ────────────────────────────────────────────────────────────────────────

sr-base:
	julia run_sr.jl

sr-gpu:
	julia run_sr_tree_loss.jl

sr-cpu:
	julia run_sr_cpu.jl


# ── Misc ──────────────────────────────────────────────────────────────────────

evolve:
	cd evolve/assmcmcbly && make -j && ./assmcmcbly

evolve-build:
	cd evolve/assmcmcbly && make -j

evolve-clean:
	cd evolve/assmcmcbly && make clean

@ls:
	just --list
