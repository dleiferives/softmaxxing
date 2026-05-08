
cuda-direct dir py-args="":
	cp softmax_incremental.csv {{dir}}
	python3 cuda_direct_bench.py {{dir}} {{py-args}}
	cd {{dir}}; make -j; make bench


cpu-direct dir py-args="":
	python3 cpu_direct_bench.py {{dir}} {{py-args}}
	cd {{dir}}; make -j; make bench

cpu-vec dir:
	python3 cpu_direct_bench.py {{dir}}
	cd {{dir}}; bash vec_report.sh 2>&1 | grep -E '(===|optimized|missed.*vectorized)' | head -60

run-cpu-eval:
	python3 eval_cpu_bench.py --workers 8

run-cuda-eval:
	python3 eval_bench.py --all --seq 8128 --outer 500 --heads 128 --workers 12

evolve:
	cd evolve/assmcmcbly && make -j && ./assmcmcbly

evolve-build:
	cd evolve/assmcmcbly && make -j

evolve-clean:
	cd evolve/assmcmcbly && make clean

@ls:
	just --list
