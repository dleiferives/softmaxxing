
build-dir dir:
	cp softmax_incremental.csv {{dir}}
	python3 csv_bench.py {{dir}}
	cd {{dir}}; make -j

run-cuda-eval:
	python3 eval_bench.py --all --seq 8128 --outer 500 --heads 128 --workers 12

@ls:
	just --list
