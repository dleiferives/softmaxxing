
build-dir dir:
	cp softmax_incremental.csv {{dir}}
	python3 csv_bench.py {{dir}}
	cd {{dir}}; make -j
