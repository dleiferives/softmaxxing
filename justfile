
build-dir dir:
	python3 csv_bench.py {{dir}}
	cd {{dir}}; make -j
