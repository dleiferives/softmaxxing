CXX      := g++
CXXFLAGS := -O2 -std=c++17 -Wall

# point at any candidate: make SRC=../../runs/.../12345.cpp
SRC  ?= ../../runs/20260502_234503/29085.cpp
STEM := $(notdir $(basename $(SRC)))

BIN := sweep_$(STEM)
CSV := $(STEM).csv
PNG := $(STEM)_plot.png

.PHONY: all run plot clean

all: $(BIN)

$(BIN): main.cpp
	cp $(SRC) .
	$(CXX) $(CXXFLAGS) -DTARGET_CPP='"$(STEM).cpp"' -o $@ main.cpp -lm

run: $(BIN)
	./$(BIN) > $(CSV)

plot: $(CSV)
	python3 plot.py $(CSV)

clean:
	rm -f sweep_* *.csv *.png $(basename $SRC)
