CXX      := g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -I src
DEBUG    := -g
TARGET   := assmcmcbly

USE_JIT ?= 1
ifeq ($(USE_JIT),1)
  CXXFLAGS += -DUSE_JIT
endif

# ── Library objects (everything except main) ──────────────────────────────────
LIB_SRCS := src/execute.cpp src/jit.cpp src/fitness.cpp src/dag.cpp \
            src/random.cpp src/mutate.cpp src/hardening.cpp \
            src/population.cpp src/print.cpp
LIB_OBJS := $(LIB_SRCS:.cpp=.o)

# ── Main binary ───────────────────────────────────────────────────────────────
$(TARGET): main.o $(LIB_OBJS)
	$(CXX) $(DEBUG) $(CXXFLAGS) -o $@ $^

# ── Tests ─────────────────────────────────────────────────────────────────────
TEST_SRCS := $(wildcard tests/*.cpp)
TEST_BINS := $(TEST_SRCS:.cpp=)

tests/%: tests/%.cpp $(LIB_OBJS)
	$(CXX) $(DEBUG) $(CXXFLAGS) -o $@ $^

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do \
	  echo "=== $$t ==="; \
	  $$t && echo PASS || { echo FAIL; exit 1; }; \
	done

# ── Generic compile ───────────────────────────────────────────────────────────
%.o: %.cpp
	$(CXX) $(DEBUG) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f main.o $(LIB_OBJS) $(TARGET) $(TEST_BINS)

.PHONY: test clean
