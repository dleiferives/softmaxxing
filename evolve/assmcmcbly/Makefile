CXX      := g++
BASE_FLAGS := -std=c++17 -Wall -Wextra -I src
TARGET   := assmcmcbly

USE_JIT ?= 1
ifeq ($(USE_JIT),1)
  BASE_FLAGS += -DUSE_JIT
endif

# ── Build modes ───────────────────────────────────────────────────────────────
# Default: release (make / make release)
# Debug:   make debug          — no opt, ASAN + UBSan, output assmcmcbly-dbg
# Profile: make profile        — O2 + debug info + gprof, output assmcmcbly-prof

RELEASE_FLAGS := -O2 -g $(BASE_FLAGS)
DEBUG_FLAGS   := -O0 -g $(BASE_FLAGS) -fsanitize=address,undefined -fno-omit-frame-pointer
PROFILE_FLAGS := -O2 -g -pg $(BASE_FLAGS)

# ── Library sources ───────────────────────────────────────────────────────────
LIB_SRCS := src/problem.cpp src/execute.cpp src/jit.cpp src/fitness.cpp src/dag.cpp \
            src/random.cpp src/mutate.cpp src/hardening.cpp \
            src/population.cpp src/print.cpp src/innovation.cpp src/speciation.cpp \
            src/codegen.cpp
LIB_OBJS         := $(LIB_SRCS:.cpp=.o)
LIB_OBJS_DBG     := $(LIB_SRCS:.cpp=.dbg.o)
LIB_OBJS_PROF    := $(LIB_SRCS:.cpp=.prof.o)
DEPS             := main.d $(LIB_SRCS:.cpp=.d)
DEPS_DBG         := main.dbg.d $(LIB_SRCS:.cpp=.dbg.d)
DEPS_PROF        := main.prof.d $(LIB_SRCS:.cpp=.prof.d)

# ── Targets ───────────────────────────────────────────────────────────────────
.PHONY: all release debug profile test clean

all: release

release: $(TARGET)
$(TARGET): main.o $(LIB_OBJS)
	$(CXX) $(RELEASE_FLAGS) -o $@ $^

debug: $(TARGET)-dbg
$(TARGET)-dbg: main.dbg.o $(LIB_OBJS_DBG)
	$(CXX) $(DEBUG_FLAGS) -o $@ $^

profile: $(TARGET)-prof
$(TARGET)-prof: main.prof.o $(LIB_OBJS_PROF)
	$(CXX) $(PROFILE_FLAGS) -o $@ $^

# ── Tests ─────────────────────────────────────────────────────────────────────
TEST_SRCS := $(wildcard tests/*.cpp)
TEST_BINS := $(TEST_SRCS:.cpp=)
TEST_DEPS := $(TEST_SRCS:.cpp=.d)

tests/%: tests/%.cpp $(LIB_OBJS)
	$(CXX) $(RELEASE_FLAGS) -MMD -MP -o $@ $^

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do \
	  echo "=== $$t ==="; \
	  $$t && echo PASS || { echo FAIL; exit 1; }; \
	done

# ── Compile rules ─────────────────────────────────────────────────────────────
%.o: %.cpp
	$(CXX) $(RELEASE_FLAGS) -MMD -MP -c -o $@ $<

%.dbg.o: %.cpp
	$(CXX) $(DEBUG_FLAGS) -MMD -MP -c -o $@ $<

%.prof.o: %.cpp
	$(CXX) $(PROFILE_FLAGS) -MMD -MP -c -o $@ $<

-include $(DEPS) $(DEPS_DBG) $(DEPS_PROF) $(TEST_DEPS)

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f main.o main.dbg.o main.prof.o \
	      $(LIB_OBJS) $(LIB_OBJS_DBG) $(LIB_OBJS_PROF) \
	      $(TARGET) $(TARGET)-dbg $(TARGET)-prof \
	      $(TEST_BINS) \
	      $(DEPS) $(DEPS_DBG) $(DEPS_PROF) $(TEST_DEPS)
