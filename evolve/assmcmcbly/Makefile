CXX      := g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra
TARGET   := assmcmcbly
DEBUG    := -g

SRCS := execute.cpp fitness.cpp random.cpp dag.cpp mutate.cpp hardening.cpp population.cpp print.cpp main.cpp
OBJS := $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CXX) $(DEBUG) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(DEBUG) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean
