CXXFLAGS = -O3 -march=native -std=c++17
all: baseline eq_001 eq_002 eq_003 eq_004 eq_005 eq_006 eq_007 eq_008

baseline: baseline.cpp data.h
	$(CXX) $(CXXFLAGS) -o $@ $<
eq_001: eq_001.cpp data.h
	$(CXX) $(CXXFLAGS) -o $@ $<
eq_002: eq_002.cpp data.h
	$(CXX) $(CXXFLAGS) -o $@ $<
eq_003: eq_003.cpp data.h
	$(CXX) $(CXXFLAGS) -o $@ $<
eq_004: eq_004.cpp data.h
	$(CXX) $(CXXFLAGS) -o $@ $<
eq_005: eq_005.cpp data.h
	$(CXX) $(CXXFLAGS) -o $@ $<
eq_006: eq_006.cpp data.h
	$(CXX) $(CXXFLAGS) -o $@ $<
eq_007: eq_007.cpp data.h
	$(CXX) $(CXXFLAGS) -o $@ $<
eq_008: eq_008.cpp data.h
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f baseline eq_001 eq_002 eq_003 eq_004 eq_005 eq_006 eq_007 eq_008 data.h
