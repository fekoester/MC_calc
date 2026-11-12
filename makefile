CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall
LDLIBS = -larmadillo -llapack -lblas

.PHONY: all clean

all: MC

MC: MC.cpp Data.h Initialization.h Memory.h
	$(CXX) $(CXXFLAGS) MC.cpp -o MC $(LDLIBS)

clean:
	rm -f MC
