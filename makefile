CC = g++
CFLAGS = -Wall
CXXFLAGS:=-std=c++17 -O2 

.PHONY: all MC clean

MC: MC.cpp
	$(CC) $(CFLAGS) $(CXXFLAGS) MC.cpp -o MC -larmadillo -llapack -lblas

clean:
	rm -f MC
