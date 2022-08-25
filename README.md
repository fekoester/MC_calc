# MC calculation program
Program to calculate the memory capacity of a given reservoir computer state.

## Tutorial

### Build Manually

You will need Armadillo and Boost to compile the program. 
Armadillo is a C++ linear algebra package, which can be found here (http://arma.sourceforge.net/).
Boost is a C++ library, which can be found here (https://www.boost.org/).
Otherwise the program uses basic C++ libraries, thus it should compile on every system.

After compiling, start the program with ./nameOfProgram.
You need to have a file called "StateData" in the folder that contains the state matrix of your reservoir and a file called "inputData" that contains the random values between -1 and 1 that are the inputs of the reservoir.

The output file "memoryData" contains the capacities in following order.
The first N columns are the steps into the past, the second N columns are the order of legendre polynomials used, the next columns gives the capacity, the next N columns give the summed memory capacities of order 1 up to N and the last columns the pure memory capacities of order 1 up to N.
