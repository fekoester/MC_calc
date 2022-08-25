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
