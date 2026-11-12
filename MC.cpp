#ifdef _WIN32
#define _CRT_SECURE_NO_DEPRECATE
#endif

//------------------------------------------------------------------------//
//Example program for calculating the MC of a given state matrix
//------------------------------------------------------------------------//

#include "Memory.h"
#include "Initialization.h"


void runMemoryCalculation(int argc, char** argv){
    //Start the simulation

	//Create Data Struct
	//------------------------------------//
	Data data;
	
	//Initialize the variables for this run
	//------------------------------------//
	initializeVariables(argc, argv, data);

	//Run the simulation and calculate the error
	//------------------------------------//
	 calcNonlinearMemory(data);
	

}


//Main
//------------------------------------------------------------------------------------------------------------------------------------------//
int main(int argc, char** argv)
{
    try {
        runMemoryCalculation(argc,argv);
    } catch (const std::exception &e) {
        cerr << "error: " << e.what() << endl;
        return 1;
    }
    cout << "Done...\n";
	return 0;
}
