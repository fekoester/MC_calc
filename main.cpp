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
	
	cout <<"\n\n\n";
}


//Main
//------------------------------------------------------------------------------------------------------------------------------------------//
int main(int argc, char** argv)
{  
    //Create the outputFile and corresponding folder for the day 
    runMemoryCalculation(argc,argv);

    cout << "Done...\n";
    
	return 0;
}
