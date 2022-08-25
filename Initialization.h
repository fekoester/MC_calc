//#include <boost/filesystem.hpp>
#include "Data.h"


//------------------------------------------------------------------------------------------------------------------------------------------//
//Creates the folder if not already existing and returns the output file where the data is outputted to
//Number is the argumentList number
void setOutputFile(Data &data){
    data.dataStringNonlinearMemorySavefile = "memoryData";
}


//------------------------------------------------------------------------------------------------------------------------------------------//
//Gets the parsed arguments
bool parsArguments(int argc, char** argv, Data &data){
     for (int i = 0; i < argc; i++) {


	if (string(argv[i]) == "-debug"){
            data.debug = atoi(argv[i+1]);
            if (data.debug)
                cout << "Set debug to " << data.debug << endl;
        }	

	if (string(argv[i]) == "-max_power_nonlinear_memory"){
            data.maxPowerNonlinearMemory = atoi(argv[i+1]);
            if (data.debug)
                cout << "Set maxPowerNonlinearMemory to " << data.maxPowerNonlinearMemory << endl;
        }

	if (string(argv[i]) == "-do_without_threshold"){
            data.doWithoutThreshold = atoi(argv[i+1]);
            if (data.debug)
                cout << "Set doWithoutThreshold to " << data.doWithoutThreshold << endl;
        }

	if (string(argv[i]) == "-threshold"){
            data.threshold = atof(argv[i+1]);
            if (data.debug)
                cout << "Set threshold to " << data.threshold << endl;
        }

	if (string(argv[i]) == "-steps_for_memory"){
            data.stepsForMemory  = atoi(argv[i+1]);
            if (data.debug)
                cout << "Set stepsForMemory to " << data.stepsForMemory << endl;
        }
        
	if (string(argv[i]) == "-regularisation_factor"){
            data.regularisationFactor = atof(argv[i+1]);
            if (data.debug)
                cout << "Set regularisationFactor to " << data.regularisationFactor << endl;
        }

	if (string(argv[i]) == "-regularisation_factor_relative"){
            data.regularisationFactorRelative = atoi(argv[i+1]);
            if (data.debug)
                cout << "Set regularisationFactorRelative to " << data.regularisationFactorRelative << endl;
        }
        
	if (string(argv[i]) == "-start_target_row"){
            data.startTargetRow = atoi(argv[i+1]);
            if (data.debug)
                cout << "Set startTargetRow to " << data.startTargetRow << endl;
        }
        
    }
     return true;
}


//------------------------------------------------------------------------------------------------------------------------------------------//
//Load stat matrix
void loadData(Data &data) {
   data.stateMatrix.load("stateData");
   data.trainingTargetFromInputFile.load("inputData");
   data.trainingTarget = arma::Col<double>(data.stateMatrix.n_rows);
}


//------------------------------------------------------------------------------------------------------------------------------------------//
//Initializes many variables. This has to be seperated from the global variables for some functions to work.
void initializeVariables(int argc, char** argv, Data &data){
	parsArguments(argc, argv, data);
	setOutputFile(data);
	loadData(data);
}
