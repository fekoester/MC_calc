//------------------------------------------------------------------------------------------------------------------------------------------//
//This header includes the Data.
//------------------------------------------------------------------------------------------------------------------------------------------//
#pragma once
#include <string>
#include <armadillo>

using namespace std;

struct Data{  
	//Member variables
	//------------------------------------------------------------------------------------------------------------------------------------------//
	arma::Mat<double> stateMatrix;
	
	string dataStringNonlinearMemorySavefile = "./data/memory"; 
	
	unsigned int maxPowerNonlinearMemory= 3;
	int momDegree = 0;
	bool doWithoutThreshold = false;
	double threshold = 0.0;
	unsigned int stepsForMemory = 50;
	double regularisationFactor = 1E-6;
	bool regularisationFactorRelative = false;
	double totalMemoryCap = 0.0;
	
	vector<int> startSteps;
	vector<int> numberLowMemAccLastPeak;
	int lowMemLastLayer;
	int lowMemLastLayerQuit = -1;
	int numberLowMemLast = 20;
	
	arma::Col<double> trainingTarget;
	arma::Col<double> trainingTargetFromInputFile;
	
	int startTargetRow = 1000;
	
	arma::Mat<double> pMatrix;
	arma::Mat<double> ghsMatrix;
	
	vector<double> totalMemCapDegrees = vector<double>(15, 0.0);
	vector<double> totalMemCapDegreesPure = vector<double>(15, 0.0);

	bool debug = false;

	//Member methods
	//------------------------------------------------------------------------------------------------------------------------------------------//

	//Constructor
	Data();
    
};



//Constructor
Data::Data()
{
}
