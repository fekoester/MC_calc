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
	
	string dataStringNonlinearMemorySavefile = "memoryData";
	string stateFile = "stateData";
	string inputFile = "inputData";
	int outputPrecision = 6;            // digits written for capacities (6 = std::ostream default, as in v0.1)
	bool thresholdGiven = false;        // -threshold given: use it instead of the chi-squared threshold
	double thresholdProbability = 1e-5; // v0.1 passed 10e-6 (= 1e-5)
	double thresholdFactor = 4.0;       // v0.1: threshold = 4 * chi2 quantile / T
	
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
	int numberLowMemLast = 20;          // early stop after this many sub-threshold delays (-patience; 0 = full scan)
	
	arma::Col<double> trainingTarget;
	arma::Col<double> trainingTargetFromInputFile;
	
	int startTargetRow = 1000;
	
	arma::Mat<double> pMatrix;
	arma::Mat<double> ghsMatrix;
	
	vector<double> totalMemCapDegrees;      // sized maxPower + 1 after parsing
	vector<double> totalMemCapDegreesPure;
	arma::Mat<double> legendreTable;        // legendreTable(n, t) = P_n(input[t]), filled once

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
