//#include <boost/filesystem.hpp>
#include "Data.h"
#include <boost/math/distributions/chi_squared.hpp>

using boost::math::chi_squared;

//Definitions
//------------------------------------------------------------------------------------------------------------------------------------------//


//Define legendre polynomials
double legendreFunction(int n, double x){
    if (n == 0)
        return 1;
    else if (n == 1)
        return x;
    else 
        return ((2.0*n - 1.0)*x*legendreFunction(n-1, x) - (n -1.0)*legendreFunction(n-2, x))/n;
}

//Returns 1.0 -  the cumulate distribution function, i.e. it returns the probability that "chi_squared" of "degree N" rolls "x" greater than "value". 
double getCDFChiSquared(int degrees, double value)
{
	chi_squared dist(degrees);
	return 1.0 - cdf(dist, value);
}

//Find the threshold for which the 1.0 - "CDF of chi_squared" is less than "probability"
double findThreshold(Data &data, double probability)
{
	double precision = 1.0;
	double value = 0.0;
	while ( precision > 10e-6)
	{
		value += precision;
		if ( getCDFChiSquared(data.stateMatrix.n_cols, value) < probability)
		{
			value -= precision;
			precision /= 10.0;
		}
	}

	return value*4.0/data.stateMatrix.n_rows ;

}

//Returns a matrix of ints, yieldin every combination possible for k legendre polynomials distributed over power n
vector<vector<int>> getCombinations(int n, int k){
    if (k == 1){
        return {{n}};
    }
    if (k == 2){
        vector<vector<int>> tmp = vector<vector<int>>(n - 1, vector<int>(2));
        for (int i = 0; i < n-1; i++){
            tmp[i][0] = n - i - 1;
            tmp[i][1] = i+1;
        }
        return tmp;
    }
    else{
        vector<vector<int>> tmp;
        vector<vector<int>> tmp2;
        vector<int> tmp3;
        for (int i = 1; i <= n - (k - 1); i++){
            tmp2 = getCombinations(n-i,k-1);
            tmp3 = {i};
            for (unsigned int j = 0; j < tmp2.size(); j++){
                tmp2[j].insert(tmp2[j].begin(),tmp3.begin(), tmp3.end());
            }
            tmp.insert(tmp.end(), tmp2.begin(), tmp2.end());
        }
        return tmp;
    }
}

//Does a loop over a combination of legendre polynomials.
//This is a recursiv function.
//Rather complicated with the break condition...maybe there is a better way?
void doLoopForOneLegendrePolynomial(Data &data, vector<int> &powerList, ofstream &dataFile, int size, vector<int> &steps, arma::Mat<double> &ssti, int startStepBefore){
    unsigned int startStep = startStepBefore + 1;
    for (unsigned int i = startStep; i < data.stepsForMemory; i++){
        if (size > 1){
            
            for (unsigned int j = size - 1; j > 0; j--){
                data.numberLowMemAccLastPeak[powerList.size() - j] = 0;
            }
            
            if (data.lowMemLastLayerQuit >= size){
                if (data.lowMemLastLayerQuit == size){
                    data.lowMemLastLayerQuit = -1;
                }
                break;
            }
            
            if (i >= startStep + data.numberLowMemLast + data.numberLowMemAccLastPeak[powerList.size() - size] + 1)
                data.lowMemLastLayer = size;
            
            steps[powerList.size() - size] = i;
            data.startSteps[powerList.size() - size] = startStep;
            doLoopForOneLegendrePolynomial(data, powerList, dataFile, size - 1, steps, ssti, i);
            
        }
        else{

            if (i >= startStep + data.numberLowMemLast + 1)
                data.lowMemLastLayer = size;
            
            steps[powerList.size() - size] = i;
            data.startSteps[powerList.size() - size] = startStep;
            for (unsigned int k = 0; k < data.stateMatrix.n_rows; k++){
                data.trainingTarget(k) = 1.0;
                for (unsigned int j = 0; j < powerList.size(); j++){
                    data.trainingTarget(k) *= legendreFunction(powerList[j],data.trainingTargetFromInputFile[k+data.startTargetRow-steps[j]]);
                }
            }


            data.trainingTarget -= mean(data.trainingTarget);
            //TRAINING_TARGET /= stddev(TRAINING_TARGET);
 
            data.pMatrix = data.stateMatrix.t()*data.trainingTarget;
            data.ghsMatrix= data.pMatrix.t()*ssti*data.pMatrix/(norm(data.trainingTarget)*norm(data.trainingTarget));

            if (data.ghsMatrix(0) > data.threshold || data.doWithoutThreshold){
                for (unsigned int j = 1; j < data.numberLowMemAccLastPeak.size() + 1; j++){
                    data.numberLowMemAccLastPeak[powerList.size() - j] = steps[powerList.size() - j] - data.startSteps[powerList.size() - j];
                }
                data.totalMemCapDegrees[data.momDegree] += data.ghsMatrix(0);
                if (powerList.size() == 1){
                    data.totalMemCapDegreesPure[data.momDegree] += data.ghsMatrix(0);
                }
                for (unsigned int j = 0; j < steps.size(); j++){
                    dataFile << steps[j] << " ";
                }
                for (unsigned int j = 1; j <= data.maxPowerNonlinearMemory - steps.size(); j++){
                    dataFile << "0 ";
                }
                for (unsigned int j = 0; j < powerList.size(); j++){
                    dataFile << powerList[j] << " ";
                }
                for (unsigned int j = 1; j <= data.maxPowerNonlinearMemory - steps.size(); j++){
                    dataFile << "0 ";
                }
                dataFile << data.ghsMatrix(0) << " ";
                for (unsigned int j = 1; j <= data.maxPowerNonlinearMemory; j++){
                    dataFile << data.totalMemCapDegrees[j] << " ";
                }
                for (unsigned int j = 1; j <= data.maxPowerNonlinearMemory; j++){
                    dataFile << data.totalMemCapDegreesPure[j] << " ";
                }
                dataFile << endl;
            }
            else{
                if (i >= startStep + data.numberLowMemLast + data.numberLowMemAccLastPeak[powerList.size() - size]){
                    data.lowMemLastLayerQuit = data.lowMemLastLayer;
                    break;
                }
            }
            
        }
    }
}


//Wrapper function to call everything needed to calculate the memory capacity
void calcErrors(Data &data, vector<int> &powerList, ofstream &dataFile, arma::Mat<double> &ssti){
    vector<int> steps = vector<int>(powerList.size());
    data.startSteps = vector<int>(powerList.size());
    data.numberLowMemAccLastPeak = vector<int>(powerList.size(), 0);
    data.lowMemLastLayer = powerList.size();
    doLoopForOneLegendrePolynomial(data,powerList, dataFile, powerList.size(), steps, ssti, -1);
}

//Calculate the nonlinear memory capacity the system
void calcNonlinearMemory(Data &data){

    data.threshold = findThreshold(data, 10e-6);

    data.totalMemoryCap = 0.0;
    for (unsigned int i = 0; i < data.stateMatrix.n_cols; i++){
        data.stateMatrix.col(i) -= mean(data.stateMatrix.col(i));
        //data.stateMatrix.col(i) /= stddev(data.stateMatrix.col(i));
    }
    
    arma::Mat<double> sst= data.stateMatrix.t()*data.stateMatrix; 
    if (data.regularisationFactorRelative){
		data.regularisationFactor = sst.max()*data.regularisationFactor;
		sst += data.regularisationFactor*arma::eye<arma::mat>(data.stateMatrix.n_cols, data.stateMatrix.n_cols);
    }
    else{
    	sst += data.regularisationFactor*arma::eye<arma::mat>(data.stateMatrix.n_cols, data.stateMatrix.n_cols);
    }

    sst = arma::pinv(sst);

    ofstream data_file_nonlinear_memory;
    data_file_nonlinear_memory.open(data.dataStringNonlinearMemorySavefile);
    data_file_nonlinear_memory << "STEP DEGREE CAPACITY" << endl;

    //Looping over every degree
    for (unsigned int power = 1; power <= data.maxPowerNonlinearMemory; power++){
        data.momDegree = power;
        if (data.debug)
			cout << "Power: " << power << endl;
		
        //Looping over the number of legendre functions
        for (unsigned int variables = 1; variables <= power; variables++){
			if (data.debug)
				cout << "Variables: " << variables << endl;
            
            //Init the powerLists
            vector<vector<int>> powerLists =  getCombinations(power, variables);

            //Go through every combination and calculate the mem capacity
            for (unsigned int combination = 0; combination < powerLists.size(); combination++){
                calcErrors(data,powerLists[combination],data_file_nonlinear_memory , sst);
            }
        }
    }

    data_file_nonlinear_memory.close();
}
