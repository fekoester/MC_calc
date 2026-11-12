//#include <boost/filesystem.hpp>
#include "Data.h"
#include <boost/math/distributions/chi_squared.hpp>

using boost::math::chi_squared;

//Definitions
//------------------------------------------------------------------------------------------------------------------------------------------//


//Legendre polynomial P_n(x) by the three-term recurrence (iterative; the same values as the v0.1 recursion)
double legendreFunction(int n, double x){
    if (n == 0)
        return 1.0;
    double p0 = 1.0, p1 = x;
    for (int k = 2; k <= n; k++){
        double p2 = ((2.0*k - 1.0)*x*p1 - (k - 1.0)*p0)/k;
        p0 = p1;
        p1 = p2;
    }
    return p1;
}

//Table P_n(u_t) for n = 0..maxPower and every input, computed once
void fillLegendreTable(Data &data){
    const arma::uword L = data.trainingTargetFromInputFile.n_elem;
    data.legendreTable.set_size(data.maxPowerNonlinearMemory + 1, L);
    for (arma::uword t = 0; t < L; t++)
        for (unsigned int n = 0; n <= data.maxPowerNonlinearMemory; n++)
            data.legendreTable(n, t) = legendreFunction(n, data.trainingTargetFromInputFile[t]);
}

//Returns 1.0 -  the cumulate distribution function, i.e. it returns the probability that "chi_squared" of "degree N" rolls "x" greater than "value". 
double getCDFChiSquared(int degrees, double value)
{
	chi_squared dist(degrees);
	return 1.0 - cdf(dist, value);
}

//Significance threshold: factor * (the value a chi-squared variable with N degrees of freedom exceeds with
//probability `probability`) / T. For a random target the capacity estimate is approximately chi2(N)/T.
//(v0.1 found the quantile by a decimal search to within 1e-5; this is the exact quantile.)
double findThreshold(Data &data, double probability)
{
	chi_squared dist(data.stateMatrix.n_cols);
	double value = boost::math::quantile(boost::math::complement(dist, probability));
	return value*data.thresholdFactor/data.stateMatrix.n_rows;
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
                    data.trainingTarget(k) *= data.legendreTable(powerList[j], k+data.startTargetRow-steps[j]);
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

    if (!data.thresholdGiven)
        data.threshold = findThreshold(data, data.thresholdProbability);
    if (data.debug)
        cout << "threshold = " << data.threshold << endl;
    fillLegendreTable(data);

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
    if (!data_file_nonlinear_memory)
        throw std::runtime_error("cannot write '" + data.dataStringNonlinearMemorySavefile + "'");
    data_file_nonlinear_memory.precision(data.outputPrecision);
    // header: one name per column (v0.1 wrote "STEP DEGREE CAPACITY" for 4P+1 columns)
    const unsigned int P = data.maxPowerNonlinearMemory;
    for (unsigned int j = 1; j <= P; j++) data_file_nonlinear_memory << "step" << j << " ";
    for (unsigned int j = 1; j <= P; j++) data_file_nonlinear_memory << "degree" << j << " ";
    data_file_nonlinear_memory << "capacity ";
    for (unsigned int j = 1; j <= P; j++) data_file_nonlinear_memory << "total_deg" << j << " ";
    for (unsigned int j = 1; j <= P; j++) data_file_nonlinear_memory << "pure_deg" << j << (j < P ? " " : "");
    data_file_nonlinear_memory << endl;

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

    double total = 0.0;
    cout << "Capacity per degree:";
    for (unsigned int p = 1; p <= data.maxPowerNonlinearMemory; p++){
        cout << " " << p << ": " << data.totalMemCapDegrees[p];
        total += data.totalMemCapDegrees[p];
    }
    cout << " | total " << total << " of N = " << data.stateMatrix.n_cols << endl;
}
