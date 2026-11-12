#include "Data.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

//------------------------------------------------------------------------------------------------------------------------------------------//
// Command-line parsing. Every flag takes one value: -flag value
static void printUsage() {
    cout << "Usage: ./MC [options]\n"
            "  -state_file FILE                    state matrix, one row per time step (default stateData)\n"
            "  -input_file FILE                    inputs in [-1, 1], one per time step (default inputData)\n"
            "  -output_file FILE                   output table (default memoryData)\n"
            "  -start_target_row K                 state row k belongs to input index k + K (default 1000)\n"
            "  -steps_for_memory D                 delays 0 .. D-1 are tested (default 50)\n"
            "  -max_power_nonlinear_memory P       highest total degree (default 3)\n"
            "  -regularisation_factor L            ridge term (default 1e-6)\n"
            "  -regularisation_factor_relative 0|1 scale L by max(X^T X) (default 0)\n"
            "  -threshold C                        fixed significance threshold (default: from chi-squared)\n"
            "  -threshold_probability P            chi-squared tail probability (default 1e-5)\n"
            "  -threshold_factor F                 threshold = F * chi2_quantile(N, 1-P) / T (default 4)\n"
            "  -do_without_threshold 0|1           report every capacity (default 0)\n"
            "  -patience N                         stop a delay scan after N sub-threshold delays; 0 = full scan "
            "(default 20)\n"
            "  -precision N                        digits written (default 6)\n"
            "  -debug 0|1\n";
}

static double numberArg(int argc, char **argv, int &i) {
    if (i + 1 >= argc)
        throw std::invalid_argument(string("missing value after ") + argv[i]);
    char *end = nullptr;
    double v = strtod(argv[i + 1], &end);
    if (end == argv[i + 1] || *end != '\0')
        throw std::invalid_argument(string("not a number after ") + argv[i] + ": " + argv[i + 1]);
    ++i;
    return v;
}

static string stringArg(int argc, char **argv, int &i) {
    if (i + 1 >= argc)
        throw std::invalid_argument(string("missing value after ") + argv[i]);
    return string(argv[++i]);
}

bool parsArguments(int argc, char **argv, Data &data) {
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-h" || a == "--help") { printUsage(); exit(0); }
        else if (a == "-debug") data.debug = numberArg(argc, argv, i) != 0;
        else if (a == "-max_power_nonlinear_memory") data.maxPowerNonlinearMemory = (unsigned)numberArg(argc, argv, i);
        else if (a == "-do_without_threshold") data.doWithoutThreshold = numberArg(argc, argv, i) != 0;
        else if (a == "-threshold") { data.threshold = numberArg(argc, argv, i); data.thresholdGiven = true; }
        else if (a == "-threshold_probability") data.thresholdProbability = numberArg(argc, argv, i);
        else if (a == "-threshold_factor") data.thresholdFactor = numberArg(argc, argv, i);
        else if (a == "-steps_for_memory") data.stepsForMemory = (unsigned)numberArg(argc, argv, i);
        else if (a == "-regularisation_factor") data.regularisationFactor = numberArg(argc, argv, i);
        else if (a == "-regularisation_factor_relative") data.regularisationFactorRelative = numberArg(argc, argv, i) != 0;
        else if (a == "-start_target_row") data.startTargetRow = (int)numberArg(argc, argv, i);
        else if (a == "-patience") data.numberLowMemLast = (int)numberArg(argc, argv, i);
        else if (a == "-precision") data.outputPrecision = (int)numberArg(argc, argv, i);
        else if (a == "-state_file") data.stateFile = stringArg(argc, argv, i);
        else if (a == "-input_file") data.inputFile = stringArg(argc, argv, i);
        else if (a == "-output_file") data.dataStringNonlinearMemorySavefile = stringArg(argc, argv, i);
        else throw std::invalid_argument("unknown option " + a + " (see -h)");
    }
    if (data.debug)
        cout << "maxPower=" << data.maxPowerNonlinearMemory << " steps=" << data.stepsForMemory
             << " startTargetRow=" << data.startTargetRow << " patience=" << data.numberLowMemLast << endl;
    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------//
// Load and check the state matrix and the inputs
void loadData(Data &data) {
    if (!data.stateMatrix.load(data.stateFile) || data.stateMatrix.n_elem == 0)
        throw std::runtime_error("could not read the state matrix from '" + data.stateFile + "'");
    if (!data.trainingTargetFromInputFile.load(data.inputFile) || data.trainingTargetFromInputFile.n_elem == 0)
        throw std::runtime_error("could not read the inputs from '" + data.inputFile + "'");
    const long T = data.stateMatrix.n_rows, nIn = data.trainingTargetFromInputFile.n_elem;
    const long K = data.startTargetRow, D = data.stepsForMemory;
    if (K < D - 1)
        throw std::runtime_error("-start_target_row (" + to_string(K) + ") must be >= steps_for_memory - 1 (" +
                                 to_string(D - 1) + "): the oldest delay would index before the first input");
    if (nIn < T + K)
        throw std::runtime_error("the input file has " + to_string(nIn) + " values, but " + to_string(T) +
                                 " state rows with start_target_row " + to_string(K) + " need at least " +
                                 to_string(T + K) + " (state row k belongs to input k + start_target_row)");
    if (arma::abs(data.trainingTargetFromInputFile).max() > 1.0 + 1e-9)
        cerr << "warning: inputs outside [-1, 1]; Legendre polynomials are only orthogonal for uniform inputs on [-1, 1]"
             << endl;
    if (T <= (long)data.stateMatrix.n_cols)
        cerr << "warning: fewer time steps (" << T << ") than state dimensions (" << data.stateMatrix.n_cols
             << "); capacities will be meaningless" << endl;
    data.trainingTarget = arma::Col<double>(data.stateMatrix.n_rows);
    data.totalMemCapDegrees.assign(data.maxPowerNonlinearMemory + 1, 0.0);
    data.totalMemCapDegreesPure.assign(data.maxPowerNonlinearMemory + 1, 0.0);
    if (data.numberLowMemLast <= 0)  // full scan
        data.numberLowMemLast = (int)data.stepsForMemory + 1;
}

void initializeVariables(int argc, char **argv, Data &data) {
    parsArguments(argc, argv, data);
    loadData(data);
}
