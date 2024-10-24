#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H
#include <vector>
using namespace std;
class HiddenLayer
{
    public:
        HiddenLayer(int inputSize, int outputSize);
        vector<double> propagateForward(const vector<double>& inputData);
        void displayInfoHiddenLayer();
        vector<double> getOutput();
        vector<double> getDeltas();
        vector <double> calculateDelta(vector<double> deltaNextLayer, vector<vector<double>> weightsNextLayer);
        void calculateGradientsWeight(vector<double> activationPrevLayer);
        void calculateGradientsBias();
        vector<double> getBiasGradient();
        vector<vector<double>> getWeightGradients();
        double calculateTotalWeightGradients();
        double calculateTotalBiasGradients();
        void updateWeights(double learningRate, double weightGradientAvg);
        void updateBias(double learningRate, double biasGradientAvg);
    private:
        int inputSize;
        int outputSize;
        vector<double> bias;
        vector<vector<double>> weights;
        vector<vector<double>> weightGradients;
        vector<double> biasGradients;
        vector<double> deltas;
        vector<double> output;
        double rectifiedLinearUnitDerivative(double activation);
        double rectifiedLinearUnit(double num);

};
#endif