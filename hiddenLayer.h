#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H
#include <vector>
using namespace std;
class HiddenLayer
{
    public:
        HiddenLayer(int inputSize, int outputSize);
        void propagateForward(const vector<double>& inputData);
        void displayInfoHiddenLayer();
        vector<double> getOutput();
        vector<double> getDeltas();
        void calculateDelta(vector<double> deltaNextLayer, vector<vector<double>> weightsNextLayer);
        void calculateGradientsWeight(vector<double> activationPrevLayer);
        void calculateGradientsBias();
        vector<double> getBiasGradient();
        vector<vector<double>> getWeightGradients();
        void updateWeights(double learningRate);
        void updateBias(double learningRate);
    private:
        int inputSize;
        int outputSize;
        vector<double> bias;
        vector<vector<double>> weights;
        vector<vector<double>> weightGradients;
        vector<double> biasGradients;
        vector<double> deltas;
        vector<double> output;
        double sigmoid(double x);
        double sigmoidDerivative(double activation);

};
#endif