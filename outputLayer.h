#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H
#include <vector>
using namespace std;
class OutputLayer
{
    public:
        OutputLayer(int inputSize, int outputSize);
        double meanSquaredErrorCostPerImage(const vector<double>& targetOutput);
        vector<double> deltaForOutputNeurons(const vector<double>& targetOutput);
        void displayInfoOutputLayer();
        vector<double> getDeltas();
        vector<vector<double>>& getWeights();
        vector<double>& getBiases();
        void calculateGradientsWeight(vector<double> activationPrevLayer);
        vector<vector<double>>& getWeightGradients();
        void propagateForward(const vector<double>& inputData);
        vector<double>& getBiasGradients();
        void calculateGradientsBias();
        void updateWeights(double learningRate);
        void updateBias(double learningRate);
        vector<double>& getOutput();
        vector<double> predict();

    private:
        int inputSize;
        vector<vector<double>> weightGradients;
        vector<double> biasGradients;
        int outputSize;
        vector<double> output;
        vector<vector<double>> weights;
        double sigmoid(double x);
        vector<double> deltas;
        vector<double> bias;
        vector<double> logits;
        double meanSquaredErrorDerivative(double target, double output);
        double sigmoidDerivative(double activation);    
};

#endif