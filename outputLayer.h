#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H
#include <vector>
using namespace std;
class OutputLayer
{
    public:
        OutputLayer(int inputSize, int outputSize);
        vector<double> propagateForward(const vector<double>& inputData);
        double meanSquaredErrorCostPerImage(const vector<double>& targetOutput);
        vector<double> deltaForOutputNeurons(const vector<double>& targetOutput);
        void setSoftmax();
        void displayInfoOutputLayer();
        vector<double> getDeltas();
        vector<vector<double>>& getWeights();
        vector<double>& getBiases();
        void calculateGradientsWeight(vector<double> activationPrevLayer);
        vector<vector<double>>& getWeightGradients();
        vector<double>& getBiasGradients();
        void calculateGradientsBias();
        double calculateTotalWeightGradients();
        double calculateTotalBiasGradients();
        void updateWeights(double learningRate, double weightGradientAvg);
        void updateBias(double learningRate, double biasGradientAvg);
        
        

    private:
        int inputSize;
        vector<vector<double>> weightGradients;
        vector<double> biasGradients;
        int outputSize;
        vector<double>& getOutput();
        vector<double> output;
        vector<double> softmaxOutput;
        vector<vector<double>> weights;
        vector<double> deltas;
        vector<double> bias;
        double meanSquaredErrorDerivative(double target, double softmax);
        double rectifiedLinearUnitDerivative(int activation);    
};

#endif