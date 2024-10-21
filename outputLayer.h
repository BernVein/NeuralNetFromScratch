#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H
#include <vector>
using namespace std;
class OutputLayer
{
    public:
        OutputLayer(int inputSize, int outputSize);
        vector<double> propagateForward(const vector<double>& inputData);
        vector<double> calculateErrorDelta(const vector<double>& targetOutput);
        vector<double> meanSquaredError(const vector<double>& softmaxOutput, const vector<double>& target);
        const vector<double>& getOutput() const;
        vector<double> softmax();
        void displayInfoOutputLayer();
        vector<double> getDeltas();
        vector<vector<double>>& getWeights();
        vector<double>& getBiases();
        void calculateGradientsWeight(vector<double> activationPrevLayer);
        vector<vector<double>>& getWeightGradients();

    private:
        int inputSize;
        vector<vector<double>> weightGradients;
        vector<double> biasGradients;
        int outputSize;
        vector<double> output;
        void calculateGradientsBias();
        vector<vector<double>> weights;
        vector<double> deltas;
        vector<double> bias;
        vector<double> meanSquaredErrorDerivative(vector<double> actual);
        vector<double> rectifiedLinearUnitDerivative();
        
};

#endif