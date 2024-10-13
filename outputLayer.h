#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H
#include <vector>
using namespace std;
class OutputLayer
{
    public:
        OutputLayer(int inputSize, int outputSize);
        vector<double> propagateForward(const vector<double>& inputData);
        void calculateErrorDelta(const vector<double>& targetOutput);
        void updateWeights(double learningRate);
        const vector<double>& getOutput() const;
        vector<double> softmax();
        void displayInfoOutputLayer();
    private:
        int inputSize;
        int outputSize;
        vector<double> output;
        vector<vector<double>> weights;
        vector<double> deltas;
        vector<double> bias;
        
};

#endif