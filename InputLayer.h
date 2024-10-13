#ifndef INPUTLAYER_H
#define INPUTLAYER_H
#include <vector>
#include <iostream>
using namespace std;

class InputLayer
{
    public:
        InputLayer(int inputSize);
        void setInputData(const vector<double>& data);
        const vector<double>& getInputData() const;
    private:
        // Number of neurons for input layer
        int size;
        // Data in the input neuron data
        vector<double> inputData;
};

#endif