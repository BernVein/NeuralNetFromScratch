#include <iostream>
#include "hiddenLayer.h"
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <algorithm>

HiddenLayer::HiddenLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize)
{
    srand(time(NULL));
    
    weights.resize(inputSize, vector<double>(outputSize));
    for (int i = 0; i < inputSize; i++)
    {
        for (int j = 0; j < outputSize; j++)
        {
            weights[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }

    bias.resize(outputSize); 
    for (int i = 0; i < outputSize; i++)
        bias[i] = (static_cast<double>(rand()) / RAND_MAX) * 0.2 - 0.1;
}



vector<double> HiddenLayer::propagateForward(const vector<double>& inputData)
{
    output.resize(outputSize);
    for(int j = 0; j < outputSize; j++)
    {
        output[j] = 0;
        for(int i = 0; i < inputSize; i++)
        {
            output[j] += inputData[i] * weights[i][j];
        }
        output[j] += bias[j];
        output[j] = rectifiedLinearUnit(output[j]);
    }
    return output;
}

vector<double> HiddenLayer::getOutput(){return output;}

void HiddenLayer::displayInfoHiddenLayer()
{
    cout << "Weight Connections (Input -> Hidden):" << endl;
    for (int i = 0; i < inputSize; i++)
    {
        for (int j = 0; j < outputSize; j++)
        {
            cout << "Input layer " << i << " to Hidden layer " << j << ": " << weights[i][j] << endl;
        }
    }

    cout << endl;

    cout << "Biases per HIDDEN neuron:" << endl;
    for(int i = 0; i < outputSize; i++)
        cout << "Hidden Neuron " << i << ": " << bias[i] << endl;    

    cout << endl;

    cout << "Outputs:" << endl;
    for (int i = 0; i < outputSize; i++)
    {
        cout << "Output " << i << ": " << output[i] << endl;
    }
    cout << endl << endl << endl;
}


double HiddenLayer::rectifiedLinearUnit(double num) {return max(0.0, num);}