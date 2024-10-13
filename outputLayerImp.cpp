#include <iostream>
#include "outputLayer.h"
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <algorithm>

OutputLayer::OutputLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize)
{
    weights.resize(inputSize, vector<double>(outputSize, 0.0));
    deltas.resize(outputSize, 0.0);
    output.resize(outputSize);

    for(int i = 0; i < inputSize; i++)
    {
        for(int j = 0; j < outputSize; j++)
        {
            // para 0.0 to 1.0, rand() divided by RAND_MAX
            weights[i][j] = static_cast<double>(rand()) / RAND_MAX;

        }
    }
    bias.resize(outputSize); 
    for (int i = 0; i < outputSize; i++)
        bias[i] = (static_cast<double>(rand()) / RAND_MAX) * 0.2 - 0.1;
}

vector<double> OutputLayer::propagateForward(const vector<double> &inputData)
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
    }
    return output;
}

vector<double> OutputLayer::softmax()
{
    vector<double> softmaxOutput(output.size()); 
    double denominator = 0.0;
    for(int i = 0; i < output.size(); i++)
        denominator += exp(output[i]);
    for(int i = 0; i < output.size(); i++)
        softmaxOutput[i] = exp(output[i]) / denominator;
    return softmaxOutput;
}

void OutputLayer::calculateErrorDelta(const vector<double>& targetOutput)
{
    for(int i = 0; i < outputSize; i++)
        deltas[i] = output[i] - targetOutput[i];
}

void OutputLayer::displayInfoOutputLayer()
{
    cout << "Weight Connections (Hidden -> Output):" << endl;
    for(int i = 0; i < inputSize; i++)
    {
        for(int j = 0; j < outputSize; j++)
        {
            cout << "Hidden layer " << i << " to Ouput layer " << j << ": " << weights[i][j] << endl;
        }
    }

    cout << endl;

    cout << "Biases per OUTPUT neuron:" << endl;
    for(int i = 0; i < outputSize; i++)
        cout << "Output Neuron " << i << ": " << bias[i] << endl;    

    cout << endl;
    cout << "Predictions (Softmax Outputs):" << endl;
    cout << "1: Diagonal, 2: Vertical, 3: Horizontal" << endl;
    vector <double> softmaxResult = softmax();
    for(int i = 0; i < outputSize; i++)
        cout << i + 1 << ": " << softmaxResult[i] * 100 << "%" << endl;

    cout << endl << endl << endl;

}

const vector<double>& OutputLayer::getOutput() const{return output;}
