#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "hiddenLayer.h"
#include <vector>

HiddenLayer::HiddenLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize)
{
    weights.resize(outputSize, vector<double>(inputSize, 0.0));
    deltas.resize(outputSize, 0.0);
    output.resize(outputSize);
    bias.resize(outputSize); 
    weightGradients.resize(outputSize, vector<double>(inputSize, 0.0));
    biasGradients.resize(outputSize, 0.0);

    srand(static_cast<unsigned int>(time(0)));
    for(int j = 0; j < outputSize; j++)
    {
        for(int i = 0; i < inputSize; i++)
        {
            weights[j][i] = static_cast<double>(rand()) / RAND_MAX - 0.5;
        }
    }

    for (int j = 0; j < outputSize; j++)
    {
        bias[j] = (static_cast<double>(rand()) / RAND_MAX) * 0.2 - 0.1;
    }
        
}

double HiddenLayer::sigmoidDerivative(double activation) 
{
    return activation * (1.0 - activation);
}

void HiddenLayer::propagateForward(const vector<double>& inputData)
{
    for(int j = 0; j < outputSize; j++)
    {
        output[j] = 0;
        for(int i = 0; i < inputSize; i++)
        {
            output[j] = output[j] + (inputData[i] * weights[j][i]);
        }
        output[j] += bias[j];
        output[j] = sigmoid(output[j]);
    }
}

vector<double> HiddenLayer::getOutput(){return output;}

void HiddenLayer::displayInfoHiddenLayer()
{
    // cout << "Weight Connections (Hidden -> Input):" << endl;
    // for (int j = 0; j < outputSize; j++)
    // {
    //     for (int i = 0; i < inputSize; i++)
    //     {
    //         cout << "Hidden layer " << j << " back to Input layer " << i << ": " << weights[j][i] << endl;
    //     }
    // }

    // cout << endl;

    // cout << "Biases per HIDDEN neuron:" << endl;
    // for(int j = 0; j < outputSize; j++)
    //     cout << "Hidden Neuron " << j << ": " << bias[j] << endl;    

    // cout << endl;

    // cout << "HIDDEN LAYER Normalized Outputs:" << endl;
    // for (int j = 0; j < outputSize; j++)
    // {
    //     cout << "Output " << j << ": " << output[j] << endl;
    // }
    // cout << endl;
        
    cout << "Deltas for each HIDDEN neuron:" << endl;
    for(int j = 0; j < outputSize; j++)
    {
        cout << "Delta " << j << ": " << deltas[j] << endl;
    }
    cout << endl << endl;
}

double HiddenLayer::sigmoid(double x) {return 1.0 / (1.0 + exp(-x));}

void HiddenLayer::calculateDelta(vector<double> deltaNextLayer, vector<vector<double>> weightsNextLayer)
{
    for(int j = 0; j < outputSize; j++)
    {
        double deltaPerNeuron = 0.0;
        
        for(int i = 0; i < weightsNextLayer.size(); i++)
        {
            //cout << "(" << deltaPerNeuron;
            deltaPerNeuron = deltaPerNeuron + deltaNextLayer[i] * weightsNextLayer[i][j] * sigmoidDerivative(output[j]);
            // cout << ")+(" << deltaNextLayer[i] << ")*(" << weightsNextLayer[i][j] << ")*(" << sigmoidDerivative(output[j]) << ")=(" << deltaPerNeuron << ")" << endl;
        }
        deltas[j] = deltaPerNeuron;
    }
}

vector<double> HiddenLayer::getDeltas(){return deltas;}

void HiddenLayer::calculateGradientsWeight(vector<double> activationPrevLayer)
{
    for(int j = 0; j < outputSize; j++)
    {
        for(int i = 0; i < inputSize; i++)
        {
            weightGradients[j][i] = deltas[j] * activationPrevLayer[i];
        } 
    }
}

void HiddenLayer::calculateGradientsBias()
{
    for(int j = 0; j < outputSize; j++)
    {
        biasGradients[j] = deltas[j];
    }
}

vector<double> HiddenLayer::getBiasGradient(){return biasGradients;}
vector<vector<double>> HiddenLayer::getWeightGradients(){return weightGradients;}

void HiddenLayer::updateWeights(double learningRate)
{
    for(int j = 0; j < outputSize; j++)
    {
        for(int i = 0; i < inputSize; i++)
        {
            weights[j][i] = weights[j][i] + (learningRate * weightGradients[j][i]);
        }
    }
}

void HiddenLayer::updateBias(double learningRate)
{
    for(int j = 0; j < outputSize; j++)
    {
        bias[j] = bias[j] + learningRate * biasGradients[j];
    }
}