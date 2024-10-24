#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "hiddenLayer.h"

HiddenLayer::HiddenLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize)
{
    weights.resize(inputSize, vector<double>(outputSize, 0.0));
    deltas.resize(outputSize, 0.0);
    output.resize(outputSize);
    
    srand(static_cast<unsigned int>(time(0)));
    for(int i = 0; i < inputSize; i++)
    {
        for(int j = 0; j < outputSize; j++)
        {
            weights[i][j] = static_cast<double>(rand()) / RAND_MAX - 0.5;
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

double HiddenLayer::rectifiedLinearUnitDerivative(double activation)
{
    vector <double> reLUDerivativePerNeuron(outputSize);
    if(activation > 0.0) return 1.0;
    else return 0.0;
}

vector <double> HiddenLayer::calculateDelta(vector<double> deltaNextLayer, vector<vector<double>> weightsNextLayer)
{
    for(int i = 0; i < outputSize; i++)
    {
        double delta = 0.0;
        for(int j = 0; j < weightsNextLayer.size(); j++)
        {
            delta += deltaNextLayer[j] * weightsNextLayer[j][i] * rectifiedLinearUnit(output[i]);
        }
        deltas[i] = delta;
    }
}
vector<double> HiddenLayer::getDeltas(){return deltas;}

void HiddenLayer::calculateGradientsWeight(vector<double> activationPrevLayer)
{
    weightGradients.resize(outputSize, vector<double>(inputSize, 0.0));
    for(int i = 0; i < outputSize; i++)
    {
        for(int j = 0; j < inputSize; j++)
        {
            weightGradients[i][j] = deltas[i] * activationPrevLayer[j];
        } 
    }
}

void HiddenLayer::calculateGradientsBias()
{
    biasGradients.resize(outputSize, 0.0);
    for(int i = 0; i < outputSize; i++)
    {
        biasGradients[i] = deltas[i];
    }
}

double HiddenLayer::rectifiedLinearUnit(double num) {return max(0.0, num);}
vector<double> HiddenLayer::getBiasGradient(){return biasGradients;}
vector<vector<double>> HiddenLayer::getWeightGradients(){return weightGradients;}

double HiddenLayer::calculateTotalWeightGradients()
{
    double total = 0.0;
    for(int i = 0; i < outputSize; i++)
    {
        for(int j = 0; j < inputSize; j++)
        {
            total += weightGradients[i][j];
        } 
    }
    return total;
}
double HiddenLayer::calculateTotalBiasGradients()
{
    double total = 0.0;
    for(int i = 0; i < outputSize; i++)
    {
        total += biasGradients[i];
    }
    return total;
}

void HiddenLayer::updateWeights(double learningRate, double weightGradientAvg)
{

    for(int i = 0; i < outputSize; i++)
    {
        for(int j = 0; j < inputSize; j++)
        {
            weightGradients[i][j] = weightGradients[i][j] - (learningRate * weightGradientAvg);
        }
    }
}

void HiddenLayer::updateBias(double learningRate, double biasGradientAvg)
{
    for(int i = 0; i < outputSize; i++)
    {
        bias[i] = bias[i] - learningRate * biasGradientAvg;
    }
}