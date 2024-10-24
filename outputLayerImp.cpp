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
    softmaxOutput.resize(outputSize, 0.0);

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

void OutputLayer::setSoftmax()
{
    double denominator = 0.0;
    for(int i = 0; i < output.size(); i++)
        denominator += exp(output[i]);
    for(int i = 0; i < output.size(); i++)
        softmaxOutput[i] = exp(output[i]) / denominator;
}

void OutputLayer::displayInfoOutputLayer()
{
    // cout << "Weight Connections (Hidden -> Output):" << endl;
    // for(int i = 0; i < inputSize; i++)
    // {
    //     for(int j = 0; j < outputSize; j++)
    //     {
    //         cout << "Hidden layer " << i << " to Ouput layer " << j << ": " << weights[i][j] << endl;
    //     }
    // }

    // cout << endl;

    // cout << "Biases per OUTPUT neuron:" << endl;
    // for(int i = 0; i < outputSize; i++)
    //     cout << "Output Neuron " << i << ": " << bias[i] << endl;    

    // cout << endl;

    // cout << "Logits per OUTPUT neuron:" << endl;
    // for(int i = 0; i < outputSize; i++)
    // {
    //     cout << "Logit of neuron " << i << ": " << output[i] << endl;
    // }

    cout << endl;
    cout << "Predictions (Softmax Outputs):" << endl;
    cout << "1: Diagonal, 2: Vertical, 3: Horizontal" << endl;
    for(int i = 0; i < outputSize; i++)
        cout << i + 1 << ": " << softmaxOutput[i] * 100 << "%" << endl;

    cout << endl << endl << endl;
}

vector<double>& OutputLayer::getOutput() {return output;}

double OutputLayer::meanSquaredErrorCostPerImage(const vector<double>& target)
{
    double costPerImage = 0.0;
    for(int i = 0; i < outputSize; i++)
    {     
        costPerImage += (pow(softmaxOutput[i] - target[i], 2));
    }
    return costPerImage;
}

vector<double> OutputLayer::deltaForOutputNeurons(const vector<double>& targetOutput)
{
    vector<double> deltas(outputSize);   
    for (int i = 0; i < outputSize; i++)
    {
        deltas[i] = meanSquaredErrorDerivative(softmaxOutput[i],targetOutput[i]) * rectifiedLinearUnitDerivative(softmaxOutput[i]);
    }

    return deltas; 
}


vector<double> OutputLayer::getDeltas(){return deltas;}

double OutputLayer::rectifiedLinearUnitDerivative(int activation)
{
    if(activation > 0) return 1.0;
    else return 0.0;
}

double OutputLayer::meanSquaredErrorDerivative(double target, double softmax) {return 2 * (softmax - target);}

void OutputLayer::calculateGradientsWeight(vector<double> activationPrevLayer)
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

void OutputLayer::calculateGradientsBias()
{
    biasGradients.resize(outputSize, 0.0);
    for(int i = 0; i < outputSize; i++)
    {
        biasGradients[i] = deltas[i];
    }
}
vector<vector<double>>& OutputLayer::getWeights(){return weights;}
vector<double>& OutputLayer::getBiases(){return bias;}
vector<vector<double>>& OutputLayer::getWeightGradients(){return weightGradients;}
vector<double>& OutputLayer::getBiasGradients(){return biasGradients;}
double OutputLayer::calculateTotalWeightGradients()
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
double OutputLayer::calculateTotalBiasGradients()
{
    double total = 0.0;
    for(int i = 0; i < outputSize; i++)
    {
        total += biasGradients[i];
    }
    return total;
}

void OutputLayer::updateWeights(double learningRate, double weightGradientAvg)
{

    for(int i = 0; i < outputSize; i++)
    {
        for(int j = 0; j < inputSize; j++)
        {
            weightGradients[i][j] = weightGradients[i][j] - (learningRate * weightGradientAvg);
        }
    }
}

void OutputLayer::updateBias(double learningRate, double biasGradientAvg)
{
    for(int i = 0; i < outputSize; i++)
    {
        bias[i] = bias[i] - learningRate * biasGradientAvg;
    }
}