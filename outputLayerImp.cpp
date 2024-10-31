#include <iostream>
#include "outputLayer.h"
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

OutputLayer::OutputLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize)
{
    weights.resize(outputSize, vector<double>(inputSize, 0.0));
    deltas.resize(outputSize, 0.0);
    output.resize(outputSize);
    weightGradients.resize(outputSize, vector<double>(inputSize, 0.0));
    output.resize(outputSize);
    vector<double> deltas(outputSize);
    biasGradients.resize(outputSize, 0.0);
    bias.resize(outputSize); 
    logits.resize(outputSize, 0.0);

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

void OutputLayer::propagateForward(const vector<double> &inputData)
{
    for(int j = 0; j < outputSize; j++)
    {
        output[j] = 0;
        for(int i = 0; i < inputData.size(); i++)
        {
            output[j] = output[j] + (inputData[i] * weights[j][i]);
        }
        output[j] += bias[j];
        logits[j] = output[j];
        output[j] = sigmoid(output[j]);
    } 
}

void OutputLayer::displayInfoOutputLayer()
{
    // cout << "Weight Connections (Output -> Hidden):" << endl;
    // for(int j = 0; j < outputSize; j++)
    // {
    //     for(int i = 0; i < inputSize; i++)
    //     {
    //         cout << "Output layer " << j << " back to to Hidden layer " << i << ": " << weights[j][i] << endl;
    //     }
    // }

    // cout << endl;

    // cout << "Biases per OUTPUT neuron:" << endl;
    // for(int j = 0; j < outputSize; j++)
    //     cout << "Output Neuron " << j << ": " << bias[j] << endl;    

    // cout << endl;

    // cout << "OUTPUT LAYER Normalized Outputs:" << endl;
    // for (int j = 0; j < outputSize; j++)
    // {
    //     cout << "Output " << j << ": " << output[j] << endl;
    // }

    // cout << endl;
    // cout << "Deltas for each OUTPUT neuron:" << endl;
    // for(int j = 0; j < outputSize; j++)
    // {
    //     cout << "Delta " << j << ": " << deltas[j] << endl;
    // }
    // cout << endl;
    cout << endl;
    cout << "Logits: " << endl;
    for(int i = 0; i < outputSize; i++)
        cout << "Logit " << i + 1 << ": " << logits[i] << endl;
}

vector<double>& OutputLayer::getOutput() {return output;}

double OutputLayer::meanSquaredErrorCostPerImage(const vector<double>& target)
{
    double costPerImage = 0.0;
    for(int i = 0; i < outputSize; i++)
    {     
        costPerImage += (pow(target[i] - output[i], 2));
    }
    return costPerImage;
}

vector<double> OutputLayer::deltaForOutputNeurons(const vector<double>& targetOutput)
{
    for (int i = 0; i < outputSize; i++)
    {
        deltas[i] = meanSquaredErrorDerivative(targetOutput[i], output[i]) * sigmoidDerivative(output[i]);
    }
    return deltas; 
}

double OutputLayer::sigmoid(double x) 
{
    return 1.0 / (1.0 + exp(-x));
}

void OutputLayer::predict()
{
    vector<double> softmaxValues(outputSize);
    double denominator = 0.0;

    for(int i = 0; i < outputSize; i++)
    {
        denominator += exp(logits[i]);
    }

    for(int i = 0; i < outputSize; i++)
    {
        softmaxValues[i] = exp(logits[i]) / denominator;
    }

    int maxIndex = 0;
    for(int i = 1; i < outputSize; i++)
    {
        if(softmaxValues[i] > softmaxValues[maxIndex]) maxIndex = i;
    }
    string prediction = "";
    if(maxIndex == 0) prediction = "Diagonal";
    else if(maxIndex == 1) prediction = "Vertical";
    else prediction = "Horizontal";

    cout << "The image is probably " << prediction << " with a confidence of " << softmaxValues[maxIndex] * 100 << "%." << endl;
}

vector<double> OutputLayer::getDeltas(){return deltas;}

double OutputLayer::sigmoidDerivative(double activation) 
{
    return activation * (1.0 - activation);
}


double OutputLayer::meanSquaredErrorDerivative(double target, double output) {return 2 * (target - output);}

void OutputLayer::calculateGradientsWeight(vector<double> activationPrevLayer)
{
    for(int j = 0; j < outputSize; j++)
    {
        for(int i = 0; i < inputSize; i++)
        {
            weightGradients[j][i] = deltas[j] * activationPrevLayer[i];
        } 
    }
}

void OutputLayer::calculateGradientsBias()
{
    for(int j = 0; j < outputSize; j++)
    {
        biasGradients[j] = deltas[j];
    }
}
vector<vector<double>>& OutputLayer::getWeights(){return weights;}
vector<double>& OutputLayer::getBiases(){return bias;}
vector<vector<double>>& OutputLayer::getWeightGradients(){return weightGradients;}
vector<double>& OutputLayer::getBiasGradients(){return biasGradients;}


void OutputLayer::updateWeights(double learningRate)
{
    for(int j = 0; j < outputSize; j++)
    {
        for(int i = 0; i < inputSize; i++)
        {
            weights[j][i] += (learningRate * weightGradients[j][i]);
        }
    }
}

void OutputLayer::updateBias(double learningRate)
{
    for(int j = 0; j < outputSize; j++)
    {
        bias[j] += learningRate * biasGradients[j];
    }
}