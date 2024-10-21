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
    vector <double> softmaxResult = softmax();
    for(int i = 0; i < outputSize; i++)
        cout << i + 1 << ": " << softmaxResult[i] * 100 << "%" << endl;

    cout << endl << endl << endl;
}

const vector<double>& OutputLayer::getOutput() const{return output;}

// Returns a vector of all COST of each output compared to the actual output
vector<double> OutputLayer::meanSquaredError(const vector<double>& softmaxOutput, const vector<double>& target)
{
    vector <double> costPerNeuron(outputSize);
    for(int i = 0; i < outputSize; i++)
    {
        costPerNeuron[i] = (pow(target[i] - output[i], 2));
    }
    return costPerNeuron;
}

vector<double> OutputLayer::calculateErrorDelta(const vector<double>& targetOutput)
{
    vector<double> softmaxOutput = softmax();
    vector<double> deltas(outputSize);
    for (int i = 0; i < outputSize; i++)
    {
        deltas[i] = softmaxOutput[i] - targetOutput[i]; // Cross-entropy gradient
    }
    return deltas;
}


vector<double> OutputLayer::getDeltas(){return deltas;}

vector<double> OutputLayer::rectifiedLinearUnitDerivative()
{
    vector <double> reLUDerivativePerNeuron(outputSize);
    for(int i = 0; i < outputSize; i++)
    {
        if(output[i] > 0) reLUDerivativePerNeuron[i] = 1.0;
        else reLUDerivativePerNeuron[i] = 0.0;
    }
    return reLUDerivativePerNeuron;
}

vector<double> OutputLayer::meanSquaredErrorDerivative(vector<double> actual)
{
    vector <double> meanSquaredErrorDerivativePerNeuron(outputSize);
    vector <double> softMaxOutput = softmax();
    for(int i = 0; i < outputSize; i++)
    {
        meanSquaredErrorDerivativePerNeuron[i] = 2 * (actual[i] - softMaxOutput[i]);
    }
    return meanSquaredErrorDerivativePerNeuron;
}

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