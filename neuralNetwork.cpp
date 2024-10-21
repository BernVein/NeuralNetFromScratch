#include "InputLayer.h"
#include "hiddenLayer.h"
#include "outputLayer.h"
#include <iostream>
#include <vector>
#include <thread> // For sleep functionality
#include <chrono> // For duration
#include <cstdlib> // For system()
#include <unistd.h> 
using namespace std;


void displayVectorContent(const vector<double>& vec);
vector<vector<double>> weightGradientAverage(const vector<vector<double>>& weightGradients, int trainingDataCount);
vector<double> biasGradientAverage(vector<vector<double>> allDeltas, int trainingDataCount);
void updateWeightsAndBiases(vector<vector<double>>& weights, 
                             vector<double>& biases, 
                             vector<vector<double>>& weightGradients, 
                             vector<double>& biasGradients, 
                             double learningRate);

int main()
{
    const int numTrainingExamples = 1; // Use only one training example
    const double learningRate = 0.01; // Learning rate
    const int epochs = 1000; // Number of training epochs

    vector<double> horizontalImage = 
    {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    vector<double> horizontalImageActual = {1, 0, 0}; // Target output for the single class

    // Initialize layers
    InputLayer inputLayer(horizontalImage.size());
    HiddenLayer hiddenLayer(horizontalImage.size(), 5);
    OutputLayer outputLayer(hiddenLayer.getOutput().size(), 3);

    // Use a single training example
    vector<vector<double>> trainingImages = {horizontalImage};
    vector<vector<double>> targetOutputs = {horizontalImageActual};

    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        // Forward propagation
        inputLayer.setInputData(trainingImages[0]);
        hiddenLayer.propagateForward(inputLayer.getInputData());
        outputLayer.propagateForward(hiddenLayer.getOutput());

        // Calculate deltas and gradients
        auto deltas = outputLayer.calculateErrorDelta(targetOutputs[0]);
        outputLayer.calculateGradientsWeight(hiddenLayer.getOutput());

        // Update weights and biases
        updateWeightsAndBiases(outputLayer.getWeights(), outputLayer.getBiases(), outputLayer.getWeightGradients(), deltas, learningRate);
        
        // Optionally display output
        cout << "Epoch " << epoch + 1 << "/" << epochs << endl;
        outputLayer.displayInfoOutputLayer();
        system("CLS");
    }
    outputLayer.displayInfoOutputLayer();
    
    return 0;
}

void displayVectorContent(const vector<double>& vec)
{
    for(double num : vec)
        cout << num << ", ";
}

vector<vector<double>> weightGradientAverage(const vector<vector<double>>& weightGradients, int trainingDataCount)
{
    vector<vector<double>> averageWeightGradients = weightGradients;
    for(int i = 0; i < averageWeightGradients.size(); i++)
    {
        for(int j = 0; j < averageWeightGradients[i].size(); j++)
        {
            averageWeightGradients[i][j] /= trainingDataCount;
        }
    }
    return averageWeightGradients;
}

vector<double> biasGradientAverage(const vector<vector<double>> allDeltas, int trainingDataCount)
{
    vector<double> averageBiasGradients(allDeltas[0].size(), 0.0);
    for (const vector<double>& deltas : allDeltas)
    {
        for (int j = 0; j < deltas.size(); j++) 
        {
            averageBiasGradients[j] += deltas[j];
        }
    }
    for(int i = 0; i < averageBiasGradients.size(); i++)
    {
        averageBiasGradients[i] /= trainingDataCount;
    }
    return averageBiasGradients;
}

void updateWeightsAndBiases(vector<vector<double>>& weights, 
                             vector<double>& biases, 
                             vector<vector<double>>& weightGradients, 
                             vector<double>& biasGradients, 
                             double learningRate)
{
    for (int i = 0; i < weights.size(); i++) 
    {
        for (int j = 0; j < weights[i].size(); j++) 
        {
            weights[i][j] -= learningRate * weightGradients[i][j];
        }
    }
    
    for (int i = 0; i < biases.size(); i++) 
    {
        biases[i] -= learningRate * biasGradients[i];
    }
}
