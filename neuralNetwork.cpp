#include "InputLayer.h"
#include "hiddenLayer.h"
#include "outputLayer.h"
#include <iostream>
#include <vector>
#include <thread> // For sleep functionality
#include <chrono> // For duration
#include <cstdlib> // For system()
#include <unistd.h> 
#include <algorithm>
using namespace std;

vector<vector<double>> weightGradientAverage(const vector<vector<double>>& weightGradients, int trainingDataCount);
vector<double> biasGradientAverage(vector<vector<double>> allDeltas, int trainingDataCount);
void updateWeightsAndBiases(vector<vector<double>>& weights, 
                            vector<double>& biases, 
                            vector<vector<double>>& weightGradients, 
                            vector<double>& biasGradients, 
                            double learningRate);

double calculateLossPerEpoch(double costAllImage, int trainingDataCount){return costAllImage / static_cast<double>(trainingDataCount);}
double calculateAvgWeightGradientPerEpoch(double weightGradientsTotal, int trainingDataCount){return weightGradientsTotal / static_cast<double>(trainingDataCount);}
double calculateAvgBiasGradientPerEpoch(double biasGradientsTotal, int trainingDataCount){return biasGradientsTotal / static_cast<double>(trainingDataCount);}


int main()
{

    vector<vector<double>> trainingImages = 
    {
        {1, 1, 1, 0, 0, 0, 0, 0, 0}, // H
        {1, 0, 0, 1, 0, 0, 0, 0, 0}, // V
        {0, 0, 0, 0, 0, 1, 0, 0, 1}, // V
        {0, 0, 0, 0, 0, 0, 1, 1, 1}, // H 
        {1, 1, 0, 0, 0, 0, 0, 0, 0}, // H
        {0, 0, 0, 0, 0, 1, 0, 1, 0}, // D
        {0, 1, 1, 0, 0, 0, 0, 0, 0}, // H
        {0, 0, 0, 1, 0, 0, 0, 1, 0}, // D
        {0, 0, 0, 1, 1, 0, 0, 0, 0}, // H
        {0, 0, 1, 0, 1, 0, 1, 0, 0}, // D
        {0, 0, 0, 0, 1, 1, 0, 0, 0}, // H
        {0, 1, 0, 0, 1, 0, 0, 0, 0}, // V
        {0, 0, 0, 0, 0, 0, 1, 1, 0}, // H
        {0, 1, 0, 0, 1, 0, 0, 1, 0}, // V
        {0, 0, 1, 0, 0, 1, 0, 0, 1}, // V
        {0, 0, 0, 1, 0, 0, 1, 0, 0}, // V
        {0, 0, 0, 0, 1, 0, 0, 1, 0}, // V
        {0, 0, 1, 0, 0, 1, 0, 0, 0}, // V
        {1, 0, 0, 0, 1, 0, 0, 0, 1}, // D
        {0, 0, 0, 1, 1, 1, 0, 0, 0}, // H
        {0, 1, 0, 0, 0, 1, 0, 0, 0}, // D
        {0, 1, 0, 1, 0, 0, 0, 0, 0}, // D
        {0, 0, 0, 0, 0, 0, 0, 1, 1}, // H
        {0, 0, 0, 0, 1, 0, 0, 0, 1}, // D
        {1, 0, 0, 1, 0, 0, 1, 0, 0}, // V
        {0, 0, 1, 0, 1, 0, 0, 0, 0}  // D
    };
    vector<vector<double>> targetOutputs = 
    {
        {0,0,1}, //H
        {0,1,0}, //V
        {0,1,0}, //V
        {0,0,1}, //H    
        {0,0,1}, //H

        {1,0,0}, //D
        {0,0,1}, //H
        {1,0,0}, //D
        {0,0,1}, //H
        {1,0,0}, //D
        {0,0,1}, //H

        {0,1,0}, //V
        {0,0,1}, //H
        {0,1,0}, //V
        {0,1,0}, //V
        {0,1,0}, //V
        {0,1,0}, //V
        {0,1,0}, //V

        {1,0,0}, //D
        {0,0,1}, //H
        {1,0,0}, //D
        {1,0,0}, //D

        {0,0,1}, //H
        {1,0,0}, //D
        {0,1,0}, //V
        {1,0,0}, //D

    };
    const int numTrainingExamples = trainingImages.size();
    const double learningRate = 0.1; 
    const int epochs = 10000; 

    InputLayer inputLayer(trainingImages[0].size());
    HiddenLayer hiddenLayer(trainingImages[0].size(), 5);
    OutputLayer outputLayer(hiddenLayer.getOutput().size(), 3);
    
    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        double cost = 0.0;
        double gradientWeightGradientTotal = 0.0;
        double gradientBiasTotal = 0.0;
        // For EACH training image
        for(int i = 0; i < numTrainingExamples; i++)
        {
           
            // Forward Pass   
            inputLayer.setInputData(trainingImages[i]);
            hiddenLayer.propagateForward(inputLayer.getInputData());
            outputLayer.propagateForward(hiddenLayer.getOutput());
            outputLayer.setSoftmax();
             
            // Backpropagation
            // Calculate cost PER image
            cost += outputLayer.meanSquaredErrorCostPerImage(targetOutputs[i]);
            // Calculate delta for outputLayer
            outputLayer.deltaForOutputNeurons(targetOutputs[i]);

            // Calculate delta for subsequent previous layer
            hiddenLayer.calculateDelta(outputLayer.getDeltas(), outputLayer.getWeights());
            
            // Calculate Weight Gradients w^L_(j,i) and bias of Neuron^L_j
            outputLayer.calculateGradientsWeight(hiddenLayer.getOutput());
            outputLayer.calculateGradientsBias();

            // Calculate Calculate Weight Gradients w^(L-1)_(j,i) and bias of Neuron^(L-1)_j
            hiddenLayer.calculateGradientsWeight(inputLayer.getInputData());
            hiddenLayer.calculateGradientsBias();
            
            // Gather total gradient weight and gradient bias for all network
            gradientWeightGradientTotal += outputLayer.calculateTotalWeightGradients() + hiddenLayer.calculateTotalWeightGradients();
            gradientBiasTotal += outputLayer.calculateTotalBiasGradients() + hiddenLayer.calculateTotalBiasGradients();
        }
        
        // Average of gradient weight and gradient bias for all entire training image
        double averageWeightGradients = calculateAvgWeightGradientPerEpoch(gradientWeightGradientTotal, trainingImages.size());
        double averageBiasGradients = calculateAvgBiasGradientPerEpoch(gradientBiasTotal, trainingImages.size());

        // Update the weights and biases based on the averaged weight and bias gradients for each layer
        outputLayer.updateWeights(learningRate, averageWeightGradients);
        outputLayer.updateBias(learningRate, averageBiasGradients);
        hiddenLayer.updateWeights(learningRate, averageWeightGradients);
        hiddenLayer.updateBias(learningRate, averageBiasGradients);

        // Calculate and display cost per epoch
        cout << "Cost for epoch " << epoch << ": " << calculateLossPerEpoch(cost, trainingImages.size()) << endl;
        system("CLS");
    }
    cout << "TRAINING DONE!" << endl;

    vector<vector<double>> testImages = 
    {
        {0, 0, 0, 0, 0, 0, 1, 1, 1}, // H 
        {0, 1, 0, 0, 1, 0, 0, 0, 0}, // V
        {0, 0, 0, 1, 1, 1, 0, 0, 0}, // H
        {0, 1, 0, 0, 0, 1, 0, 0, 0}, // D
        {0, 1, 0, 1, 0, 0, 0, 0, 0}, // D
    };
    
    vector<vector<double>> testLabels = 
    {
        {0, 0, 1}, // H
        {0, 1, 0}, // V
        {0, 0, 1}, // H
        {1, 0, 0}, // D
        {1, 0, 0}  // D
    };
    return 0;
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