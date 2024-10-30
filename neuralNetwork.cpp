#include "InputLayer.h"
#include "hiddenLayer.h"
#include "outputLayer.h"
#include <iostream>
#include <vector>
#include <cstdlib> 
#include <algorithm>
#include <stdexcept> 
using namespace std;

vector<vector<double>> weightGradientAverage(const vector<vector<double>>& weightGradients, int trainingDataCount);
vector<double> biasGradientAverage(vector<vector<double>> allDeltas, int trainingDataCount);
double calculateLossPerEpoch(double costAllImage, int trainingDataCount){return costAllImage / static_cast<double>(trainingDataCount);}
double calculateAvgWeightGradientPerEpoch(double weightGradientsTotal, int trainingDataCount){return weightGradientsTotal / static_cast<double>(trainingDataCount);}
double calculateAvgBiasGradientPerEpoch(double biasGradientsTotal, int trainingDataCount){return biasGradientsTotal / static_cast<double>(trainingDataCount);}

int getMaxValueIndex(const std::vector<double>& data)
{
    if (data.empty())
    {
        throw std::invalid_argument("Vector is empty.");
    }

    // Use std::max_element to find the iterator to the max element
    auto maxIter = std::max_element(data.begin(), data.end());

    // Calculate the index by subtracting the iterator positions
    return std::distance(data.begin(), maxIter);
}

int main()
{
    system("CLS");
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
    const double learningRate = 0.05; 
    const int epochs = 10000; 
    double firstLoss = 0.0;

    InputLayer inputLayer(trainingImages[0].size());
    HiddenLayer hiddenLayer(trainingImages[0].size(), 5);
    OutputLayer outputLayer(5, 3);

    double finalCost = 0.0;
    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        double cost = 0.0;
        // For EACH training image
        for(int i = 0; i < trainingImages.size(); i++)
        {
            // Forward Pass 
            inputLayer.setInputData(trainingImages[i]);
            hiddenLayer.propagateForward(inputLayer.getInputData());
            outputLayer.propagateForward(hiddenLayer.getOutput());

            // Backpropagation
            // Calculate cost PER image
            cost += outputLayer.meanSquaredErrorCostPerImage(targetOutputs[i]);

            // Calculate delta for outputLayer
            outputLayer.deltaForOutputNeurons(targetOutputs[i]);
            //outputLayer.displayInfoOutputLayer();

            // Calculate delta for subsequent previous layer
            hiddenLayer.calculateDelta(outputLayer.getDeltas(), outputLayer.getWeights());

            // Calculate Weight Gradients w^L_(j,i) and bias of Neuron^L_j
            outputLayer.calculateGradientsWeight(hiddenLayer.getOutput());
            outputLayer.calculateGradientsBias();

            // Calculate Calculate Weight Gradients w^(L-1)_(j,i) and bias of Neuron^(L-1)_j
            hiddenLayer.calculateGradientsWeight(inputLayer.getInputData());
            hiddenLayer.calculateGradientsBias();

            // Update the weights and biases based on the averaged weight and bias gradients for each layer
            outputLayer.updateWeights(learningRate);
            outputLayer.updateBias(learningRate);

            hiddenLayer.updateWeights(learningRate);
            hiddenLayer.updateBias(learningRate);;
        }

        // Calculate and display cost per epoch
        finalCost = calculateLossPerEpoch(cost, trainingImages.size());
        if(epoch == 0) firstLoss = finalCost;

        //cout << "Epoch " << epoch << ": " << finalCost << endl; 
    }
    cout << "TRAINING DONE!" << endl;
    cout << "First Cost: " << firstLoss << " Final Cost: " << finalCost << endl;
    vector<vector<double>> testImages = 
    {
        {1,0,1,0,1,0,0,0,0}
    };
    
    vector<vector<double>> testLabels = 
    {
        {1,0,0}
    };
    
    cout << "Testing..." << endl;
    for(int i = 0; i < testImages.size(); i++)
    {
        inputLayer.setInputData(testImages[i]);
        hiddenLayer.propagateForward(inputLayer.getInputData());
        outputLayer.propagateForward(hiddenLayer.getOutput());
        cout << getMaxValueIndex(outputLayer.getOutput()) << endl;
    }


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
