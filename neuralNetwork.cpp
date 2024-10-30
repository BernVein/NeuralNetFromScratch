#include "InputLayer.h"
#include "hiddenLayer.h"
#include "outputLayer.h"
#include <iostream>
#include <vector>
#include <cstdlib> 
#include <algorithm>
#include <chrono>
using namespace std;

double calculateLossPerEpoch(double costAllImage, int trainingDataCount){return costAllImage / static_cast<double>(trainingDataCount);}

int main()
{  
    system("cls");
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
        {0, 0, 1, 0, 1, 0, 0, 0, 0}, // D
        {1, 0, 0, 0, 1, 0, 0, 0, 0} //D
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
        {1,0,0}, //D
    };

    const int numTrainingExamples = trainingImages.size();
    const double learningRate = 0.1; 
    const int epochs = 2000;
    double firstLoss = 0.0;

    int hiddenLayerNeurons = 10;
    int classifications = 3;
    InputLayer inputLayer(trainingImages[0].size());
    HiddenLayer hiddenLayer(trainingImages[0].size(), hiddenLayerNeurons);
    OutputLayer outputLayer(hiddenLayerNeurons, classifications);

    auto start = std::chrono::high_resolution_clock::now();

    int trainCount = 0;
    double finalCost = 1.0;
    while(finalCost >= 0.001)
    {
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
                // Calculate cost PER image
                cost += outputLayer.meanSquaredErrorCostPerImage(targetOutputs[i]);

                // Backpropagation
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

                // Update the weights and biases
                outputLayer.updateWeights(learningRate);
                outputLayer.updateBias(learningRate);
                hiddenLayer.updateWeights(learningRate);
                hiddenLayer.updateBias(learningRate);;
            }
            
            finalCost = calculateLossPerEpoch(cost, trainingImages.size());
            if(trainCount == 0 && epoch == 0) firstLoss = finalCost;
        }
        trainCount++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Train repeated " << trainCount << " time/s. " << endl;
    cout << "Training completed in " << elapsed.count() << " seconds." << endl;
    cout << "Initial Cost: " << firstLoss << endl << "Final Cost for final training: " << finalCost << endl << endl;
    vector<vector<double>> testImages = 
    {
        {1,0,0,
         0,0,0,
         1,0,0},

        {1,1,1,
         0,0,0,
         0,0,1},

        {0,0,0,
         1,0,0,
         0,1,0},

        {1,0,0,
         0,1,0,
         0,0,1},

        {0,1,0,
         0,1,1,
         0,1,0},

        {1,1,0,
         0,0,0,
         1,1,1}
    };
    
    for(int i = 0; i < testImages.size(); i++)
    {
        inputLayer.setInputData(testImages[i]);
        hiddenLayer.propagateForward(inputLayer.getInputData());
        outputLayer.propagateForward(hiddenLayer.getOutput());
        cout << "Image " << i + 1 << ": ";
        outputLayer.predict();
    }
    cout << endl << endl;
}



