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
    const int epochs = 1000;
    double firstLoss = 0.0;
    double threshold = .001;
    int hiddenLayerNeurons = 10;
    int classifications = 3;
    // Step 1: Initialize network with random weights and biases
    InputLayer inputLayer(trainingImages[0].size());
    HiddenLayer hiddenLayer(trainingImages[0].size(), hiddenLayerNeurons);
    OutputLayer outputLayer(hiddenLayerNeurons, classifications);
    
    auto start = std::chrono::high_resolution_clock::now();

    int trainCount = 0;
    double finalCost = 1.0;
    while(finalCost >= threshold)
    {
        // Loop again until the cost is acceptable
        for (int epoch = 0; epoch < epochs; epoch++) 
        {
            double cost = 0.0;
            // Step 2: For EACH training image
            for(int i = 0; i < trainingImages.size(); i++)
            {
                // Step 2a: Forward Pass
                inputLayer.setInputData(trainingImages[i]);
                hiddenLayer.propagateForward(inputLayer.getInputData());
                outputLayer.propagateForward(hiddenLayer.getOutput());

                // Step 2b: Calculate cost PER image
                cost += outputLayer.meanSquaredErrorCostPerImage(targetOutputs[i]);

                // Backpropagation
                // Step 2c: Calculate delta for outputLayer
                outputLayer.deltaForOutputNeurons(targetOutputs[i]);

                // Step 2d: Calculate delta for previous layer
                hiddenLayer.calculateDelta(outputLayer.getDeltas(), outputLayer.getWeights());

                // Step 2e: Calculate gradients for each weight and biases for the network
                // Calculate Weight Gradients w^L_(j,i) and bias of Neuron^L_j
                outputLayer.calculateGradientsWeight(hiddenLayer.getOutput());
                outputLayer.calculateGradientsBias();
                // Calculate Calculate Weight Gradients w^(L-1)_(j,i) and bias of Neuron^(L-1)_j
                hiddenLayer.calculateGradientsWeight(inputLayer.getInputData());
                hiddenLayer.calculateGradientsBias();

                // Step 2f: Update the weights and biases using Gradient Descent(?)
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

    cout << "Trained " << trainCount << " time/s to reach <=" << threshold << " cost threshold." << endl;
    cout << "Training completed in " << elapsed.count() << " second/s." << endl;
    cout << "Initial Cost: " << firstLoss << endl << "Final Cost for final training: " << finalCost << endl << endl;
    vector<vector<double>> testImages = 
    {
        // 1 - 2
        {1,0,0,
         1,0,0,
         1,0,0},
        // 2 - 0
        {0,1,1,
         0,1,0,
         1,0,0},
        // 3 - 1
        {0,0,1,
         1,0,1,
         0,0,1},
        // 4 - 2
        {0,1,0,
         0,0,0,
         0,1,1},
        // 5- 2
        {1,1,0,
         0,0,0,
         1,0,1},
        // 6 - 1
        {1,0,0,
         0,1,0,
         0,1,0},
        // 7 - 0
        {1,0,0,
         0,1,1,
         0,0,1},
        // 8 - 1
        {1,0,1,
         1,0,1,
         0,0,1},
        // 9 - 2
        {1,1,0,
         1,1,1,
         0,0,0},
        // 10 - 0
        {1,0,1,
         0,1,0,
         1,1,0},
    };
    // Diagonal = 0, Vertical = 1, Horizontal = 2
    vector<int> expectedOutput = {1, 0, 1, 2, 2, 1, 0, 1, 2, 0};
    int correctGuesses = 0;
    for(int i = 0; i < testImages.size(); i++)
    {
        inputLayer.setInputData(testImages[i]);
        hiddenLayer.propagateForward(inputLayer.getInputData());
        outputLayer.propagateForward(hiddenLayer.getOutput());
        vector<double> prediction = outputLayer.predict();
        string res = "";
        if(prediction[0] == 0) res = "Diagonal";
        else if(prediction[0] == 1) res = "Vertical";
        else res = "Horizontal";
        cout << "Image " << i + 1 << " is probably a " << res << " with a confidence of " << prediction[1] * 100 << "%." << endl;
        if(prediction[0] == expectedOutput[i]) correctGuesses++;
    }
    cout << endl;
    cout << "Accuracy: " << (static_cast<double>(correctGuesses) / testImages.size()) * 100 << "%." << endl;
    cout << endl << endl;
}



