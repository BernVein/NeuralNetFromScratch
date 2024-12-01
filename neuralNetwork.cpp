#include "InputLayer.h"
#include "hiddenLayer.h"
#include "outputLayer.h"
#include <iostream>
#include <vector>
#include <cstdlib> 
#include <algorithm>
#include <chrono>
#include <windows.h>

using namespace std;

double calculateLossPerEpoch(double costAllImage, int trainingDataCount) {
    return costAllImage / static_cast<double>(trainingDataCount);
}

void drawGrid(HANDLE hConsole, const vector<vector<int>>& gridState, int cursorX, int cursorY) {
    COORD coord;
    for (int i = 0; i < gridState.size(); ++i) {
        for (int j = 0; j < gridState[i].size(); ++j) {
            coord.X = j * 2;
            coord.Y = i;
            SetConsoleCursorPosition(hConsole, coord);
            if (i == cursorY && j == cursorX) {
                SetConsoleTextAttribute(hConsole, BACKGROUND_BLUE | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
            } else {
                SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
            }
            cout << (gridState[i][j] ? "■" : "□");
            
        }
       
    }
    cout << "\nPress Enter to add/remove pixel, Press C to predict."<< endl;
    SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
}

vector<double> getUserInput(vector<vector<int>>& gridState) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    int x = 0, y = 0;
    drawGrid(hConsole, gridState, x, y);

    while (true) {
        if (GetAsyncKeyState(VK_UP) & 0x8000) {
            y = (y > 0) ? y - 1 : 2;
            drawGrid(hConsole, gridState, x, y);
            Sleep(150);
        }
        if (GetAsyncKeyState(VK_DOWN) & 0x8000) {
            y = (y < 2) ? y + 1 : 0;
            drawGrid(hConsole, gridState, x, y);
            Sleep(150);
        }
        if (GetAsyncKeyState(VK_LEFT) & 0x8000) {
            x = (x > 0) ? x - 1 : 2;
            drawGrid(hConsole, gridState, x, y);
            Sleep(150);
        }
        if (GetAsyncKeyState(VK_RIGHT) & 0x8000) {
            x = (x < 2) ? x + 1 : 0;
            drawGrid(hConsole, gridState, x, y);
            Sleep(150);
        }
        if (GetAsyncKeyState(VK_RETURN) & 0x8000) {
            gridState[y][x] = !gridState[y][x];  // Toggle the state
            drawGrid(hConsole, gridState, x, y);
            Sleep(150);
        }
        if (GetAsyncKeyState('C') & 0x8000) {
            break;
        }
    }

    vector<double> inputData;
    for (const auto& row : gridState) {
        for (int cell : row) {
            inputData.push_back(static_cast<double>(cell));
        }
    }
    return inputData;
}

int main() {
    system("cls");
    SetConsoleOutputCP(CP_UTF8);
    vector<vector<double>> trainingImages = {
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
        {1, 0, 0, 0, 1, 0, 0, 0, 0}  // D
    };
    vector<vector<double>> targetOutputs = {
        {0, 0, 1}, // H
        {0, 1, 0}, // V
        {0, 1, 0}, // V
        {0, 0, 1}, // H    
        {0, 0, 1}, // H
        {1, 0, 0}, // D
        {0, 0, 1}, // H
        {1, 0, 0}, // D
        {0, 0, 1}, // H
        {1, 0, 0}, // D
        {0, 0, 1}, // H
        {0, 1, 0}, // V
        {0, 0, 1}, // H
        {0, 1, 0}, // V
        {0, 1, 0}, // V
        {0, 1, 0}, // V
        {0, 1, 0}, // V
        {0, 1, 0}, // V
        {1, 0, 0}, // D
        {0, 0, 1}, // H
        {1, 0, 0}, // D
        {1, 0, 0}, // D
        {0, 0, 1}, // H
        {1, 0, 0}, // D
        {0, 1, 0}, // V
        {1, 0, 0}, // D
        {1, 0, 0}, // D
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
    while(finalCost >= threshold) {
        // Loop again until the cost is acceptable
        for (int epoch = 0; epoch < epochs; epoch++) {
            double cost = 0.0;
            // Step 2: For EACH training image
            for(int i = 0; i < trainingImages.size(); i++) {
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
                hiddenLayer.updateBias(learningRate);
            }
            
            finalCost = calculateLossPerEpoch(cost, trainingImages.size());
            if(trainCount == 0 && epoch == 0) firstLoss = finalCost;
        }
        trainCount++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // cout << "Trained " << trainCount << " time/s to reach <=" << threshold << " cost threshold." << endl;
    // cout << "Training completed in " << elapsed.count() << " second/s." << endl;
    // cout << "Initial Cost: " << firstLoss << endl << "Final Cost for final training: " << finalCost << endl << endl;

    vector<vector<int>> gridState(3, vector<int>(3, 0)); // Initialid state

    while (true) {
        vector<double> userInput = getUserInput(gridState);
        inputLayer.setInputData(userInput);
        hiddenLayer.propagateForward(inputLayer.getInputData());
        outputLayer.propagateForward(hiddenLayer.getOutput());
        vector<double> prediction = outputLayer.predict();

        string res;
        if (prediction[0] == 0) res = "Diagonal";
        else if (prediction[0] == 1) res = "Vertical";
        else res = "Horizontal";

        cout << "Your input is " << prediction[1] * 100<< "% a " << res << "." << endl;

        Sleep(500); 
    }
    return 0;
}
