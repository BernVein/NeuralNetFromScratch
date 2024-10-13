#include "InputLayer.h"
#include "hiddenLayer.h"
#include "outputLayer.h"
#include <iostream>
#include <vector>
using namespace std;

void displayVectorContent(vector <double> vec)
{
    for(double num : vec)
        cout << num << ", ";
}

int main()
{
    vector<double> horizontalImage = 
    {
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    };
    vector<double> horizontalImageActual = {0, 0, 1};

    // Forward propagation
    InputLayer inputLayer(horizontalImage.size());
    inputLayer.setInputData(horizontalImage);
    cout << "Inputs from input Layer:" << endl;
    displayVectorContent(inputLayer.getInputData());
    cout << endl;
    HiddenLayer hiddenLayer(horizontalImage.size(), 5);
    hiddenLayer.propagateForward(inputLayer.getInputData());
    hiddenLayer.displayInfoHiddenLayer();

    OutputLayer outputLayer(hiddenLayer.getOutput().size(), 3);
    outputLayer.propagateForward(hiddenLayer.getOutput());
    outputLayer.displayInfoOutputLayer();

    // Backpropagation
    outputLayer.calculateErrorDelta(horizontalImageActual);
    outputLayer.propagateBackward(0.01, hiddenLayer.getOutput());
    outputLayer.displayInfoOutputLayer();
}

