#include "InputLayer.h"

InputLayer::InputLayer(int inputSize) : size(inputSize) {inputData.resize(inputSize);}
void InputLayer::setInputData(const vector<double>& data)
{
    if(data.size() != size)
    {
        cout << "Size mismatch." << endl;
        return;
    }
    inputData = data;
}

const vector<double>& InputLayer::getInputData() const{return inputData;}