#ifndef INPUTLAYER_H
#define INPUTLAYER_H
#include <vector>
#include <iostream>
using namespace std;

class InputLayer
{
    public:
        InputLayer(int inputSize);
        void setInputData(const vector<double>& data);
        const vector<double>& getInputData() const;
    private:
        int size;
        vector<double> inputData;
};

#endif