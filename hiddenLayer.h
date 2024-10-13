#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H
#include <vector>
using namespace std;
class HiddenLayer
{
    public:
        HiddenLayer(int inputSize, int outputSize);
        vector<double> propagateForward(const vector<double>& inputData);
        void displayInfoHiddenLayer();
        vector<double> getOutput();

    private:
        int inputSize;
        int outputSize;
        vector<double> bias;
        vector<vector<double>> weights;
        vector<double> output;
        double rectifiedLinearUnit(double num);
};

#endif