#include <cmath>
#include <vector>

// Sigmoid activation function
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoidDerivative(double x) {
    return exp(x) / pow((exp(x) + 1), 2);
}

std::vector<double> vectSigmoid(const std::vector<double> x) {
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(sigmoid(i));
    return result;
}

std::vector<double> vectSigmoidDerivative(const std::vector<double> x) {
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(sigmoidDerivative(i));
    return result;
}

// ReLU activation function
double relu(double x) {
    return (x > 0) ? x : 0;
}

double reluDerivative(double x) {
    return (x >= 0) ? 1 : 0;
}

std::vector<double> vectRelu(const std::vector<double> x) {
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(relu(i));
    return result;
}

std::vector<double> vectReluDerivative(const std::vector<double> x) {
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(reluDerivative(i));
    return result;
}