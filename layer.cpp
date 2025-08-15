#include <vector>
#include "activation.cpp"
#include "utils.cpp"

// Base Layer class
class Layer {
public:
    mutable std::vector<double> input;
    mutable std::vector<double> output;
    virtual std::vector<double> forward(const std::vector<double> input_data) const = 0;
    virtual std::vector<double> backward(std::vector<double> error, double learning_rate) = 0;
};

// Sigmoid activation layer
class Sigmoid : public Layer {
public:
    std::vector<double> forward(const std::vector<double> input_data) const override {
        input = input_data;
        output = vectSigmoid(input);
        return output;
    }
    
    std::vector<double> backward(std::vector<double> error, double learning_rate) override {
        std::vector<double> derivative = vectSigmoidDerivative(input);
        std::vector<double> grad_input;
        for (size_t i = 0; i < derivative.size(); ++i) {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }
};

// ReLU activation layer
class Relu : public Layer {
public:
    std::vector<double> forward(const std::vector<double> input_data) const override {
        input = input_data;
        output = vectRelu(input);
        return output;
    }
    
    std::vector<double> backward(std::vector<double> error, double learning_rate) override {
        std::vector<double> derivative = vectReluDerivative(input);
        std::vector<double> grad_input;
        for (size_t i = 0; i < derivative.size(); ++i) {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }
};

// Linear (fully connected) layer
class Linear : public Layer {
public:
    int input_neuron;
    int output_neuron;
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;

    Linear(int num_in, int num_out) {
        input_neuron = num_in;
        output_neuron = num_out;
        weights = uniformWeightInitializer(num_out, num_in);
        bias = biasInitailizer(num_out);
    }

    std::vector<double> forward(const std::vector<double> input_data) const override {
        input = input_data;
        output.clear();
        for (int i = 0; i < output_neuron; i++) {
            output.push_back(dotProduct(const_cast<std::vector<double>&>(weights[i]), 
                                      const_cast<std::vector<double>&>(input)) + bias[i]);
        }
        return output;
    }
    
    std::vector<double> backward(std::vector<double> error, double learning_rate) override {
        std::vector<double> input_error;
        std::vector<std::vector<double>> weight_transpose = transpose(weights);
        
        // Calculate input error
        for (size_t i = 0; i < weight_transpose.size(); i++) {
            input_error.push_back(dotProduct(weight_transpose[i], error));
        }
        
        // Update weights and biases
        for (size_t j = 0; j < error.size(); j++) {
            for (size_t i = 0; i < input.size(); i++) {
                weights[j][i] -= learning_rate * error[j] * input[i];
            }
            bias[j] -= learning_rate * error[j];
        }

        return input_error;
    }
};