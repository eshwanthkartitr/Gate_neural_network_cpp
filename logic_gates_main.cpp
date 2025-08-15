#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "NN.cpp"

struct GateConfig {
    std::string name;
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;
    int epochs;
    double learning_rate;
};

class SimpleJSONParser {
public:
    static std::vector<GateConfig> parseGatesConfig(const std::string& filename) {
        std::vector<GateConfig> configs;
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            std::cout << "Error: Could not open " << filename << std::endl;
            return configs;
        }

        std::string line;
        GateConfig current_config;
        bool in_gate = false;
        bool in_inputs = false;
        bool in_outputs = false;
        
        while (std::getline(file, line)) {
            // Remove whitespace
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t") + 1);
            
            if (line.find("\"name\":") != std::string::npos) {
                size_t start = line.find("\"", line.find(":") + 1) + 1;
                size_t end = line.find("\"", start);
                current_config.name = line.substr(start, end - start);
                in_gate = true;
            }
            else if (line.find("\"epochs\":") != std::string::npos) {
                size_t start = line.find(":") + 1;
                std::string value = line.substr(start);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t,") + 1);
                current_config.epochs = std::stoi(value);
            }
            else if (line.find("\"learning_rate\":") != std::string::npos) {
                size_t start = line.find(":") + 1;
                std::string value = line.substr(start);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t,") + 1);
                current_config.learning_rate = std::stod(value);
            }
            else if (line.find("\"inputs\":") != std::string::npos) {
                in_inputs = true;
                in_outputs = false;
                current_config.inputs.clear();
            }
            else if (line.find("\"outputs\":") != std::string::npos) {
                in_outputs = true;
                in_inputs = false;
                current_config.outputs.clear();
            }
            else if (in_inputs && line.find("[") != std::string::npos && line.find("]") != std::string::npos) {
                std::vector<double> input_row = parseArray(line);
                if (!input_row.empty()) {
                    current_config.inputs.push_back(input_row);
                }
            }
            else if (in_outputs && line.find("[") != std::string::npos && line.find("]") != std::string::npos) {
                std::vector<double> output_row = parseArray(line);
                if (!output_row.empty()) {
                    current_config.outputs.push_back(output_row);
                }
            }
            else if (line.find("}") != std::string::npos && in_gate && !current_config.name.empty()) {
                in_outputs = false;
                in_inputs = false;
                configs.push_back(current_config);
                current_config = GateConfig();
                in_gate = false;
            }
        }
        
        file.close();
        return configs;
    }

private:
    static std::vector<double> parseArray(const std::string& line) {
        std::vector<double> result;
        size_t start = line.find("[") + 1;
        size_t end = line.find("]");
        if (start == std::string::npos || end == std::string::npos) return result;
        
        std::string array_content = line.substr(start, end - start);
        std::stringstream ss(array_content);
        std::string item;
        
        while (std::getline(ss, item, ',')) {
            item.erase(0, item.find_first_not_of(" \t"));
            item.erase(item.find_last_not_of(" \t") + 1);
            if (!item.empty()) {
                result.push_back(std::stod(item));
            }
        }
        
        return result;
    }
};

class LogicGateTrainer {
private:
    std::map<std::string, std::vector<std::vector<double>>> gate_inputs;
    std::map<std::string, std::vector<std::vector<double>>> gate_outputs;
    std::map<std::string, NN> trained_networks;
    std::vector<GateConfig> gate_configs;

public:
    LogicGateTrainer(const std::string& config_file = "gates_config.json") {
        loadGatesFromJSON(config_file);
    }

    void loadGatesFromJSON(const std::string& config_file) {
        gate_configs = SimpleJSONParser::parseGatesConfig(config_file);
        
        if (gate_configs.empty()) {
            std::cout << "No gates loaded from config file!" << std::endl;
            return;
        }
        
        std::cout << "Loaded " << gate_configs.size() << " gates from " << config_file << std::endl;
        
        // Store in maps for easy access
        for (const auto& config : gate_configs) {
            gate_inputs[config.name] = config.inputs;
            gate_outputs[config.name] = config.outputs;
        }
    }

    NN createNetwork(const std::string& gate_type) {
        NN network;
        
        if (gate_type == "NOT") {
            // NOT gate needs single input
            network.add(new Linear(1, 3));
            network.add(new Relu());
            network.add(new Linear(3, 1));
            network.add(new Sigmoid());
        } else {
            // Other gates need two inputs
            network.add(new Linear(2, 4));
            network.add(new Relu());
            network.add(new Linear(4, 1));
            network.add(new Sigmoid());
        }
        
        return network;
    }

    void trainGate(const GateConfig& config) {
        std::cout << "\n=== Training " << config.name << " Gate ===" << std::endl;
        std::cout << "Epochs: " << config.epochs << ", Learning Rate: " << config.learning_rate << std::endl;
        
        NN network = createNetwork(config.name);
        network.fit(config.inputs, config.outputs, config.epochs, config.learning_rate);
        
        trained_networks[config.name] = std::move(network);
        
        std::cout << config.name << " Gate training completed!" << std::endl;
    }

    void testGate(const std::string& gate_type) {
        if (trained_networks.find(gate_type) == trained_networks.end()) {
            std::cout << "Error: " << gate_type << " gate not trained yet!" << std::endl;
            return;
        }

        std::cout << "\n=== Testing " << gate_type << " Gate ===" << std::endl;
        std::cout << "Input -> Output (Probability) -> Predicted -> Expected" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        auto& network = trained_networks[gate_type];
        auto& inputs = gate_inputs[gate_type];
        auto& expected_outputs = gate_outputs[gate_type];

        int correct = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<double> output_prob = network.predict(inputs[i]);
            int predicted = (output_prob[0] > 0.5) ? 1 : 0;
            bool is_correct = (predicted == (int)expected_outputs[i][0]);
            if (is_correct) correct++;
            
            // Print input
            if (gate_type == "NOT") {
                std::cout << inputs[i][0];
            } else {
                std::cout << inputs[i][0] << "," << inputs[i][1];
            }
            
            std::cout << " -> " << std::fixed << std::setprecision(4) << output_prob[0] 
                      << " -> " << predicted 
                      << " -> " << (int)expected_outputs[i][0]
                      << (is_correct ? " [OK]" : " [FAIL]") << std::endl;
        }
        
        std::cout << "Accuracy: " << correct << "/" << inputs.size() 
                  << " (" << (100.0 * correct / inputs.size()) << "%)" << std::endl;
    }

    void visualizeHyperplane(const std::string& gate_type) {
        if (trained_networks.find(gate_type) == trained_networks.end()) {
            std::cout << "Error: " << gate_type << " gate not trained yet!" << std::endl;
            return;
        }

        if (gate_type == "NOT") {
            std::cout << "Hyperplane visualization not applicable for NOT gate (1D input)" << std::endl;
            return;
        }

        std::cout << "\n=== Hyperplane Visualization for " << gate_type << " Gate ===" << std::endl;
        std::cout << "Decision boundary (0=blue, 1=red):" << std::endl;
        std::cout << "   ";
        
        // Print column headers
        for (double x1 = 0.0; x1 <= 1.0; x1 += 0.2) {
            std::cout << std::fixed << std::setprecision(1) << x1 << " ";
        }
        std::cout << std::endl;

        auto& network = trained_networks[gate_type];
        
        for (double x2 = 1.0; x2 >= 0.0; x2 -= 0.2) {
            std::cout << std::fixed << std::setprecision(1) << x2 << " ";
            
            for (double x1 = 0.0; x1 <= 1.0; x1 += 0.2) {
                std::vector<double> input = {x1, x2};
                std::vector<double> output = network.predict(input);
                
                if (output[0] > 0.5) {
                    std::cout << "1 ";
                } else {
                    std::cout << "0 ";
                }
            }
            std::cout << std::endl;
        }
    }

    void trainAllGates() {
        if (gate_configs.empty()) {
            std::cout << "ERROR: No gate configurations loaded!" << std::endl;
            return;
        }
        
        std::cout << "Training All Logic Gates from JSON..." << std::endl;
        
        for (const auto& config : gate_configs) {
            trainGate(config);
            testGate(config.name);
            
            // Show hyperplane for 2D gates
            if (config.name != "NOT") {
                visualizeHyperplane(config.name);
            }
        }
        
        printSummary();
    }

private:
    void printSummary() {
        std::cout << "\n=== Training Complete! ===" << std::endl;
        std::cout << "Successfully trained " << trained_networks.size() << " logic gates:" << std::endl;
        
        for (const auto& pair : trained_networks) {
            std::cout << "- " << pair.first << " Gate" << std::endl;
        }
        
        std::cout << "\nTips:" << std::endl;
        std::cout << "- Edit gates_config.json to adjust epochs/learning rates" << std::endl;
        std::cout << "- Modify network architecture in createNetwork()" << std::endl;
        std::cout << "- Add custom gates to gates_config.json" << std::endl;
    }
};

int main() {
    LogicGateTrainer trainer;
    trainer.trainAllGates();
    return 0;
}