CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2

# Default target
all: logic_gates original_main

# JSON-configured logic gates (main program)
logic_gates: logic_gates_main.cpp NN.cpp layer.cpp activation.cpp losses.cpp utils.cpp gates_config.json
	$(CXX) $(CXXFLAGS) -o logic_gates logic_gates_main.cpp

# Original XOR example
original_main: main.cpp NN.cpp layer.cpp activation.cpp losses.cpp utils.cpp
	$(CXX) $(CXXFLAGS) -o original_main main.cpp

# Clean build files
clean:
	rm -f logic_gates original_main *.exe

# Run main program
run: logic_gates
	./logic_gates

# Run original example
run-original: original_main
	./original_main

.PHONY: all clean run run-original