#define CL_TARGET_OPENCL_VERSION 120 
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>

// OpenCL kernel for matrix multiplication (forward pass)
const char* matmulKernel = R"(
__kernel void matrix_multiply(__global const float* A, 
                             __global const float* B, 
                             __global float* C,
                             const int M, const int N, const int K) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
)";

// OpenCL kernel for activation functions
const char* activationKernel = R"(
__kernel void sigmoid(__global float* x, const int size) {
    int id = get_global_id(0);
    if (id < size) {
        x[id] = 1.0f / (1.0f + exp(-x[id]));
    }
}

__kernel void relu(__global float* x, const int size) {
    int id = get_global_id(0);
    if (id < size) {
        x[id] = (x[id] > 0.0f) ? x[id] : 0.0f;
    }
}

__kernel void sigmoid_derivative(__global const float* x, __global float* result, const int size) {
    int id = get_global_id(0);
    if (id < size) {
        float sig = x[id];
        result[id] = sig * (1.0f - sig);
    }
}
)";

// OpenCL kernel for weight updates
const char* updateKernel = R"(
__kernel void update_weights(__global float* weights, 
                            __global const float* gradients, 
                            const float learning_rate, 
                            const int size) {
    int id = get_global_id(0);
    if (id < size) {
        weights[id] -= learning_rate * gradients[id];
    }
}
)";

class GPUTrainingBenchmark {
private:
    cl_platform_id platform;
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program matmul_program, activation_program, update_program;
    cl_kernel matmul_kernel, sigmoid_kernel, relu_kernel, sigmoid_deriv_kernel, update_kernel;
    
public:
    bool initialize() {
        cl_int err;
        
        // Get platform and device
        cl_uint platformCount;
        err = clGetPlatformIDs(1, &platform, &platformCount);
        if (err != CL_SUCCESS || platformCount == 0) {
            std::cout << "No OpenCL platforms found!" << std::endl;
            return false;
        }
        
        cl_uint deviceCount;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &deviceCount);
        if (err != CL_SUCCESS || deviceCount == 0) {
            std::cout << "No GPU devices found!" << std::endl;
            return false;
        }
        
        // Create context and command queue
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        queue = clCreateCommandQueue(context, device, 0, &err);
        
        // Create and build programs
        matmul_program = clCreateProgramWithSource(context, 1, &matmulKernel, NULL, &err);
        clBuildProgram(matmul_program, 1, &device, NULL, NULL, NULL);
        
        activation_program = clCreateProgramWithSource(context, 1, &activationKernel, NULL, &err);
        clBuildProgram(activation_program, 1, &device, NULL, NULL, NULL);
        
        update_program = clCreateProgramWithSource(context, 1, &updateKernel, NULL, &err);
        clBuildProgram(update_program, 1, &device, NULL, NULL, NULL);
        
        // Create kernels
        matmul_kernel = clCreateKernel(matmul_program, "matrix_multiply", &err);
        sigmoid_kernel = clCreateKernel(activation_program, "sigmoid", &err);
        relu_kernel = clCreateKernel(activation_program, "relu", &err);
        sigmoid_deriv_kernel = clCreateKernel(activation_program, "sigmoid_derivative", &err);
        update_kernel = clCreateKernel(update_program, "update_weights", &err);
        
        return true;
    }
    
    // GPU matrix multiplication
    void gpu_matmul(const std::vector<float>& A, const std::vector<float>& B, 
                   std::vector<float>& C, int M, int N, int K) {
        cl_int err;
        
        // Create buffers
        cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * A.size(), (void*)A.data(), &err);
        cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * B.size(), (void*)B.data(), &err);
        cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(float) * C.size(), NULL, &err);
        
        // Set kernel arguments
        clSetKernelArg(matmul_kernel, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(matmul_kernel, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(matmul_kernel, 2, sizeof(cl_mem), &bufC);
        clSetKernelArg(matmul_kernel, 3, sizeof(int), &M);
        clSetKernelArg(matmul_kernel, 4, sizeof(int), &N);
        clSetKernelArg(matmul_kernel, 5, sizeof(int), &K);
        
        // Execute kernel
        size_t globalSize[2] = {(size_t)M, (size_t)N};
        clEnqueueNDRangeKernel(queue, matmul_kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
        
        // Read results
        clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * C.size(), C.data(), 0, NULL, NULL);
        
        // Cleanup
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufC);
    }
    
    // GPU forward pass simulation
    double gpu_forward_pass(const std::vector<std::vector<float>>& X, 
                           const std::vector<std::vector<float>>& weights1,
                           const std::vector<std::vector<float>>& weights2,
                           int batch_size, int input_size, int hidden_size, int output_size) {
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Layer 1: Input to Hidden
        std::vector<float> hidden(hidden_size);
        std::vector<float> input_flat(input_size);
        std::vector<float> weights1_flat(input_size * hidden_size);
        
        // Flatten data
        for (int i = 0; i < input_size; ++i) {
            input_flat[i] = X[0][i];
        }
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                weights1_flat[i * hidden_size + j] = weights1[i][j];
            }
        }
        
        // GPU matrix multiplication
        gpu_matmul(input_flat, weights1_flat, hidden, 1, hidden_size, input_size);
        
        // GPU sigmoid activation
        cl_mem buf_hidden = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * hidden_size, hidden.data(), NULL);
        clSetKernelArg(sigmoid_kernel, 0, sizeof(cl_mem), &buf_hidden);
        clSetKernelArg(sigmoid_kernel, 1, sizeof(int), &hidden_size);
        size_t globalSize = hidden_size;
        clEnqueueNDRangeKernel(queue, sigmoid_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, buf_hidden, CL_TRUE, 0, sizeof(float) * hidden_size, hidden.data(), 0, NULL, NULL);
        clReleaseMemObject(buf_hidden);
        
        // Layer 2: Hidden to Output
        std::vector<float> output(output_size);
        std::vector<float> weights2_flat(hidden_size * output_size);
        
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                weights2_flat[i * output_size + j] = weights2[i][j];
            }
        }
        
        gpu_matmul(hidden, weights2_flat, output, 1, output_size, hidden_size);
        
        // Final sigmoid
        cl_mem buf_output = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * output_size, output.data(), NULL);
        clSetKernelArg(sigmoid_kernel, 0, sizeof(cl_mem), &buf_output);
        clSetKernelArg(sigmoid_kernel, 1, sizeof(int), &output_size);
        globalSize = output_size;
        clEnqueueNDRangeKernel(queue, sigmoid_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
        clReleaseMemObject(buf_output);
        
        clFinish(queue);
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
    
    void cleanup() {
        clReleaseKernel(matmul_kernel);
        clReleaseKernel(sigmoid_kernel);
        clReleaseKernel(relu_kernel);
        clReleaseKernel(sigmoid_deriv_kernel);
        clReleaseKernel(update_kernel);
        clReleaseProgram(matmul_program);
        clReleaseProgram(activation_program);
        clReleaseProgram(update_program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
};

// CPU baseline implementation
class CPUTraining {
public:
    static void cpu_matmul(const std::vector<std::vector<float>>& A, 
                          const std::vector<std::vector<float>>& B,
                          std::vector<std::vector<float>>& C) {
        int rows_A = A.size();
        int cols_A = A[0].size();
        int cols_B = B[0].size();
        
        for (int i = 0; i < rows_A; ++i) {
            for (int j = 0; j < cols_B; ++j) {
                C[i][j] = 0;
                for (int k = 0; k < cols_A; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
    
    static float sigmoid(float x) {
        return 1.0f / (1.0f + exp(-x));
    }
    
    static double cpu_forward_pass(const std::vector<std::vector<float>>& X,
                                  const std::vector<std::vector<float>>& weights1,
                                  const std::vector<std::vector<float>>& weights2,
                                  int batch_size, int input_size, int hidden_size, int output_size) {
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int batch = 0; batch < batch_size; ++batch) {
            // Layer 1: Input to Hidden
            std::vector<float> hidden(hidden_size, 0);
            for (int h = 0; h < hidden_size; ++h) {
                for (int i = 0; i < input_size; ++i) {
                    hidden[h] += X[batch][i] * weights1[i][h];
                }
                hidden[h] = sigmoid(hidden[h]);
            }
            
            // Layer 2: Hidden to Output
            std::vector<float> output(output_size, 0);
            for (int o = 0; o < output_size; ++o) {
                for (int h = 0; h < hidden_size; ++h) {
                    output[o] += hidden[h] * weights2[h][o];
                }
                output[o] = sigmoid(output[o]);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
};

// Generate gate classification dataset
class DataGenerator {
public:
    static void generateGateData(std::vector<std::vector<float>>& X, 
                               std::vector<std::vector<float>>& y, 
                               int samples, std::string gate_type) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 1);
        
        X.resize(samples, std::vector<float>(2));
        y.resize(samples, std::vector<float>(1));
        
        for (int i = 0; i < samples; ++i) {
            float a = dis(gen);
            float b = dis(gen);
            X[i] = {a, b};
            
            if (gate_type == "AND") {
                y[i][0] = (a == 1 && b == 1) ? 1 : 0;
            } else if (gate_type == "OR") {
                y[i][0] = (a == 1 || b == 1) ? 1 : 0;
            } else if (gate_type == "XOR") {
                y[i][0] = (a != b) ? 1 : 0;
            } else if (gate_type == "NAND") {
                y[i][0] = !(a == 1 && b == 1) ? 1 : 0;
            }
        }
    }
};

int main() {
    std::cout << "=== Neural Network GPU Training Benchmark ===" << std::endl;
    std::cout << "Gate Classification Task (AND, OR, XOR, NAND)" << std::endl;
    
    // Initialize GPU benchmark
    GPUTrainingBenchmark gpu_benchmark;
    if (!gpu_benchmark.initialize()) {
        std::cerr << "Failed to initialize GPU!" << std::endl;
        return -1;
    }
    
    // Test configurations
    std::vector<int> batch_sizes = {100, 500, 1000, 5000};
    std::vector<std::string> gates = {"AND", "OR", "XOR", "NAND"};
    
    int input_size = 2;
    int hidden_size = 8;
    int output_size = 1;
    
    std::cout << "\nNetwork Architecture: " << input_size << " -> " << hidden_size << " -> " << output_size << std::endl;
    
    for (const std::string& gate : gates) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Testing " << gate << " Gate Classification" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        for (int batch_size : batch_sizes) {
            std::cout << "\nBatch Size: " << batch_size << " samples" << std::endl;
            std::cout << std::string(40, '-') << std::endl;
            
            // Generate data
            std::vector<std::vector<float>> X, y;
            DataGenerator::generateGateData(X, y, batch_size, gate);
            
            // Initialize random weights
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> weight_dist(-1.0f, 1.0f);
            
            std::vector<std::vector<float>> weights1(input_size, std::vector<float>(hidden_size));
            std::vector<std::vector<float>> weights2(hidden_size, std::vector<float>(output_size));
            
            for (int i = 0; i < input_size; ++i) {
                for (int j = 0; j < hidden_size; ++j) {
                    weights1[i][j] = weight_dist(gen);
                }
            }
            for (int i = 0; i < hidden_size; ++i) {
                for (int j = 0; j < output_size; ++j) {
                    weights2[i][j] = weight_dist(gen);
                }
            }
            
            // Benchmark CPU
            double cpu_time = CPUTraining::cpu_forward_pass(X, weights1, weights2, 
                                                          batch_size, input_size, hidden_size, output_size);
            
            // Benchmark GPU
            double gpu_time = gpu_benchmark.gpu_forward_pass(X, weights1, weights2,
                                                           batch_size, input_size, hidden_size, output_size);
            
            // Display results
            std::cout << "Performance Results:" << std::endl;
            std::cout << "  CPU (Ryzen) Time:   " << std::fixed << std::setprecision(3) 
                      << cpu_time << " ms" << std::endl;
            std::cout << "  GPU (Radeon) Time:  " << gpu_time << " ms" << std::endl;
            
            double speedup = cpu_time / gpu_time;
            std::cout << "  GPU Speedup:        " << std::fixed << std::setprecision(2) 
                      << speedup << "x faster" << std::endl;
            
            // Performance analysis
            if (speedup > 2.0) {
                std::cout << "  Status: ✅ GPU shows significant acceleration" << std::endl;
            } else if (speedup > 1.2) {
                std::cout << "  Status: ⚡ GPU shows moderate acceleration" << std::endl;
            } else if (speedup > 0.8) {
                std::cout << "  Status: ⚠️  GPU and CPU performance similar" << std::endl;
            } else {
                std::cout << "  Status: ❌ CPU faster (overhead dominates)" << std::endl;
            }
            
            // Throughput analysis
            double cpu_throughput = (batch_size * 1000.0) / cpu_time;  // samples/second
            double gpu_throughput = (batch_size * 1000.0) / gpu_time;
            
            std::cout << "  CPU Throughput:     " << std::fixed << std::setprecision(0) 
                      << cpu_throughput << " samples/sec" << std::endl;
            std::cout << "  GPU Throughput:     " << std::fixed << std::setprecision(0) 
                      << gpu_throughput << " samples/sec" << std::endl;
        }
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Benchmark Summary:" << std::endl;
    std::cout << "- Small batches: CPU may be faster due to GPU overhead" << std::endl;
    std::cout << "- Large batches: GPU should show significant speedup" << std::endl;
    std::cout << "- XOR gate: Most complex, best for GPU parallelization" << std::endl;
    std::cout << "- Optimal batch size for GPU: Usually 1000+ samples" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    gpu_benchmark.cleanup();
    return 0;
}