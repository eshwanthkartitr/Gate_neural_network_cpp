#define CL_TARGET_OPENCL_VERSION 120 
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>  // Add this for std::setprecision

const char* kernelSource = R"(
__kernel void vector_add(__global const float* a, 
                        __global const float* b, 
                        __global float* result, 
                        const unsigned int n) {
    int id = get_global_id(0);
    if (id < n) {
        result[id] = a[id] + b[id] + sin(a[id]) * cos(b[id]);
    }
}
)";

class OpenCLBenchmark {
private:
    cl_platform_id platform;
    cl_context context;
    cl_program program;
    cl_kernel kernel;
    
public:  // Move these to public section
    cl_device_id cpu_device, gpu_device;
    
    bool initialize() {
        cl_int err;
        
        // Get platform
        cl_uint platformCount;
        err = clGetPlatformIDs(1, &platform, &platformCount);
        if (err != CL_SUCCESS || platformCount == 0) {
            std::cout << "No OpenCL platforms found!" << std::endl;
            return false;
        }
        
        // Get devices
        cl_uint deviceCount;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        if (err != CL_SUCCESS || deviceCount == 0) {
            std::cout << "No OpenCL devices found!" << std::endl;
            return false;
        }
        
        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), NULL);
        
        // Find CPU and GPU devices
        bool foundCPU = false, foundGPU = false;
        for (auto device : devices) {
            cl_device_type type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
            
            if (type == CL_DEVICE_TYPE_CPU && !foundCPU) {
                cpu_device = device;
                foundCPU = true;
            } else if (type == CL_DEVICE_TYPE_GPU && !foundGPU) {
                gpu_device = device;
                foundGPU = true;
            }
        }
        
        if (!foundCPU) {
            std::cout << "No CPU OpenCL device found. This is expected with AMD-only drivers." << std::endl;
            std::cout << "Comparing: Pure CPU (Ryzen) vs GPU (Radeon) performance." << std::endl;
            cpu_device = gpu_device;
        }
        
        if (!foundGPU) {
            std::cout << "No GPU device found, using CPU for both tests" << std::endl;
            gpu_device = cpu_device;
        }
        
        return true;
    }
    
    double runBenchmark(cl_device_id device, const std::vector<float>& a, 
                       const std::vector<float>& b, std::vector<float>& result) {
        
        cl_int err;
        size_t dataSize = a.size();
        
        // Create context for the specific device
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        
        // Create command queue (use OpenCL 1.2 compatible API)
        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
        
        // Create program
        program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
        clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        
        // Create kernel
        kernel = clCreateKernel(program, "vector_add", &err);
        
        // Create buffers
        cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * dataSize, (void*)a.data(), &err);
        cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * dataSize, (void*)b.data(), &err);
        cl_mem bufResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         sizeof(float) * dataSize, NULL, &err);
        
        // Set kernel arguments
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufResult);
        clSetKernelArg(kernel, 3, sizeof(unsigned int), &dataSize);
        
        // Execute kernel and measure time
        size_t globalSize = dataSize;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
        clFinish(queue); // Wait for completion
        
        auto end = std::chrono::high_resolution_clock::now();
        
        // Read back results
        clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, 
                           sizeof(float) * dataSize, result.data(), 0, NULL, NULL);
        
        // Cleanup
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufResult);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / 1000.0; // Return milliseconds
    }
    
    void printDeviceInfo(cl_device_id device, const std::string& label) {
        char deviceName[256] = {0}; // Initialize buffer and make it larger
        cl_int err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
        
        cl_uint computeUnits = 0;
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, NULL);
        
        cl_ulong globalMemSize = 0;
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, NULL);
        
        cl_uint maxFreq = 0;
        clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxFreq), &maxFreq, NULL);
        
        std::cout << "\n" << label << " Device Info:" << std::endl;
        std::cout << "  Name: " << (err == CL_SUCCESS ? deviceName : "Unknown Device") << std::endl;
        std::cout << "  Compute Units: " << computeUnits << std::endl;
        std::cout << "  Global Memory: " << globalMemSize / (1024*1024) << " MB" << std::endl;
        std::cout << "  Max Frequency: " << maxFreq << " MHz" << std::endl;
    }
};

// CPU reference implementation for comparison
double cpuBenchmark(const std::vector<float>& a, const std::vector<float>& b, 
                   std::vector<float>& result) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i] + sin(a[i]) * cos(b[i]);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; // Return milliseconds
}

int main() {
    std::cout << "=== OpenCL CPU vs GPU Benchmark ===" << std::endl;
    
    // Test with different data sizes
    std::vector<size_t> sizes = {100000, 1000000, 5000000, 10000000};
    
    OpenCLBenchmark benchmark;
    if (!benchmark.initialize()) {
        std::cerr << "Failed to initialize OpenCL!" << std::endl;
        return -1;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    
    for (size_t dataSize : sizes) {
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Testing with " << dataSize << " elements" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        // Generate random data
        std::vector<float> a(dataSize), b(dataSize);
        std::vector<float> cpuResult(dataSize), gpuResult(dataSize), oclCpuResult(dataSize);
        
        for (size_t i = 0; i < dataSize; ++i) {
            a[i] = dis(gen);
            b[i] = dis(gen);
        }
        
        // Run CPU reference
        double cpuTime = cpuBenchmark(a, b, cpuResult);
        
        // Run OpenCL on CPU
        double oclCpuTime = benchmark.runBenchmark(benchmark.cpu_device, a, b, oclCpuResult);
        
        // Run OpenCL on GPU
        double gpuTime = benchmark.runBenchmark(benchmark.gpu_device, a, b, gpuResult);
        
        // Print device info for first run
        if (dataSize == sizes[0]) {
            benchmark.printDeviceInfo(benchmark.cpu_device, "CPU");
            benchmark.printDeviceInfo(benchmark.gpu_device, "GPU");
            std::cout << std::endl;
        }
        
        // Display results
        std::cout << "Performance Results:" << std::endl;
        std::cout << "  Pure CPU (Ryzen):  " << std::fixed << std::setprecision(3) 
                  << cpuTime << " ms" << std::endl;
        std::cout << "  GPU (Radeon):      " << gpuTime << " ms" << std::endl;
        
        std::cout << "\nSpeedup Analysis:" << std::endl;
        std::cout << "  GPU vs CPU:        " << std::fixed << std::setprecision(2) 
                  << cpuTime / gpuTime << "x faster" << std::endl;
        
        // Verify results are correct (check first few elements)
        bool resultsMatch = true;
        for (size_t i = 0; i < std::min(size_t(100), dataSize); ++i) {
            if (std::abs(cpuResult[i] - gpuResult[i]) > 1e-5f) {
                resultsMatch = false;
                break;
            }
        }
        std::cout << "  Result Verification: " << (resultsMatch ? "PASSED" : "FAILED") << std::endl;
    }
    
    return 0;
}
