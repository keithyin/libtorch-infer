#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device name: " << prop.name << "\n";
    std::cout << "Concurrent copy and kernel execution: "
              << (prop.deviceOverlap ? "Yes" : "No") << "\n";
    std::cout << "Number of async engines: " << prop.asyncEngineCount << "\n";

    return 0;
}