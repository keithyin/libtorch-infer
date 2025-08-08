#include <cuda_runtime.h>
#include <iostream>

__global__ void dummy_kernel(float* device_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = device_data[idx];
        for (int i = 0; i < 10000; ++i) {
            x = x * 1.0001f + 0.0001f;
        }
        device_data[idx] = x;
    }
}

int main() {
    constexpr int N = 1 << 30; // 1M elements
    constexpr size_t bytes = N * sizeof(float);

    // Allocate pinned host memory
    float* h_data = nullptr;
    cudaMallocHost(&h_data, bytes);  // pinned memory
    for (int i = 0; i < N; ++i) h_data[i] = 1.0f;

    // Allocate device memory
    float* d_data = nullptr;
    cudaMalloc(&d_data, bytes);

    // Create streams
    cudaStream_t stream_memcpy, stream_compute;
    cudaStreamCreate(&stream_memcpy);
    cudaStreamCreate(&stream_compute);

    // Async H2D copy on stream 1
    cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream_memcpy);

    // Launch kernel on stream 2 (assume no dependency, simulate overlap)
    int elements = N / 4096;
    dummy_kernel<<<elements / 256, 256, 0, stream_compute>>>(d_data, elements);

    // Sync both streams
    cudaStreamSynchronize(stream_memcpy);
    cudaStreamSynchronize(stream_compute);

    // Cleanup
    cudaFree(d_data);
    cudaFreeHost(h_data);
    cudaStreamDestroy(stream_memcpy);
    cudaStreamDestroy(stream_compute);

    std::cout << "Done with async memcpy and compute with overlap.\n";
    return 0;
}