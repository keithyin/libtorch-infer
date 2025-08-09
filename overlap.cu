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
    float* d_data2 = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_data2, bytes);

    // Create streams
    cudaStream_t stream_memcpy, stream_compute;
    cudaStreamCreate(&stream_memcpy);
    cudaStreamCreate(&stream_compute);

    // Async H2D copy on stream 1
    // 该代码观察到的现象是，dummy_kernel 直到 cudaMemcpyAsync 中的 内存拷贝之后才被调度执行
    // dummy_kernel 的执行和 dummy_kernel 后面提交的 cudaMemcpyAsync 的执行是 overlap 的

    cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream_memcpy);
    cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream_memcpy);
    // Launch kernel on stream 2 (assume no dependency, simulate overlap)
    int elements = N / 1024;
    dummy_kernel<<<elements / 256, 256, 0, stream_compute>>>(d_data2, elements);
    cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream_memcpy);


    // Sync both streams
    cudaStreamSynchronize(stream_memcpy);
    cudaStreamSynchronize(stream_compute);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_data2);
    cudaFreeHost(h_data);
    cudaStreamDestroy(stream_memcpy);
    cudaStreamDestroy(stream_compute);

    std::cout << "Done with async memcpy and compute with overlap.\n";
    return 0;
}

// nvcc -o overlap.exe overlap.cu