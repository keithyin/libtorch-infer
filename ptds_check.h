#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <iostream>

void ptds_check() {
    at::Device device(at::kCUDA, 0);

    // 拿到一个非默认 stream
    auto s1 = at::cuda::getStreamFromPool();
    auto d = torch::empty({1}, torch::dtype(torch::kInt32).device(device));

    {
        at::cuda::CUDAStreamGuard guard(s1);
        // 在 stream s1 上操作，把 d[0] = 123 (模拟 kernel 写 threadIdx.x=0)
        d.fill_(123);
    }

    {
        // 在 default stream 上操作，把 d[0] = 456 (模拟另一个 kernel)
        d.fill_(456);
    }

    // 强制同步
    auto host_d = d.cpu();
    std::cout << "Result = " << host_d.item<int>() << std::endl;
}