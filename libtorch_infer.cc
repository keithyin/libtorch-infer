
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>

#include <c10/cuda/CUDAGuard.h>
#include <memory>
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <mutex>
#include <cmath>

extern "C"
{

    struct NnInferContext
    {
        void *nn;
        void *stream;
        int device_idx;
        float *input_feature_host;
        long *input_length_host;

        float *input_feature_device;
        long *input_length_device;

        float *output_host;
    };

    NnInferContext build_nn_infer_context(const char *model_name, int device_idx)
    {
        c10::Device device(torch::kCUDA, device_idx);
        torch::jit::Module nn = torch::jit::load(model_name);
        nn.eval();
        nn.to(device);

        torch::jit::Module *nn_ptr = new torch::jit::Module(std::move(nn));
        at::cuda::CUDAStream stream = at::cuda::getStreamFromPool(false, device_idx);
        at::cuda::CUDAStream *stream_ptr = new at::cuda::CUDAStream(std::move(stream));

        float 

    }

    void do_infer(NnInferContext nn_infer_context)
    {
    }

    void destroy_nn_infer_context(NnInferContext nn_infer_context)
    {
    }
}
