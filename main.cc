#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <memory>
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

torch::jit::Module get_model_for_infer(c10::Device device)
{
    std::cout << "LibTorch Version: " << TORCH_VERSION << std::endl;

    torch::jit::Module nn;

    nn = torch::jit::load("/root/projects/libtorch-infer/models/model");
    nn.eval();
    nn.to(device);
    return nn;
}

void warm_up(torch::jit::Module &nn, c10::Device device)
{
    float tot_sum = 0.0;
    for (int i = 0; i < 10; i++)
    {
        torch::Tensor feature = torch::ones({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).device(device));
        torch::Tensor length = torch::ones({256}, c10::TensorOptions().dtype(torch::kInt64).device(device)) * 200;

        std::vector<torch::jit::IValue> inp;
        inp.push_back(feature);
        inp.push_back(length);

        // to cpu is needed
        torch::Tensor result = nn.forward(inp).toTensor().contiguous().to(torch::kCPU);

        std::vector<float> output(result.data_ptr<float>(), result.data_ptr<float>() + result.numel());

        tot_sum += output[0];
    }

    std::cout << "warm up: tot_sum: " << tot_sum << std::endl;
}

void single_thread_infer(torch::jit::Module &nn, c10::Device device)
{
    float tot_sum = 0.0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++)
    {
        torch::Tensor feature = torch::ones({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).device(device));
        torch::Tensor length = torch::ones({256}, c10::TensorOptions().dtype(torch::kInt64).device(device)) * 200;

        std::vector<torch::jit::IValue> inp;
        inp.push_back(feature);
        inp.push_back(length);
        torch::Tensor result = nn.forward(inp).toTensor().contiguous().to(torch::kCPU);
        std::vector<float> output(result.data_ptr<float>(), result.data_ptr<float>() + result.numel());
        tot_sum += output[0];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "single_thread_infer: tot_sum: " << tot_sum << " elapsed: " << elapsed.count() << std::endl;
}

void single_thread_real_scenerio(torch::jit::Module &nn, c10::Device device)
{
    int num_feature = 256 * 200 * 61;

    std::vector<float> feature_origin(num_feature, 1);
    std::vector<int64_t> length_origin(256, 200);

    std::cout << "feature_origin.size=" << feature_origin.size() << std::endl;

    torch::Tensor feature = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor length = torch::zeros({256}, c10::TensorOptions().dtype(torch::kInt64));
    float tot_sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; i++)
    {
        memcpy(feature.data_ptr<float>(), feature_origin.data(), feature_origin.size());
        memcpy(length.data_ptr<int64_t>(), length_origin.data(), length_origin.size());

        auto gpu_feature = feature.to(device);
        auto gpu_length = length.to(device);

        std::vector<torch::jit::IValue> inp;
        inp.push_back(gpu_feature);
        inp.push_back(gpu_length);
        torch::Tensor result = nn.forward(inp).toTensor().contiguous().to(torch::kCPU);
        std::vector<float> output(result.data_ptr<float>(), result.data_ptr<float>() + result.numel());
        tot_sum += output[0];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "single_thread_real_scenerio: tot_sum: " << tot_sum << " elapsed: " << elapsed.count() << std::endl;
}

void single_thread_real_scenerio_with_pinned_memory(torch::jit::Module &nn, c10::Device device)
{
    int num_feature = 256 * 200 * 61;

    std::vector<float> feature_origin(num_feature, 1);
    std::vector<int64_t> length_origin(256, 200);

    std::cout << "feature_origin.size=" << feature_origin.size() << std::endl;

    torch::Tensor feature = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
    torch::Tensor length = torch::zeros({256}, c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
    float tot_sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; i++)
    {
        memcpy(feature.data_ptr<float>(), feature_origin.data(), feature_origin.size());
        memcpy(length.data_ptr<int64_t>(), length_origin.data(), length_origin.size());

        auto gpu_feature = feature.to(device);
        auto gpu_length = length.to(device);

        std::vector<torch::jit::IValue> inp;
        inp.push_back(gpu_feature);
        inp.push_back(gpu_length);
        torch::Tensor result = nn.forward(inp).toTensor().contiguous().to(torch::kCPU);
        std::vector<float> output(result.data_ptr<float>(), result.data_ptr<float>() + result.numel());
        tot_sum += output[0];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "single_thread_real_scenerio: tot_sum: " << tot_sum << " elapsed: " << elapsed.count() << std::endl;
}

void single_thread_real_scenerio_with_pinned_memory_and_stream(torch::jit::Module &nn, c10::Device device)
{
    int num_feature = 256 * 200 * 61;

    std::vector<float> feature_origin(num_feature, 1);
    std::vector<int64_t> length_origin(256, 200);

    std::cout << "feature_origin.size=" << feature_origin.size() << std::endl;

    at::cuda::CUDAStream stream1 = at::cuda::getStreamFromPool(false, 3);
    at::cuda::CUDAStream stream2 = at::cuda::getStreamFromPool(false, 3);

    torch::Tensor feature_s1 = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
    torch::Tensor length_s1 = torch::zeros({256}, c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));

    torch::Tensor feature_s1_cuda = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor length_s1_cuda = torch::zeros({256}, c10::TensorOptions().dtype(torch::kInt64).device(device));

    torch::Tensor result_s1 = torch::zeros({256, 200, 5}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));

    torch::Tensor feature_s2 = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
    torch::Tensor length_s2 = torch::zeros({256}, c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));

    torch::Tensor feature_s2_cuda = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor length_s2_cuda = torch::zeros({256}, c10::TensorOptions().dtype(torch::kInt64).device(device));
    torch::Tensor result_s2 = torch::zeros({256, 200, 5}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));

    float tot_sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 500; i++)
    {
        {
            if (i > 0)
            {
                stream1.synchronize();
                std::vector<float> output(result_s1.data_ptr<float>(), result_s1.data_ptr<float>() + result_s1.numel());
                tot_sum += output[0];
            }

            at::cuda::CUDAStreamGuard guard(stream1);
            memcpy(feature_s1.data_ptr<float>(), feature_origin.data(), feature_origin.size());
            memcpy(length_s1.data_ptr<int64_t>(), length_origin.data(), length_origin.size());

            feature_s1_cuda.copy_(feature_s1, true);
            length_s1_cuda.copy_(length_s1, true);

            std::vector<torch::jit::IValue> inp;
            inp.push_back(feature_s1_cuda);
            inp.push_back(length_s1_cuda);
            result_s1.copy_(nn.forward(inp).toTensor().contiguous(), true);
        }

        {
            if (i > 0)
            {
                stream2.synchronize();
                std::vector<float> output(result_s2.data_ptr<float>(), result_s2.data_ptr<float>() + result_s2.numel());
                tot_sum += output[0];
            }
            at::cuda::CUDAStreamGuard guard(stream2);
            memcpy(feature_s2.data_ptr<float>(), feature_origin.data(), feature_origin.size());
            memcpy(length_s2.data_ptr<int64_t>(), length_origin.data(), length_origin.size());

            feature_s2_cuda.copy_(feature_s2, true);
            length_s2_cuda.copy_(length_s2, true);

            std::vector<torch::jit::IValue> inp;
            inp.push_back(feature_s2_cuda);
            inp.push_back(length_s2_cuda);
            result_s2.copy_(nn.forward(inp).toTensor().contiguous(), true);
        }
    }

    stream1.synchronize();
    std::vector<float> output(result_s1.data_ptr<float>(), result_s1.data_ptr<float>() + result_s1.numel());
    tot_sum += output[0];

    stream2.synchronize();
    std::vector<float> output2(result_s2.data_ptr<float>(), result_s2.data_ptr<float>() + result_s2.numel());
    tot_sum += output2[0];

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "single_thread_real_scenerio: tot_sum: " << tot_sum << " elapsed: " << elapsed.count() << std::endl;
}

void single_thread_real_scenerio_with_pinned_memory_and_stream_gemm(c10::Device device)
{
    int num_feature = 256 * 200 * 61;

    std::vector<float> feature_origin(num_feature, 1);
    std::vector<int64_t> length_origin(256, 200);

    std::cout << "feature_origin.size=" << feature_origin.size() << std::endl;

    at::cuda::CUDAStream stream1 = at::cuda::getStreamFromPool(false, 0);
    at::cuda::CUDAStream stream2 = at::cuda::getStreamFromPool(false, 0);

    std::cout << "s1 == s2 ? " << (stream1 == stream2) << std::endl;

    torch::Tensor feature_s1 = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));

    torch::Tensor feature_s1_cuda = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).device(device));

    torch::Tensor result_s1 = torch::zeros({256, 200, 5}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));

    torch::Tensor feature_s2 = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));

    torch::Tensor feature_s2_cuda = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor result_s2 = torch::zeros({256, 200, 5}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));

    float tot_sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    memcpy(feature_s1.data_ptr<float>(), feature_origin.data(), feature_origin.size());
    memcpy(feature_s2.data_ptr<float>(), feature_origin.data(), feature_origin.size());

    torch::Tensor weight = torch::ones({61, 5}, c10::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor weight2 = torch::ones({5, 5}, c10::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor weight3 = torch::ones({5, 5}, c10::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor weight4 = torch::ones({5, 5}, c10::TensorOptions().dtype(torch::kFloat32).device(device));

    // WARM-UP: trigger lazy init (do this synchronously)
    // {
    //     at::cuda::CUDAStreamGuard guard(stream1);
    //     feature_s1_cuda.copy_(feature_s1, /*non_blocking=*/true);
    //     auto tmp = feature_s1_cuda.matmul(weight).matmul(weight2).matmul(weight3).matmul(weight4);
    //     result_s1.copy_(tmp, /*non_blocking=*/true);
    // }
    // {
    //     at::cuda::CUDAStreamGuard guard(stream2);
    //     feature_s2_cuda.copy_(feature_s2, /*non_blocking=*/true);
    //     auto tmp = feature_s2_cuda.matmul(weight).matmul(weight2).matmul(weight3).matmul(weight4);
    //     result_s2.copy_(tmp, /*non_blocking=*/true);
    // }
    // // ensure warm-up finished
    // cudaDeviceSynchronize();

    for (int i = 0; i < 10; i++)
    {
        {
            if (i > 0)
            {
                stream1.synchronize();
                // std::vector<float> output(result_s1.data_ptr<float>(), result_s1.data_ptr<float>() + result_s1.numel());
                // tot_sum += output[0];
            }

            at::cuda::CUDAStreamGuard guard(stream1);

            feature_s1_cuda.copy_(feature_s1, true);

            torch::Tensor result_s1_cuda = feature_s1_cuda.matmul(weight).matmul(weight2).matmul(weight3).matmul(weight4);
            result_s1.copy_(result_s1_cuda.contiguous(), true);
        }

        {
            if (i > 0)
            {
                stream2.synchronize();
                // std::vector<float> output(result_s2.data_ptr<float>(), result_s2.data_ptr<float>() + result_s2.numel());
                // tot_sum += output[0];
            }
            at::cuda::CUDAStreamGuard guard(stream2);

            feature_s2_cuda.copy_(feature_s2, true);

            torch::Tensor result_s2_cuda = feature_s2_cuda.matmul(weight).matmul(weight2).matmul(weight3).matmul(weight4);

            result_s2.copy_(result_s2_cuda.contiguous(), true);
        }
    }

    stream1.synchronize();
    std::vector<float> output(result_s1.data_ptr<float>(), result_s1.data_ptr<float>() + result_s1.numel());
    tot_sum += output[0];

    stream2.synchronize();
    std::vector<float> output2(result_s2.data_ptr<float>(), result_s2.data_ptr<float>() + result_s2.numel());
    tot_sum += output2[0];

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "single_thread_real_scenerio: tot_sum: " << tot_sum << " elapsed: " << elapsed.count() << std::endl;
}

// helper to create and record events on ATen stream
inline void record_event(cudaEvent_t e, at::cuda::CUDAStream &s)
{
    cudaEventRecord(e, s.stream());
}

void test_pytorch_stream_overlap(c10::Device device)
{
    int B = 256, L = 200, F = 61;
    int num_feature = B * L * F;
    size_t bytes = sizeof(float) * num_feature;

    std::vector<float> feature_origin(num_feature, 1.0f);

    // two independent pinned host tensors
    torch::Tensor feature_s1 = torch::from_blob(feature_origin.data(), {B, L, F}, torch::kFloat32).clone().pin_memory();
    torch::Tensor feature_s2 = feature_s1.clone().pin_memory();

    // device tensors preallocated
    auto opts_cuda = c10::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor feature_s1_cuda = torch::empty({B, L, F}, opts_cuda);
    torch::Tensor feature_s2_cuda = torch::empty({B, L, F}, opts_cuda);

    // preallocate results on device and host pinned
    torch::Tensor result_s1_cuda = torch::empty({B, L, 5}, opts_cuda);
    torch::Tensor result_s2_cuda = torch::empty({B, L, 5}, opts_cuda);
    torch::Tensor result_s1 = torch::empty({B, L, 5}, torch::kFloat32).pin_memory();
    torch::Tensor result_s2 = result_s1.clone().pin_memory();

    // weight on device
    torch::Tensor weight = torch::ones({F, 5}, opts_cuda);

    // create two independent native CUDA streams
    at::cuda::CUDAStream s1 = at::cuda::getStreamFromPool(false, 0);
    at::cuda::CUDAStream s2 = at::cuda::getStreamFromPool(false, 0);
    std::cout << "s1 == s2 ? " << (s1 == s2) << std::endl;

    // create events
    cudaEvent_t e_s1_pre, e_s1_post, e_s2_pre, e_s2_post;
    cudaEventCreate(&e_s1_pre);
    cudaEventCreate(&e_s1_post);
    cudaEventCreate(&e_s2_pre);
    cudaEventCreate(&e_s2_post);

    // WARM-UP: trigger lazy init (do this synchronously)
    {
        at::cuda::CUDAStreamGuard guard(s1);
        feature_s1_cuda.copy_(feature_s1, /*non_blocking=*/true);
        auto tmp = torch::matmul(feature_s1_cuda, weight);
        result_s1.copy_(tmp, /*non_blocking=*/true);
    }
    {
        at::cuda::CUDAStreamGuard guard(s2);
        feature_s2_cuda.copy_(feature_s2, /*non_blocking=*/true);
        auto tmp = torch::matmul(feature_s2_cuda, weight);
        result_s2.copy_(tmp, /*non_blocking=*/true);
    }
    // ensure warm-up finished
    cudaDeviceSynchronize();
    std::cout << "warm-up done\n";

    const int ITER = 8;
    for (int i = 0; i < ITER; ++i)
    {
        // stream1 block
        {
            if (i > 0)
            {
                s1.synchronize();
            }
            at::cuda::CUDAStreamGuard guard(s1);
            record_event(e_s1_pre, s1);
            feature_s1_cuda.copy_(feature_s1, /*non_blocking=*/true);
            auto tmp = torch::matmul(feature_s1_cuda, weight); // may call cuBLAS
            result_s1.copy_(tmp, /*non_blocking=*/true);
            // cudaMemcpyAsync(result_s1.data_ptr<float>(), tmp.data_ptr<float>(), tmp.numel() * sizeof(float), cudaMemcpyDeviceToHost, s1.stream());
            record_event(e_s1_post, s1);
        }

        // stream2 block
        {
            if (i > 0)
            {
                s2.synchronize();
            }
            at::cuda::CUDAStreamGuard guard(s2);
            record_event(e_s2_pre, s2);
            feature_s2_cuda.copy_(feature_s2, /*non_blocking=*/true);
            auto tmp2 = torch::matmul(feature_s2_cuda, weight);
            result_s2.copy_(tmp2, /*non_blocking=*/true);
            // cudaMemcpyAsync(result_s2.data_ptr<float>(), tmp2.data_ptr<float>(), tmp2.numel() * sizeof(float), cudaMemcpyDeviceToHost, s2.stream());

            record_event(e_s2_post, s2);
        }
    }

    // sync and query event timings
    cudaEventSynchronize(e_s1_post);
    cudaEventSynchronize(e_s2_post);

    float ms_s1 = 0.0f, ms_s2 = 0.0f;
    // measure each stream's pre->post time
    cudaEventElapsedTime(&ms_s1, e_s1_pre, e_s1_post);
    cudaEventElapsedTime(&ms_s2, e_s2_pre, e_s2_post);
    std::cout << "stream1 block elapsed (ms): " << ms_s1 << " | stream2 block elapsed (ms): " << ms_s2 << std::endl;

    // Finally copy results back (one-time)
    {
        at::cuda::CUDAStreamGuard guard(s1);
        result_s1.copy_(result_s1_cuda, /*non_blocking=*/true);
    }
    {
        at::cuda::CUDAStreamGuard guard(s2);
        result_s2.copy_(result_s2_cuda, /*non_blocking=*/true);
    }
    // wait for both streams
    s1.synchronize();
    s2.synchronize();

    // cleanup events
    cudaEventDestroy(e_s1_pre);
    cudaEventDestroy(e_s1_post);
    cudaEventDestroy(e_s2_pre);
    cudaEventDestroy(e_s2_post);

    std::cout << "done test\n";
}

int main()
{
    torch::set_num_interop_threads(1);
    torch::set_num_threads(1);
    c10::Device device("cuda:0");
    torch::NoGradGuard no_grad;

    // test_pytorch_stream_overlap(device);
    // torch::jit::Module nn = get_model_for_infer(device);
    // warm_up(nn, device);

    // single_thread_real_scenerio_with_pinned_memory_and_stream(nn, device);

    // single_thread_real_scenerio(nn, device);
    // single_thread_infer(nn, device);
    single_thread_real_scenerio_with_pinned_memory_and_stream_gemm(device);
}
