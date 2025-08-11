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

torch::jit::Module get_model_for_infer(c10::Device device)
{
    std::cout << "LibTorch Version: " << TORCH_VERSION << std::endl;

    torch::jit::Module nn;

    nn = torch::jit::load("/root/projects/libtorch-infer/models/model");
    nn.eval();
    nn.to(device);
    return nn;
}

torch::jit::Module get_model_for_infer_selfattn(c10::Device device)
{
    std::cout << "LibTorch Version: " << TORCH_VERSION << std::endl;

    torch::jit::Module nn;

    nn = torch::jit::load("/root/projects/libtorch-infer/models/selfattn/model");
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

void single_thread_real_scenerio(torch::jit::Module &nn, c10::Device device, int tot_iterations = 1000)
{
    int num_feature = 256 * 200 * 61;

    std::vector<float> feature_origin(num_feature, 1);
    std::vector<int64_t> length_origin(256, 200);

    std::cout << "feature_origin.size=" << feature_origin.size() << std::endl;

    torch::Tensor feature = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor length = torch::zeros({256}, c10::TensorOptions().dtype(torch::kInt64));
    float tot_sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < tot_iterations; i++)
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

void single_thread_real_scenerio_with_pinned_memory_and_stream(torch::jit::Module &nn, c10::Device device, int tot_iterations = 1000)
{
    int num_feature = 256 * 200 * 61;

    std::vector<float> feature_origin(num_feature, 1);
    std::vector<int64_t> length_origin(256, 200);

    std::cout << "feature_origin.size=" << feature_origin.size() << std::endl;

    at::cuda::CUDAStream stream1 = at::cuda::getStreamFromPool(false, device.index());
    at::cuda::CUDAStream stream2 = at::cuda::getStreamFromPool(false, device.index());

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

    for (int i = 0; i < tot_iterations / 2; i++)
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

void thread_worker(torch::jit::Module &nn, c10::Device device, int iterations, float &sum_out)
{
    torch::NoGradGuard no_grad;

    float local_sum = 0.0;
    at::cuda::CUDAStream stream = at::cuda::getStreamFromPool(false, device.index());
    stream.synchronize();

    int num_feature = 256 * 200 * 61;

    std::vector<float> feature_origin(num_feature, 1);
    std::vector<int64_t> length_origin(256, 200);

    torch::Tensor feature = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
    torch::Tensor length = torch::zeros({256}, c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));

    torch::Tensor feature_cuda = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor length_cuda = torch::zeros({256}, c10::TensorOptions().dtype(torch::kInt64).device(device));

    torch::Tensor result = torch::zeros({256, 200, 5}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
    at::cuda::CUDAStreamGuard guard(stream);

    for (int i = 0; i < iterations; ++i)
    {
        // Host buffer 填充
        memcpy(feature.data_ptr<float>(), feature_origin.data(), feature_origin.size());
        memcpy(length.data_ptr<int64_t>(), length_origin.data(), length_origin.size());

        feature_cuda.copy_(feature, true);
        length_cuda.copy_(length, true);

        std::vector<torch::jit::IValue> inp;
        inp.push_back(feature_cuda);
        inp.push_back(length_cuda);
        result.copy_(nn.forward(inp).toTensor().contiguous(), true);
        stream.synchronize();
        std::vector<float> output(result.data_ptr<float>(), result.data_ptr<float>() + result.numel());
        local_sum += output[0];
        sum_out = local_sum;
    }
    sum_out = local_sum;
}

void thread_worker_with_cuda_graph(torch::jit::Module &nn, c10::Device device, int iterations, float &sum_out)
{
    torch::NoGradGuard no_grad;

    float local_sum = 0.0;
    at::cuda::CUDAStream stream = at::cuda::getStreamFromPool(false, device.index());
    stream.synchronize();

    at::cuda::CUDAGraph graph;

    int num_feature = 256 * 200 * 61;

    std::vector<float> feature_origin(num_feature, 1);
    std::vector<int64_t> length_origin(256, 200);

    torch::Tensor feature = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
    torch::Tensor length = torch::zeros({256}, c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));

    torch::Tensor feature_cuda = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor length_cuda = torch::zeros({256}, c10::TensorOptions().dtype(torch::kInt64).device(device));

    torch::Tensor result = torch::zeros({256, 200, 5}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
    at::cuda::CUDAStreamGuard guard(stream);

    // capture
    graph.capture_begin();
    feature_cuda.copy_(feature, true);
    length_cuda.copy_(length, true);

    std::vector<torch::jit::IValue> inp;
    inp.push_back(feature_cuda);
    inp.push_back(length_cuda);
    result.copy_(nn.forward(inp).toTensor().contiguous(), true);
    graph.capture_end();

    for (int i = 0; i < iterations; ++i)
    {
        // Host buffer 填充
        memcpy(feature.data_ptr<float>(), feature_origin.data(), feature_origin.size());
        memcpy(length.data_ptr<int64_t>(), length_origin.data(), length_origin.size());

        graph.replay();
        // feature_cuda.copy_(feature, true);
        // length_cuda.copy_(length, true);

        // std::vector<torch::jit::IValue> inp;
        // inp.push_back(feature_cuda);
        // inp.push_back(length_cuda);
        // result.copy_(nn.forward(inp).toTensor().contiguous(), true);
        stream.synchronize();
        std::vector<float> output(result.data_ptr<float>(), result.data_ptr<float>() + result.numel());
        local_sum += output[0];
        sum_out = local_sum;
    }
    sum_out = local_sum;
}

void multi_thread_real_scenerio_with_pinned_memory_and_stream(torch::jit::Module &nn, c10::Device device, int iterations, int num_threads)
{
    const int batch = 256;
    const int len = 200;
    const int feat = 61;
    const int num_features = batch * len * feat;

    std::vector<std::thread> threads;
    std::vector<float> sums(num_threads, 0.0);

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_threads; ++t)
    {
        std::thread thread(thread_worker, std::ref(nn), device, iterations / num_threads, std::ref(sums[t]));
        threads.emplace_back(std::move(thread));
    }

    for (auto &th : threads)
    {
        th.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_sum = 0.0;
    for (float s : sums)
        total_sum += s;

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "multi_thread_real_scenerio: total_sum: " << total_sum
              << " elapsed: " << elapsed.count() << " ms" << std::endl;
}

void multi_thread_real_scenerio_with_pinned_memory_and_stream_and_cuda_graph(torch::jit::Module &nn, c10::Device device, int iterations, int num_threads)
{
    const int batch = 256;
    const int len = 200;
    const int feat = 61;
    const int num_features = batch * len * feat;

    std::vector<std::thread> threads;
    std::vector<float> sums(num_threads, 0.0);

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_threads; ++t)
    {
        std::thread thread(thread_worker_with_cuda_graph, std::ref(nn), device, iterations / num_threads, std::ref(sums[t]));
        threads.emplace_back(std::move(thread));
    }

    for (auto &th : threads)
    {
        th.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_sum = 0.0;
    for (float s : sums)
        total_sum += s;

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "multi_thread_real_scenerio_cuda_graph: total_sum: " << total_sum
              << " elapsed: " << elapsed.count() << " ms" << std::endl;
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

    torch::Tensor result_s1_cuda;
    torch::Tensor result_s2_cuda;

    torch::Tensor result_s1 = torch::empty({B, L, 5}, torch::kFloat32).pin_memory();
    torch::Tensor result_s2 = result_s1.clone().pin_memory();

    // weight on device
    torch::Tensor weight = torch::ones({F, 5}, opts_cuda);

    // create two independent native CUDA streams
    at::cuda::CUDAStream s1 = at::cuda::getStreamFromPool(false, 0);
    at::cuda::CUDAStream s2 = at::cuda::getStreamFromPool(false, 0);
    std::cout << "s1 == s2 ? " << (s1 == s2) << std::endl;

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
            feature_s1_cuda.copy_(feature_s1, /*non_blocking=*/true);
            result_s1_cuda = torch::matmul(feature_s1_cuda, weight); // may call cuBLAS
            result_s1.copy_(result_s1_cuda, /*non_blocking=*/true);
            // cudaMemcpyAsync(result_s1.data_ptr<float>(), tmp.data_ptr<float>(), tmp.numel() * sizeof(float), cudaMemcpyDeviceToHost, s1.stream());
        }

        // stream2 block
        {
            if (i > 0)
            {
                s2.synchronize();
            }
            at::cuda::CUDAStreamGuard guard(s2);
            feature_s2_cuda.copy_(feature_s2, /*non_blocking=*/true);
            result_s2_cuda = torch::matmul(feature_s2_cuda, weight);
            result_s2.copy_(result_s2_cuda, /*non_blocking=*/true);
        }
    }

    // sync and query event timings

    float ms_s1 = 0.0f, ms_s2 = 0.0f;
    // measure each stream's pre->post time
    std::cout << "stream1 block elapsed (ms): " << ms_s1 << " | stream2 block elapsed (ms): " << ms_s2 << std::endl;

    // Finally copy results back
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

    std::cout << "done test\n";
}

void test_pytorch_stream_overlap_mt(c10::Device device)
{
    int B = 256, L = 200, F = 61;
    int num_feature = B * L * F;
    size_t bytes = sizeof(float) * num_feature;

    std::vector<float> feature_origin(num_feature, 1.0f);

    torch::Tensor feature_s1 = torch::from_blob(feature_origin.data(), {B, L, F}, torch::kFloat32).clone().pin_memory();
    torch::Tensor feature_s2 = feature_s1.clone().pin_memory();

    auto opts_cuda = c10::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor feature_s1_cuda = torch::empty({B, L, F}, opts_cuda);
    torch::Tensor feature_s2_cuda = torch::empty({B, L, F}, opts_cuda);

    torch::Tensor result_s1 = torch::empty({B, L, 5}, torch::kFloat32).pin_memory();
    torch::Tensor result_s2 = result_s1.clone().pin_memory();

    torch::Tensor weight = torch::ones({F, 5}, opts_cuda);

    at::cuda::CUDAStream s1 = at::cuda::getStreamFromPool(false, 0);
    at::cuda::CUDAStream s2 = at::cuda::getStreamFromPool(false, 0);

    const int ITER = 8;
    auto worker_s1 = [&](int tot_iter)
    {
        s1.synchronize();

        for (int iter = 0; iter < tot_iter; iter++)
        {
            at::cuda::CUDAStreamGuard guard(s1);
            feature_s1_cuda.copy_(feature_s1, true);
            auto result_s1_cuda = torch::matmul(feature_s1_cuda, weight);
            result_s1.copy_(result_s1_cuda, true);
            s1.synchronize();
        }
    };

    auto worker_s2 = [&](int tot_iter)
    {
        s2.synchronize();
        for (int iter = 0; iter < tot_iter; iter++)
        {
            at::cuda::CUDAStreamGuard guard(s2);
            feature_s2_cuda.copy_(feature_s2, true);
            auto result_s2_cuda = torch::matmul(feature_s2_cuda, weight);
            result_s2.copy_(result_s2_cuda, true);
            s2.synchronize();
        }
    };

    std::thread t1(worker_s1, ITER);
    std::thread t2(worker_s2, ITER);
    t1.join();
    t2.join();

    std::cout << "done test\n";
}

void test_pytorch_stream_overlap_mt_no_pin(c10::Device device)
{
    int B = 256, L = 200, F = 61;
    int num_feature = B * L * F;
    size_t bytes = sizeof(float) * num_feature;

    std::vector<float> feature_origin(num_feature, 1.0f);

    torch::Tensor feature_s1 = torch::from_blob(feature_origin.data(), {B, L, F}, torch::kFloat32).clone();
    torch::Tensor feature_s2 = feature_s1.clone();

    auto opts_cuda = c10::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor feature_s1_cuda = torch::empty({B, L, F}, opts_cuda);
    torch::Tensor feature_s2_cuda = torch::empty({B, L, F}, opts_cuda);

    torch::Tensor result_s1_cuda;
    torch::Tensor result_s2_cuda;
    torch::Tensor result_s1 = torch::empty({B, L, 5}, torch::kFloat32);
    torch::Tensor result_s2 = result_s1.clone();

    torch::Tensor weight = torch::ones({F, 5}, opts_cuda);

    at::cuda::CUDAStream s1 = at::cuda::getStreamFromPool(false, 0);
    at::cuda::CUDAStream s2 = at::cuda::getStreamFromPool(false, 0);

    const int ITER = 8;
    auto worker_s1 = [&](int tot_iter)
    {
        for (int iter = 0; iter < tot_iter; iter++)
        {
            if (iter > 0)
                s1.synchronize();
            at::cuda::CUDAStreamGuard guard(s1);
            feature_s1_cuda.copy_(feature_s1, true);
            result_s1_cuda = torch::matmul(feature_s1_cuda, weight);
            result_s1.copy_(result_s1_cuda, true);
        }
    };

    auto worker_s2 = [&](int tot_iter)
    {
        for (int iter = 0; iter < tot_iter; iter++)
        {
            if (iter > 0)
                s2.synchronize();
            at::cuda::CUDAStreamGuard guard(s2);
            feature_s2_cuda.copy_(feature_s2, true);
            result_s2_cuda = torch::matmul(feature_s2_cuda, weight);
            result_s2.copy_(result_s2_cuda, true);
        }
    };

    std::thread t1(worker_s1, ITER);
    std::thread t2(worker_s2, ITER);
    t1.join();
    t2.join();

    s1.synchronize();
    s2.synchronize();

    std::cout << "done test\n";
}

void test_pytorch_default_stream_mt(c10::Device device)
{
    int B = 256, L = 200, F = 61;
    int num_feature = B * L * F;
    size_t bytes = sizeof(float) * num_feature;

    std::vector<float> feature_origin(num_feature, 1.0f);

    torch::Tensor feature_s1 = torch::from_blob(feature_origin.data(), {B, L, F}, torch::kFloat32).clone().pin_memory();
    torch::Tensor feature_s2 = feature_s1.clone().pin_memory();

    auto opts_cuda = c10::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor feature_s1_cuda = torch::empty({B, L, F}, opts_cuda);
    torch::Tensor feature_s2_cuda = torch::empty({B, L, F}, opts_cuda);

    torch::Tensor result_s1_cuda;
    torch::Tensor result_s2_cuda;
    torch::Tensor result_s1 = torch::empty({B, L, 5}, torch::kFloat32).pin_memory();
    torch::Tensor result_s2 = result_s1.clone().pin_memory();

    torch::Tensor weight = torch::ones({F, 5}, opts_cuda);

    const int ITER = 8;
    auto worker_s1 = [&](int tot_iter)
    {
        for (int iter = 0; iter < tot_iter; iter++)
        {
            feature_s1_cuda.copy_(feature_s1, true);
            result_s1_cuda = torch::matmul(feature_s1_cuda, weight);
            result_s1.copy_(result_s1_cuda, true);
        }
    };

    auto worker_s2 = [&](int tot_iter)
    {
        for (int iter = 0; iter < tot_iter; iter++)
        {
            feature_s2_cuda.copy_(feature_s2, true);
            result_s2_cuda = torch::matmul(feature_s2_cuda, weight);
            result_s2.copy_(result_s2_cuda, true);
        }
    };

    std::thread t1(worker_s1, ITER);
    std::thread t2(worker_s2, ITER);
    t1.join();
    t2.join();

    cudaDeviceSynchronize();

    std::cout << "done test\n";
}

int main()
{
    torch::set_num_interop_threads(1);
    torch::set_num_threads(1);
    c10::Device device("cuda:3");
    torch::NoGradGuard no_grad;

    test_pytorch_stream_overlap(device);
    // torch::jit::Module nn = get_model_for_infer(device);
    torch::jit::Module nn = get_model_for_infer_selfattn(device);
    warm_up(nn, device);
    int tot_iter = 4000;
    single_thread_real_scenerio(nn, device, tot_iter);
    single_thread_real_scenerio_with_pinned_memory_and_stream(nn, device, tot_iter);
    multi_thread_real_scenerio_with_pinned_memory_and_stream(nn, device, tot_iter, 2);
    multi_thread_real_scenerio_with_pinned_memory_and_stream_and_cuda_graph(nn, device, tot_iter, 2);

    // multi_thread_real_scenerio_with_pinned_memory_and_stream(nn, device, tot_iter, 4);

    // single_thread_real_scenerio(nn, device);
    // single_thread_infer(nn, device);
    // single_thread_real_scenerio_with_pinned_memory_and_stream_gemm(device);
    // test_pytorch_stream_overlap_mt(device);
    // test_pytorch_default_stream_mt(device);
    // test_pytorch_stream_overlap(device);
    // test_pytorch_stream_overlap_mt(device);
    // test_pytorch_stream_overlap_mt_no_pin(device);
}
