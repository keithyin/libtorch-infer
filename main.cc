#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <iostream>
#include <vector>
#include <chrono>

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

void single_thread_real_scenerio()
{
    int num_feature = 256 * 200 * 61;

    std::vector<float> feature_origin(1, num_feature);
    std::vector<int64_t> length_origin(200, 256);

    torch::Tensor feature = torch::zeros({256, 200, 61}, c10::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor length = torch::zeros({256}, c10::TensorOptions().dtype(torch::kInt64));

    for (int i = 0; i < 1000; i++)
    {
        memcpy(feature.data_ptr<float>(), feature_origin.data(), feature_origin.size());
        memcpy(length.data_ptr<int64_t>(), length_origin.data(), length_origin.size());

        

    }
}

int main()
{
    torch::set_num_interop_threads(1);
    torch::set_num_threads(1);
    c10::Device device("cuda:3");
    torch::NoGradGuard no_grad;
    torch::jit::Module nn = get_model_for_infer(device);
    warm_up(nn, device);

    single_thread_infer(nn, device);
}
