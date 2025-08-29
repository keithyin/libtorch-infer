
#include "libtorch_stream_infer_ffi.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>

#include <c10/cuda/CUDAGuard.h>
#include <sstream>
#include <assert.h>

ModuleInferCtx build_module_infer_ctx(const char *model_path, int device_id, int batch, int tt, int feat_len, int output_size, bool pin_memory)
{
    c10::Device device("cpu");
    if (device_id >= 0)
    {
        std::ostringstream oss;
        oss << "cuda:" << device_id;
        device = c10::Device(oss.str());
    }

    torch::jit::Module *nn = new torch::jit::Module(torch::jit::load(std::string(model_path), device));
    nn->eval();

    at::cuda::CUDAStream *stream_ptr = nullptr;
    if (device_id >= 0)
    {
        stream_ptr = new at::cuda::CUDAStream(at::cuda::getStreamFromPool(false, device_id));
    }

    torch::Tensor *feature = new torch::Tensor(torch::empty({batch, tt, feat_len}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(pin_memory)));
    torch::Tensor *length = new torch::Tensor(torch::empty({batch}, c10::TensorOptions().dtype(torch::kInt64).pinned_memory(pin_memory)));

    torch::Tensor *feature_cuda = nullptr;
    torch::Tensor *length_cuda = nullptr;
    if (device_id >= 0)
    {
        feature_cuda = new torch::Tensor(torch::empty({batch, tt, feat_len}, c10::TensorOptions().dtype(torch::kFloat32).device(device)));
        length_cuda = new torch::Tensor(torch::empty({batch}, c10::TensorOptions().dtype(torch::kInt64).device(device)));
    }

    torch::Tensor *result = new torch::Tensor(torch::empty({batch, tt, output_size}, c10::TensorOptions().dtype(torch::kFloat32).pinned_memory(pin_memory)));

    /*
    // not thread-safe
    struct ModuleInferCtx {
        void* _nn_module;
        float* batch_feature;
        long* batch_lengths;
        float* output;
        void* _stream;
        int device_id;

        void* _batch_feature_tensor;
        void* _batch_lengths_tensor;
        void* _batch_feature_cuda_tensor;
        void* _batch_lengths_cuda_tensor;
        void* _output_tensor;
    };
    */
    ModuleInferCtx ctx{
        nn,
        feature->data_ptr<float>(),
        length->data_ptr<long>(),
        result->data_ptr<float>(),
        stream_ptr,
        device_id,

        new c10::Device(device),

        feature,
        length,
        feature_cuda,
        length_cuda,
        result,
    };
    return ctx;
}

void do_infer(ModuleInferCtx *ctx)
{
    try
    {
        c10::DeviceGuard _device_guard(*static_cast<c10::Device *>(ctx->_device));
        c10::NoGradGuard _no_grad;
        assert(ctx->_stream != nullptr);

        c10::StreamGuard _stream_guard(*static_cast<c10::Stream *>(ctx->_stream));

        static_cast<torch::Tensor *>(ctx->_batch_feature_cuda_tensor)->copy_(*static_cast<torch::Tensor *>(ctx->_batch_feature_tensor), true);
        static_cast<torch::Tensor *>(ctx->_batch_lengths_cuda_tensor)->copy_(*static_cast<torch::Tensor *>(ctx->_batch_lengths_tensor), true);
        auto output_cuda = static_cast<torch::jit::Module *>(ctx->_nn_module)->forward({*static_cast<torch::Tensor *>(ctx->_batch_feature_cuda_tensor), *static_cast<torch::Tensor *>(ctx->_batch_lengths_cuda_tensor)});
        static_cast<torch::Tensor *>(ctx->_output_tensor)->copy_(output_cuda.toTensor(), true);
        static_cast<c10::Stream *>(ctx->_stream)->synchronize();
    }
    catch (const std::exception &e)
    {
        std::cerr << "C++ exception caught: " << e.what() << std::endl;
        throw e;
    }
    catch (...)
    {
        std::cerr << "do_infer exception" << std::endl;
        throw std::runtime_error("do_infer exception");
    }
}

void free_module_infer_ctx(ModuleInferCtx *ctx)
{
    if (!ctx)
        return;

    if (ctx->_nn_module)
    {
        delete static_cast<torch::jit::Module *>(ctx->_nn_module);
        ctx->_nn_module = nullptr;
    }

    if (ctx->_stream)
    {
        delete static_cast<at::cuda::CUDAStream *>(ctx->_stream);
        ctx->_stream = nullptr;
    }

    if (ctx->_device)
    {
        delete static_cast<c10::Device *>(ctx->_device);
        ctx->_device = nullptr;
    }

    auto del_tensor = [](void *&p)
    {
        if (p)
        {
            delete static_cast<at::Tensor *>(p);
            p = nullptr;
        }
    };

    del_tensor(ctx->_batch_feature_tensor);
    del_tensor(ctx->_batch_lengths_tensor);
    del_tensor(ctx->_batch_feature_cuda_tensor);
    del_tensor(ctx->_batch_lengths_cuda_tensor);
    del_tensor(ctx->_output_tensor);
}