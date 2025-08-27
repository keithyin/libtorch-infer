#pragma once

extern "C" {
    // not thread-safe
    struct ModuleInferCtx
    {
        void *_nn_module;
        float *batch_feature;
        long *batch_lengths;
        float *output;
        void *_stream;
        int device_id;
        void *_device;

        void *_batch_feature_tensor;
        void *_batch_lengths_tensor;
        void *_batch_feature_cuda_tensor;
        void *_batch_lengths_cuda_tensor;
        void *_output_tensor;
    };

    ModuleInferCtx build_module_infer_ctx(const char *model_path, int device_id, int batch, int tt, int feat_len, int output_size, bool pin_memory);

    void do_infer(ModuleInferCtx *ctx);

    void free_module_infer_ctx(ModuleInferCtx *ctx);
}