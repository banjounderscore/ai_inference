#include "TrtInference.h"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

// CUDA runtime forward declarations
typedef int          cudaError_t;
typedef void*        cudaStream_t;
typedef enum { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 } cudaMemcpyKind;

extern "C" {
    cudaError_t cudaMalloc(void** devPtr, size_t size);
    cudaError_t cudaMallocHost(void** ptr, size_t size);
    cudaError_t cudaFree(void* devPtr);
    cudaError_t cudaFreeHost(void* ptr);
    cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
    cudaError_t cudaMemset(void* devPtr, int value, size_t count);
    cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
    cudaError_t cudaStreamDestroy(cudaStream_t stream);
    cudaError_t cudaStreamSynchronize(cudaStream_t stream);
    const char* cudaGetErrorString(cudaError_t error);
}

static const cudaError_t  cudaSuccess          = 0;
static const unsigned int cudaStreamNonBlocking = 0x01;

static inline cudaError_t cudaMallocF(float** p, size_t n)  { return cudaMalloc(reinterpret_cast<void**>(p), n); }
static inline cudaError_t cudaMallocHF(float** p, size_t n) { return cudaMallocHost(reinterpret_cast<void**>(p), n); }

static thread_local std::string g_lastError;
static void SetError(const std::string& msg) { g_lastError = msg; }

struct TrtContext
{
    Ort::Env            env{ ORT_LOGGING_LEVEL_WARNING, "TrtDll" };
    Ort::RunOptions     runOpts;  // cached — avoids per-call heap alloc
    Ort::Session*       session     = nullptr;
    Ort::IoBinding*     binding     = nullptr;
    Ort::MemoryInfo*    cudaMemInfo = nullptr;

    // Device buffers at stable addresses — required for ORT's internal CUDA graph
    float* d_input  = nullptr;
    float* d_output = nullptr;

    // Pinned host input buffer (output goes directly to caller's pinned buffer)
    float* h_input  = nullptr;

    cudaStream_t stream = nullptr;

    size_t inputBytes     = 0;
    size_t outputBytes    = 0;
    int    outputElements = 0;

    std::string inputName;
    std::string outputName;

    ~TrtContext()
    {
        delete binding;
        delete session;
        delete cudaMemInfo;
        if (stream)   cudaStreamDestroy(stream);
        if (d_input)  cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (h_input)  cudaFreeHost(h_input);
    }
};

static bool WideToUtf8(const wchar_t* wide, std::string& out)
{
    if (!wide) { out = ""; return true; }
    int n = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) return false;
    out.resize(n - 1);
    WideCharToMultiByte(CP_UTF8, 0, wide, -1, out.data(), n, nullptr, nullptr);
    return true;
}

// Builds a session with the TensorRT EP always registered, and the CUDA EP appended
// only when includeCudaFallback is set. cuDNN's handle creation (needed by the CUDA EP)
// loads a large precompiled-engine/kernel catalog unconditionally — a big, fixed memory
// cost paid just by registering the provider, independent of algo-search or workspace
// settings. CreateContext below only pays that cost if TensorRT-only actually can't
// build the model.
static Ort::Session* BuildSession(Ort::Env& env, const void* modelData, size_t modelSize,
                                   const wchar_t* modelPath, const std::string& cacheDir8,
                                   const std::string& prefix8, bool includeCudaFallback)
{
    Ort::SessionOptions opts;

    OrtTensorRTProviderOptionsV2* trtOpts = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trtOpts));
    const char* keys[] = {
        "device_id", "trt_fp16_enable",
        "trt_engine_cache_enable", "trt_engine_cache_path", "trt_engine_cache_prefix",
        "trt_engine_decryption_enable", "trt_force_sequential_engine_build",
        "trt_timing_cache_enable", "trt_timing_cache_path",
        "trt_builder_optimization_level", "trt_max_workspace_size",
        "trt_dla_enable", "trt_dump_subgraphs", "trt_auxiliary_streams",
        "trt_cuda_graph_enable"
    };
    const char* vals[] = {
        "0", "1",
        "1", cacheDir8.c_str(), prefix8.c_str(),
        "0", "0",
        "1", cacheDir8.c_str(),
        "3", "1073741824",
        "0", "0", "0",
        "1"
    };
    Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(trtOpts, keys, vals, 15));
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_TensorRT_V2(opts, trtOpts));
    Ort::GetApi().ReleaseTensorRTProviderOptions(trtOpts);

    if (includeCudaFallback)
    {
        OrtCUDAProviderOptionsV2* cudaOpts = nullptr;
        Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cudaOpts));
        const char* ck[] = {
            "device_id",
            "cudnn_conv_algo_search", "cudnn_conv_use_max_workspace",
            "do_copy_in_default_stream", "enable_cuda_graph", "use_tf32"
        };
        const char* cv[] = {
            "0",
            "DEFAULT", "0",
            "0", "1", "1"
        };
        Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cudaOpts, ck, cv, 6));
        Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(opts, cudaOpts));
        Ort::GetApi().ReleaseCUDAProviderOptions(cudaOpts);
    }

    if (modelData && modelSize > 0)
        return new Ort::Session(env, modelData, modelSize, opts);
    else
        return new Ort::Session(env, modelPath, opts);
}

static TrtContext* CreateContext(const void*    modelData,
                                 size_t         modelSize,
                                 const wchar_t* modelPath,
                                 const wchar_t* engineCacheDir,
                                 const wchar_t* cachePrefix,
                                 int            imageSize,
                                 int*           outOutputElements)
{
    auto* ctx = new TrtContext();
    try
    {
        std::string cacheDir8, prefix8;
        if (!WideToUtf8(engineCacheDir, cacheDir8) || !WideToUtf8(cachePrefix, prefix8))
            throw std::runtime_error("Path conversion failed");

        ctx->inputBytes = (size_t)imageSize * imageSize * 3 * sizeof(float);

        try
        {
            ctx->session = BuildSession(ctx->env, modelData, modelSize, modelPath, cacheDir8, prefix8, false);
        }
        catch (const std::exception&)
        {
            ctx->session = BuildSession(ctx->env, modelData, modelSize, modelPath, cacheDir8, prefix8, true);
        }

        Ort::AllocatorWithDefaultOptions alloc;
        ctx->inputName  = ctx->session->GetInputNameAllocated(0, alloc).get();
        ctx->outputName = ctx->session->GetOutputNameAllocated(0, alloc).get();

        auto outInfo = ctx->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto shape   = outInfo.GetShape();
        int64_t total = 1;
        for (size_t i = 1; i < shape.size(); i++) { int64_t d = shape[i]; if (d < 0) d = 8400; total *= d; }
        ctx->outputElements = (int)total;
        ctx->outputBytes    = (size_t)total * sizeof(float);

        cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);

        // Device buffers at stable addresses — must not move after IoBinding is set up
        cudaMallocF(&ctx->d_input,  ctx->inputBytes);
        cudaMemset(ctx->d_input,  0, ctx->inputBytes);
        cudaMallocF(&ctx->d_output, ctx->outputBytes);
        cudaMemset(ctx->d_output, 0, ctx->outputBytes);

        // Pinned host input buffer only — output goes directly to the caller's pinned buffer
        cudaMallocHF(&ctx->h_input, ctx->inputBytes);
        memset(ctx->h_input, 0, ctx->inputBytes);

        ctx->cudaMemInfo = new Ort::MemoryInfo(
            "Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);

        // IoBinding with stable device-side tensors — required for trt_cuda_graph_enable
        ctx->binding = new Ort::IoBinding(*ctx->session);

        std::vector<int64_t> inShape = { 1, 3, imageSize, imageSize };
        auto inVal = Ort::Value::CreateTensor(*ctx->cudaMemInfo,
            ctx->d_input, ctx->inputBytes, inShape.data(), inShape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        ctx->binding->BindInput(ctx->inputName.c_str(), inVal);

        std::vector<int64_t> outShape;
        for (size_t i = 0; i < shape.size(); i++) { int64_t d = shape[i]; if (d < 0) d = (i == 0) ? 1 : 8400; outShape.push_back(d); }
        auto outVal = Ort::Value::CreateTensor(*ctx->cudaMemInfo,
            ctx->d_output, ctx->outputBytes, outShape.data(), outShape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        ctx->binding->BindOutput(ctx->outputName.c_str(), outVal);

        // Warmup — triggers TRT engine build and ORT's internal CUDA graph capture
        for (int i = 0; i < 3; i++)
            ctx->session->Run(ctx->runOpts, *ctx->binding);
        ctx->binding->SynchronizeOutputs();

        if (outOutputElements) *outOutputElements = ctx->outputElements;
        return ctx;
    }
    catch (const std::exception& e)
    {
        SetError(e.what());
        delete ctx;
        return nullptr;
    }
}

static int RunInference(TrtContext* ctx, float* outputData, int outputElements)
{
    try
    {
        // H2D: pinned h_input → stable d_input
        cudaError_t e = cudaMemcpyAsync(ctx->d_input, ctx->h_input, ctx->inputBytes,
                                        cudaMemcpyHostToDevice, ctx->stream);
        if (e != cudaSuccess) { SetError(cudaGetErrorString(e)); return -1; }

        // Sync so d_input is fully written before ORT's internal stream reads it
        e = cudaStreamSynchronize(ctx->stream);
        if (e != cudaSuccess) { SetError(cudaGetErrorString(e)); return -1; }

        // CUDA graph replay on ORT's internal stream
        ctx->session->Run(ctx->runOpts, *ctx->binding);
        ctx->binding->SynchronizeOutputs();  // d_output is ready

        // D2H: d_output → caller's buffer (h_output hop eliminated)
        int n = std::min(outputElements, ctx->outputElements);
        e = cudaMemcpyAsync(outputData, ctx->d_output, (size_t)n * sizeof(float),
                            cudaMemcpyDeviceToHost, ctx->stream);
        if (e != cudaSuccess) { SetError(cudaGetErrorString(e)); return -1; }

        e = cudaStreamSynchronize(ctx->stream);
        if (e != cudaSuccess) { SetError(cudaGetErrorString(e)); return -1; }

        return 0;
    }
    catch (const std::exception& ex) { SetError(ex.what()); return -1; }
}

extern "C"
{

TRTDLL_API void* TrtCreate(const wchar_t* modelPath, const wchar_t* cacheDir,
                           const wchar_t* prefix, int imageSize, int* outElems)
{
    return CreateContext(nullptr, 0, modelPath, cacheDir, prefix, imageSize, outElems);
}

TRTDLL_API void* TrtCreateFromMemory(const uint8_t* data, size_t size,
                                     const wchar_t* cacheDir, const wchar_t* prefix,
                                     int imageSize, int* outElems)
{
    return CreateContext(data, size, nullptr, cacheDir, prefix, imageSize, outElems);
}

TRTDLL_API float* TrtGetInputBuffer(void* handle)
{
    if (!handle) return nullptr;
    return reinterpret_cast<TrtContext*>(handle)->h_input;
}

TRTDLL_API int TrtRunFromPinned(void* handle, float* outputData, int outputElements)
{
    if (!handle || !outputData) { SetError("null arg"); return -1; }
    return RunInference(reinterpret_cast<TrtContext*>(handle), outputData, outputElements);
}

TRTDLL_API int TrtRun(void* handle, const float* inputData, float* outputData, int outputElements)
{
    if (!handle || !inputData || !outputData) { SetError("null arg"); return -1; }
    auto* ctx = reinterpret_cast<TrtContext*>(handle);
    memcpy(ctx->h_input, inputData, ctx->inputBytes);
    return RunInference(ctx, outputData, outputElements);
}

TRTDLL_API void TrtDestroy(void* handle)
{
    delete reinterpret_cast<TrtContext*>(handle);
}

TRTDLL_API const char* TrtGetLastError()
{
    return g_lastError.c_str();
}

} // extern "C"
