#include "AiInference.h"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
// Not linking onnxruntime.lib (see EnsureBackend below) — ORT_API_MANUAL_INIT stops the
// header from auto-initializing Ort::Global<>::api_ at static-init time, which would
// otherwise require OrtGetApiBase to be a resolvable link-time import.
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT
#include <dml_provider_factory.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <type_traits>

// CUDA runtime forward declarations (Nvidia backend only)
typedef int          cudaError_t;
typedef void*        cudaStream_t;
typedef enum { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 } cudaMemcpyKind;

static const cudaError_t  cudaSuccess          = 0;
static const cudaError_t  cudaErrorNotLoaded   = 1;   // cudaErrorInvalidValue — runtime not loaded
static const unsigned int cudaStreamNonBlocking = 0x01;

// cudart64_12.dll is resolved at runtime (LoadCudaRuntime, called only when the Nvidia backend
// is selected) instead of being linked: a static import would make this whole module fail to
// load (DllNotFound) on machines without the CUDA runtime — i.e. every AMD/Intel/DirectML user.
// The wrappers keep the cuda* names so the Nvidia code paths below read as plain CUDA calls.
static HMODULE g_cudaRuntime = nullptr;

struct CudaApi
{
    cudaError_t (*Malloc)(void**, size_t);
    cudaError_t (*MallocHost)(void**, size_t);
    cudaError_t (*Free)(void*);
    cudaError_t (*FreeHost)(void*);
    cudaError_t (*MemcpyAsync)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
    cudaError_t (*Memset)(void*, int, size_t);
    cudaError_t (*StreamCreateWithFlags)(cudaStream_t*, unsigned int);
    cudaError_t (*StreamDestroy)(cudaStream_t);
    cudaError_t (*StreamSynchronize)(cudaStream_t);
    cudaError_t (*DeviceReset)(void);
    const char* (*GetErrorString)(cudaError_t);
};
static CudaApi g_cuda = {};

static cudaError_t cudaMalloc(void** p, size_t n)        { return g_cuda.Malloc ? g_cuda.Malloc(p, n) : cudaErrorNotLoaded; }
static cudaError_t cudaMallocHost(void** p, size_t n)    { return g_cuda.MallocHost ? g_cuda.MallocHost(p, n) : cudaErrorNotLoaded; }
static cudaError_t cudaFree(void* p)                     { return g_cuda.Free ? g_cuda.Free(p) : cudaErrorNotLoaded; }
static cudaError_t cudaFreeHost(void* p)                 { return g_cuda.FreeHost ? g_cuda.FreeHost(p) : cudaErrorNotLoaded; }
static cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind k, cudaStream_t st)
                                                         { return g_cuda.MemcpyAsync ? g_cuda.MemcpyAsync(d, s, n, k, st) : cudaErrorNotLoaded; }
static cudaError_t cudaMemset(void* p, int v, size_t n)  { return g_cuda.Memset ? g_cuda.Memset(p, v, n) : cudaErrorNotLoaded; }
static cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned int f)
                                                         { return g_cuda.StreamCreateWithFlags ? g_cuda.StreamCreateWithFlags(s, f) : cudaErrorNotLoaded; }
static cudaError_t cudaStreamDestroy(cudaStream_t s)     { return g_cuda.StreamDestroy ? g_cuda.StreamDestroy(s) : cudaErrorNotLoaded; }
static cudaError_t cudaStreamSynchronize(cudaStream_t s) { return g_cuda.StreamSynchronize ? g_cuda.StreamSynchronize(s) : cudaErrorNotLoaded; }
static cudaError_t cudaDeviceReset(void)                 { return g_cuda.DeviceReset ? g_cuda.DeviceReset() : cudaErrorNotLoaded; }
static const char* cudaGetErrorString(cudaError_t e)     { return g_cuda.GetErrorString ? g_cuda.GetErrorString(e) : "CUDA runtime (cudart64_12.dll) is not loaded"; }

static inline cudaError_t cudaMallocF(float** p, size_t n)  { return cudaMalloc(reinterpret_cast<void**>(p), n); }
static inline cudaError_t cudaMallocHF(float** p, size_t n) { return cudaMallocHost(reinterpret_cast<void**>(p), n); }

static thread_local std::string g_lastError;
static thread_local std::string g_inspectJson;
static std::atomic<int> g_liveContexts{ 0 };
static void SetError(const std::string& msg) { g_lastError = msg; }

static void CheckCuda(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Nvidia (CUDA/TensorRT) and DirectML are separate onnxruntime.dll builds — DirectML doesn't
// include the CUDA/TensorRT execution providers, and Nvidia's build doesn't include DML. Only
// one can be loaded (and therefore active in Ort's global API table) at a time, so switching
// families requires every AiContext from the other family to be destroyed first.
enum class Backend { None, Nvidia, DirectML };
static Backend g_backend = Backend::None;
static HMODULE g_ortModule = nullptr;

static std::wstring ModuleDirectory()
{
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCWSTR)&ModuleDirectory, &self);
    wchar_t path[MAX_PATH];
    DWORD n = GetModuleFileNameW(self, path, MAX_PATH);
    std::wstring full(path, n);
    return full.substr(0, full.find_last_of(L"\\/"));
}

static void UnloadCudaRuntime()
{
    g_cuda = {};
    if (g_cudaRuntime)
    {
        FreeLibrary(g_cudaRuntime);
        g_cudaRuntime = nullptr;
    }
}

// Same lookup order a static import used to get: next to this module first, then the normal
// search path (PATH / an installed CUDA toolkit).
static bool LoadCudaRuntime()
{
    if (g_cudaRuntime) return true;

    g_cudaRuntime = LoadLibraryW((ModuleDirectory() + L"\\cudart64_12.dll").c_str());
    if (!g_cudaRuntime) g_cudaRuntime = LoadLibraryW(L"cudart64_12.dll");
    if (!g_cudaRuntime)
    {
        SetError("Failed to load cudart64_12.dll (required for CUDA/TensorRT)");
        return false;
    }

    bool ok = true;
    auto resolve = [&](auto& fn, const char* name) {
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(GetProcAddress(g_cudaRuntime, name));
        if (!fn) ok = false;
    };
    resolve(g_cuda.Malloc,                "cudaMalloc");
    resolve(g_cuda.MallocHost,            "cudaMallocHost");
    resolve(g_cuda.Free,                  "cudaFree");
    resolve(g_cuda.FreeHost,              "cudaFreeHost");
    resolve(g_cuda.MemcpyAsync,           "cudaMemcpyAsync");
    resolve(g_cuda.Memset,                "cudaMemset");
    resolve(g_cuda.StreamCreateWithFlags, "cudaStreamCreateWithFlags");
    resolve(g_cuda.StreamDestroy,         "cudaStreamDestroy");
    resolve(g_cuda.StreamSynchronize,     "cudaStreamSynchronize");
    resolve(g_cuda.DeviceReset,           "cudaDeviceReset");
    resolve(g_cuda.GetErrorString,        "cudaGetErrorString");
    if (!ok)
    {
        SetError("cudart64_12.dll is missing expected exports");
        UnloadCudaRuntime();
        return false;
    }
    return true;
}

static bool EnsureBackend(int provider)
{
    Backend needed = (provider == AI_PROVIDER_DIRECTML) ? Backend::DirectML : Backend::Nvidia;
    if (g_backend == needed) return true;

    if (g_liveContexts.load() != 0)
    {
        SetError("Cannot switch inference backend while a model is still loaded");
        return false;
    }

    if (g_ortModule)
    {
        FreeLibrary(g_ortModule);
        g_ortModule = nullptr;
        g_backend = Backend::None;
    }
    UnloadCudaRuntime();

    if (needed == Backend::Nvidia && !LoadCudaRuntime()) return false;

    const wchar_t* dllNameW = (needed == Backend::DirectML) ? L"onnxruntime_directml.dll" : L"onnxruntime.dll";
    const char*    dllName8 = (needed == Backend::DirectML) ? "onnxruntime_directml.dll" : "onnxruntime.dll";
    std::wstring fullPath = ModuleDirectory() + L"\\" + dllNameW;
    g_ortModule = LoadLibraryW(fullPath.c_str());
    if (!g_ortModule)
    {
        SetError(std::string("Failed to load ") + dllName8);
        return false;
    }

    // Fixed at the older of the two onnxruntime builds we load (DirectML: 1.24.4, API v24;
    // Nvidia: 1.25.1, API v25) — GetApi(v) is backward compatible (a newer runtime honors an
    // older request), but not forward compatible, so using the compiled header's ORT_API_VERSION
    // (25, from the Nvidia package headers we build against) would fail against the older
    // DirectML build. Every OrtApi member this file touches has existed since well before v24.
    constexpr uint32_t kOrtApiVersion = 24;
    using FnGetApiBase = const OrtApiBase*(ORT_API_CALL*)() NO_EXCEPTION;
    auto getApiBase = reinterpret_cast<FnGetApiBase>(GetProcAddress(g_ortModule, "OrtGetApiBase"));
    const OrtApi* api = getApiBase ? getApiBase()->GetApi(kOrtApiVersion) : nullptr;
    if (!api)
    {
        SetError(std::string("OrtGetApiBase/GetApi failed for ") + dllName8);
        FreeLibrary(g_ortModule);
        g_ortModule = nullptr;
        return false;
    }

    Ort::InitApi(api);
    g_backend = needed;
    return true;
}

struct AiContext
{
    int backend = AI_PROVIDER_CUDA;

    Ort::Env*        env     = nullptr;
    Ort::RunOptions* runOpts = nullptr;
    Ort::Session*    session = nullptr;

    // Nvidia only — stable device-side IoBinding for CUDA graph capture
    Ort::IoBinding*  binding     = nullptr;
    Ort::MemoryInfo* cudaMemInfo = nullptr;
    float* d_input  = nullptr;
    float* d_output = nullptr;
    float* h_input  = nullptr;  // pinned host input buffer
    cudaStream_t stream = nullptr;

    // DirectML only — plain host buffers, ORT owns device memory internally
    std::vector<float> dmlInput;
    int dmlImageSize = 0;

    size_t inputBytes     = 0;
    size_t outputBytes    = 0;
    int    outputElements = 0;
    std::vector<int64_t> outputShape;

    std::string inputName;
    std::string outputName;

    ~AiContext()
    {
        delete binding;
        delete session;
        delete cudaMemInfo;
        delete runOpts;
        delete env;
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

static void AppendTensorRT(Ort::SessionOptions& opts, const std::string& cacheDir8, const std::string& prefix8, int deviceId)
{
    OrtTensorRTProviderOptionsV2* trtOpts = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trtOpts));
    std::string deviceId8 = std::to_string(deviceId);
    const char* keys[] = {
        "device_id", "trt_fp16_enable",
        "trt_engine_cache_enable", "trt_engine_cache_path", "trt_engine_cache_prefix",
        "trt_engine_decryption_enable", "trt_force_sequential_engine_build",
        "trt_timing_cache_enable", "trt_timing_cache_path",
        "trt_builder_optimization_level", "trt_max_workspace_size",
        "trt_dla_enable", "trt_dump_subgraphs", "trt_auxiliary_streams",
        "trt_cuda_graph_enable"
    };
    // trt_builder_optimization_level 5 (max) — the extra one-time build cost buys real,
    // permanent runtime latency: level 5 has the builder search harder for the fastest
    // kernels rather than settling early. Not worth it before a build already taking
    // 5-10 minutes was already an accepted cost.
    const char* vals[] = {
        deviceId8.c_str(), "1",
        "1", cacheDir8.c_str(), prefix8.c_str(),
        "0", "0",
        "1", cacheDir8.c_str(),
        "5", "1073741824",
        "0", "0", "0",
        "1"
    };
    Ort::Status st(Ort::GetApi().UpdateTensorRTProviderOptions(trtOpts, keys, vals, 15));
    if (st.IsOK())
        st = Ort::Status(Ort::GetApi().SessionOptionsAppendExecutionProvider_TensorRT_V2(opts, trtOpts));
    Ort::GetApi().ReleaseTensorRTProviderOptions(trtOpts);
    if (!st.IsOK()) throw Ort::Exception(st.GetErrorMessage(), st.GetErrorCode());
}

// cudnn_conv_algo_search is a naming trap: OrtCudnnConvAlgoSearchDefault ("DEFAULT") is enum
// value 2, and ORT's cuDNN-frontend-based conv kernel treats value 2 as HeurMode_t::FALLBACK —
// literally logging "running in Fallback mode. May be extremely slow." for every conv op, which
// is exactly what we were seeing (measured ~8ms/inference vs ~1.2ms on TensorRT/DirectML for
// the same model). "EXHAUSTIVE" (value 0) is what actually benchmarks real algorithms via
// cuDNN's HeurMode_t::B and caches the winner per input shape — worth the one-time search cost
// during the warmup runs below since our input shape is fixed for the life of the session.
// cudnn_conv_use_max_workspace isn't set — defaults to true, letting cuDNN pick freely.
// prefer_nhwc lets tensor cores engage more directly for conv-heavy models (the layout
// tensor cores actually operate on); cuDNN/ORT transpose around it as needed.
static void AppendCuda(Ort::SessionOptions& opts, int deviceId)
{
    OrtCUDAProviderOptionsV2* cudaOpts = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cudaOpts));
    std::string deviceId8 = std::to_string(deviceId);
    const char* ck[] = { "device_id",    "cudnn_conv_algo_search", "do_copy_in_default_stream", "enable_cuda_graph", "use_tf32", "prefer_nhwc" };
    const char* cv[] = { deviceId8.c_str(), "EXHAUSTIVE",          "0",                          "1",                 "1",        "1" };
    Ort::Status st(Ort::GetApi().UpdateCUDAProviderOptions(cudaOpts, ck, cv, 6));
    if (st.IsOK())
        st = Ort::Status(Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(opts, cudaOpts));
    Ort::GetApi().ReleaseCUDAProviderOptions(cudaOpts);
    if (!st.IsOK()) throw Ort::Exception(st.GetErrorMessage(), st.GetErrorCode());
}

// AMD/Intel/any DX12 GPU, via Windows' own DirectML runtime (no vendor SDK required).
// No engine build/cache step, unlike TensorRT — sessions are ready after the warmup runs.
// deviceId is a DXGI adapter enumeration index, not a CUDA ordinal — the caller is
// responsible for translating its D3D device's adapter LUID into that index (see
// AIAimbot.DirectXCapture.cs on the C# side), since DirectML's own docs warn adapter 0 is
// often not the most performant GPU on hybrid-graphics/multi-GPU systems.
static void AppendDirectML(Ort::SessionOptions& opts, int deviceId)
{
    // Required by DirectML: ORT's default memory-pattern/parallel-exec optimizations assume
    // CPU-shaped memory reuse that doesn't hold for the DML allocator.
    opts.DisableMemPattern();
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    const OrtDmlApi* dmlApi = nullptr;
    Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dmlApi)));
    if (!dmlApi) throw std::runtime_error("DirectML execution provider API not available in this onnxruntime build");
    Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML(opts, deviceId));
}

static Ort::Session* BuildSession(Ort::Env& env, const void* modelData, size_t modelSize,
                                   const wchar_t* modelPath, const std::string& cacheDir8,
                                   const std::string& prefix8, int provider, int deviceId)
{
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    if (provider == AI_PROVIDER_DIRECTML)
    {
        AppendDirectML(opts, deviceId);
    }
    else if (provider == AI_PROVIDER_CUDA)
    {
        opts.SetIntraOpNumThreads(4);
        AppendCuda(opts, deviceId);
    }
    else
    {
        // No CUDA-EP fallback for ops TensorRT can't natively handle: RatEngine's own models
        // all build cleanly as pure TensorRT, and dropping this makes the CUDA-only cuDNN
        // engine-selection/nvrtc DLLs (~970MB) genuinely unreachable rather than a rare-path
        // dependency nothing exercises. A model that doesn't fully build under TensorRT now
        // fails AiCreate outright, which the caller already handles by falling back to
        // DirectML at a higher level (see AIAimbot.ModelManagement.cs).
        AppendTensorRT(opts, cacheDir8, prefix8, deviceId);
    }

    if (modelData && modelSize > 0)
        return new Ort::Session(env, modelData, modelSize, opts);
    else
        return new Ort::Session(env, modelPath, opts);
}

// Same choice the app makes when parsing: "output0" if present, else the first rank-3 output.
static size_t PickDetectionOutput(Ort::Session& session)
{
    Ort::AllocatorWithDefaultOptions alloc;
    size_t count = session.GetOutputCount();
    for (size_t i = 0; i < count; i++)
        if (std::string(session.GetOutputNameAllocated(i, alloc).get()) == "output0") return i;
    for (size_t i = 0; i < count; i++)
        if (session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape().size() == 3) return i;
    return 0;
}

static AiContext* CreateNvidiaContext(const void* modelData, size_t modelSize, const wchar_t* modelPath,
                                      const wchar_t* engineCacheDir, const wchar_t* cachePrefix,
                                      int imageSize, int provider, int deviceId, int* outOutputElements)
{
    auto* ctx = new AiContext();
    ctx->backend = provider;
    try
    {
        ctx->env     = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "AiDll");
        ctx->runOpts = new Ort::RunOptions();

        std::string cacheDir8, prefix8;
        if (!WideToUtf8(engineCacheDir, cacheDir8) || !WideToUtf8(cachePrefix, prefix8))
            throw std::runtime_error("Path conversion failed");

        ctx->inputBytes = (size_t)imageSize * imageSize * 3 * sizeof(float);

        ctx->session = BuildSession(*ctx->env, modelData, modelSize, modelPath, cacheDir8, prefix8, provider, deviceId);

        size_t outIndex = PickDetectionOutput(*ctx->session);
        Ort::AllocatorWithDefaultOptions alloc;
        ctx->inputName  = ctx->session->GetInputNameAllocated(0, alloc).get();
        ctx->outputName = ctx->session->GetOutputNameAllocated(outIndex, alloc).get();

        CheckCuda(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking), "cudaStreamCreate");

        // Device buffers at stable addresses — must not move after IoBinding is set up
        CheckCuda(cudaMallocF(&ctx->d_input, ctx->inputBytes), "cudaMalloc(input)");
        cudaMemset(ctx->d_input, 0, ctx->inputBytes);

        // Pinned host input buffer only — output goes directly to the caller's pinned buffer
        CheckCuda(cudaMallocHF(&ctx->h_input, ctx->inputBytes), "cudaMallocHost(input)");
        memset(ctx->h_input, 0, ctx->inputBytes);

        ctx->cudaMemInfo = new Ort::MemoryInfo(
            "Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);

        // IoBinding with stable device-side tensors — required for CUDA graph capture
        ctx->binding = new Ort::IoBinding(*ctx->session);

        std::vector<int64_t> inShape = { 1, 3, imageSize, imageSize };
        auto inVal = Ort::Value::CreateTensor(*ctx->cudaMemInfo,
            ctx->d_input, ctx->inputBytes, inShape.data(), inShape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        ctx->binding->BindInput(ctx->inputName.c_str(), inVal);

        std::vector<int64_t> shape = ctx->session->GetOutputTypeInfo(outIndex).GetTensorTypeAndShapeInfo().GetShape();
        if (!shape.empty() && shape[0] <= 0) shape[0] = 1;
        bool dynamic = std::any_of(shape.begin() + (shape.empty() ? 0 : 1), shape.end(), [](int64_t d) { return d <= 0; });
        if (dynamic)
        {
            // The exported shape has symbolic dims: run once with an ORT-allocated output to learn
            // the real shape, then bind a fixed buffer of that size. Happens before the warmup runs,
            // so CUDA graph capture only ever sees the final bindings.
            ctx->binding->BindOutput(ctx->outputName.c_str(), *ctx->cudaMemInfo);
            ctx->session->Run(*ctx->runOpts, *ctx->binding);
            ctx->binding->SynchronizeOutputs();
            auto probe = ctx->binding->GetOutputValues();
            shape = probe[0].GetTensorTypeAndShapeInfo().GetShape();
            ctx->binding->ClearBoundOutputs();
        }
        ctx->outputShape = shape;

        int64_t total = 1;
        for (size_t i = 1; i < shape.size(); i++) total *= shape[i];
        ctx->outputElements = (int)total;
        ctx->outputBytes    = (size_t)total * sizeof(float);

        CheckCuda(cudaMallocF(&ctx->d_output, ctx->outputBytes), "cudaMalloc(output)");
        cudaMemset(ctx->d_output, 0, ctx->outputBytes);

        auto outVal = Ort::Value::CreateTensor(*ctx->cudaMemInfo,
            ctx->d_output, ctx->outputBytes, shape.data(), shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        ctx->binding->BindOutput(ctx->outputName.c_str(), outVal);

        // Warmup — triggers TRT engine build and ORT's internal CUDA graph capture
        for (int i = 0; i < 5; i++)
            ctx->session->Run(*ctx->runOpts, *ctx->binding);
        ctx->binding->SynchronizeOutputs();

        if (outOutputElements) *outOutputElements = ctx->outputElements;
        g_liveContexts.fetch_add(1);
        return ctx;
    }
    catch (const std::exception& e)
    {
        SetError(e.what());
        delete ctx;
        return nullptr;
    }
}

static AiContext* CreateDirectMLContext(const void* modelData, size_t modelSize, const wchar_t* modelPath,
                                        int imageSize, int deviceId, int* outOutputElements)
{
    auto* ctx = new AiContext();
    ctx->backend = AI_PROVIDER_DIRECTML;
    try
    {
        ctx->env     = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "AiDll");
        ctx->runOpts = new Ort::RunOptions();

        Ort::SessionOptions opts;
        AppendDirectML(opts, deviceId);

        ctx->session = (modelData && modelSize > 0)
            ? new Ort::Session(*ctx->env, modelData, modelSize, opts)
            : new Ort::Session(*ctx->env, modelPath, opts);

        size_t outIndex = PickDetectionOutput(*ctx->session);
        Ort::AllocatorWithDefaultOptions alloc;
        ctx->inputName  = ctx->session->GetInputNameAllocated(0, alloc).get();
        ctx->outputName = ctx->session->GetOutputNameAllocated(outIndex, alloc).get();

        ctx->inputBytes   = (size_t)imageSize * imageSize * 3 * sizeof(float);
        ctx->dmlImageSize = imageSize;
        ctx->dmlInput.assign((size_t)imageSize * imageSize * 3, 0.0f);

        std::vector<int64_t> inShape = { 1, 3, imageSize, imageSize };
        Ort::MemoryInfo cpuMem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        auto runOnce = [&](std::vector<int64_t>& outShape) -> Ort::Value
        {
            auto inVal = Ort::Value::CreateTensor(cpuMem, ctx->dmlInput.data(), ctx->dmlInput.size() * sizeof(float),
                inShape.data(), inShape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            const char* inNames[]  = { ctx->inputName.c_str() };
            const char* outNames[] = { ctx->outputName.c_str() };
            auto results = ctx->session->Run(*ctx->runOpts, inNames, &inVal, 1, outNames, 1);
            outShape = results[0].GetTensorTypeAndShapeInfo().GetShape();
            return std::move(results[0]);
        };

        // Warmup — also resolves any symbolic output dims from the real run.
        std::vector<int64_t> shape;
        for (int i = 0; i < 3; i++) runOnce(shape);
        if (!shape.empty() && shape[0] <= 0) shape[0] = 1;
        ctx->outputShape = shape;

        int64_t total = 1;
        for (size_t i = 1; i < shape.size(); i++) total *= shape[i];
        ctx->outputElements = (int)total;
        ctx->outputBytes    = (size_t)total * sizeof(float);

        if (outOutputElements) *outOutputElements = ctx->outputElements;
        g_liveContexts.fetch_add(1);
        return ctx;
    }
    catch (const std::exception& e)
    {
        SetError(e.what());
        delete ctx;
        return nullptr;
    }
}

static int RunInferenceNvidia(AiContext* ctx, float* outputData, int outputElements)
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
        ctx->session->Run(*ctx->runOpts, *ctx->binding);
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

static int RunInferenceDirectML(AiContext* ctx, const float* inputData, float* outputData, int outputElements)
{
    try
    {
        if (inputData != ctx->dmlInput.data())
            memcpy(ctx->dmlInput.data(), inputData, ctx->inputBytes);

        Ort::MemoryInfo cpuMem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> inShape = { 1, 3, ctx->dmlImageSize, ctx->dmlImageSize };

        auto inVal = Ort::Value::CreateTensor(cpuMem, ctx->dmlInput.data(), ctx->dmlInput.size() * sizeof(float),
            inShape.data(), inShape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        const char* inNames[]  = { ctx->inputName.c_str() };
        const char* outNames[] = { ctx->outputName.c_str() };
        auto results = ctx->session->Run(*ctx->runOpts, inNames, &inVal, 1, outNames, 1);

        auto info = results[0].GetTensorTypeAndShapeInfo();
        int n = std::min(outputElements, (int)info.GetElementCount());
        memcpy(outputData, results[0].GetTensorData<float>(), (size_t)n * sizeof(float));
        return 0;
    }
    catch (const std::exception& ex) { SetError(ex.what()); return -1; }
}

static void AppendJsonString(std::string& o, const std::string& s)
{
    o += '"';
    for (unsigned char c : s)
    {
        switch (c)
        {
        case '"':  o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n";  break;
        case '\r': o += "\\r";  break;
        case '\t': o += "\\t";  break;
        default:
            if (c < 0x20) { char buf[8]; snprintf(buf, sizeof(buf), "\\u%04x", c); o += buf; }
            else o += (char)c;
        }
    }
    o += '"';
}

static void AppendNodes(std::string& o, Ort::Session& s, bool inputs)
{
    Ort::AllocatorWithDefaultOptions alloc;
    size_t count = inputs ? s.GetInputCount() : s.GetOutputCount();
    o += '[';
    for (size_t i = 0; i < count; i++)
    {
        if (i) o += ',';
        std::string name = inputs ? s.GetInputNameAllocated(i, alloc).get() : s.GetOutputNameAllocated(i, alloc).get();
        auto info = inputs ? s.GetInputTypeInfo(i) : s.GetOutputTypeInfo(i);
        o += "{\"name\":";
        AppendJsonString(o, name);
        o += ",\"dims\":[";
        if (info.GetONNXType() == ONNX_TYPE_TENSOR)
        {
            auto dims = info.GetTensorTypeAndShapeInfo().GetShape();
            for (size_t d = 0; d < dims.size(); d++) { if (d) o += ','; o += std::to_string(dims[d]); }
        }
        o += "]}";
    }
    o += ']';
}

extern "C"
{

AIDLL_API void* AiCreate(const wchar_t* modelPath, const uint8_t* modelData, size_t modelSize,
                         const wchar_t* engineCacheDir, const wchar_t* cachePrefix,
                         int imageSize, int provider, int deviceId, int* outOutputElements)
{
    if (!EnsureBackend(provider)) return nullptr;
    if (provider == AI_PROVIDER_DIRECTML)
        return CreateDirectMLContext(modelData, modelSize, modelPath, imageSize, deviceId, outOutputElements);
    return CreateNvidiaContext(modelData, modelSize, modelPath, engineCacheDir, cachePrefix, imageSize, provider, deviceId, outOutputElements);
}

AIDLL_API const char* AiInspect(const wchar_t* modelPath, const uint8_t* modelData, size_t modelSize)
{
    // Either build's CPU EP is sufficient for reading metadata regardless of which provider
    // will actually run inference, so don't force a backend the machine may not support:
    // reuse whichever is already loaded (switching is refused while contexts are live), else
    // prefer Nvidia (it's what Create will want on an Nvidia machine, avoiding a reload) and
    // fall back to DirectML when the CUDA runtime isn't there (AMD/Intel).
    bool backendReady;
    if (g_backend != Backend::None)
        backendReady = true;
    else
        backendReady = EnsureBackend(AI_PROVIDER_CUDA) || EnsureBackend(AI_PROVIDER_DIRECTML);
    if (!backendReady) return nullptr;
    try
    {
        Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "AiInspect" };
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        opts.SetIntraOpNumThreads(1);
        Ort::Session s = (modelData && modelSize > 0) ? Ort::Session(env, modelData, modelSize, opts)
                                                       : Ort::Session(env, modelPath, opts);

        std::string o = "{\"inputs\":";
        AppendNodes(o, s, true);
        o += ",\"outputs\":";
        AppendNodes(o, s, false);
        o += ",\"names\":";
        Ort::AllocatorWithDefaultOptions alloc;
        auto names = s.GetModelMetadata().LookupCustomMetadataMapAllocated("names", alloc);
        if (names) AppendJsonString(o, names.get());
        else o += "null";
        o += '}';

        g_inspectJson = std::move(o);
        return g_inspectJson.c_str();
    }
    catch (const std::exception& e)
    {
        SetError(e.what());
        return nullptr;
    }
}

AIDLL_API int AiGetOutputShape(void* handle, int64_t* dims, int maxDims)
{
    if (!handle) return 0;
    auto* ctx = reinterpret_cast<AiContext*>(handle);
    int rank = (int)ctx->outputShape.size();
    for (int i = 0; i < rank && i < maxDims; i++) dims[i] = ctx->outputShape[i];
    return rank;
}

AIDLL_API float* AiGetInputBuffer(void* handle)
{
    if (!handle) return nullptr;
    return reinterpret_cast<AiContext*>(handle)->h_input;
}

AIDLL_API int AiRunFromPinned(void* handle, float* outputData, int outputElements)
{
    if (!handle || !outputData) { SetError("null arg"); return -1; }
    auto* ctx = reinterpret_cast<AiContext*>(handle);
    if (!ctx->h_input) { SetError("AiRunFromPinned is not supported by this backend; use AiRun"); return -1; }
    return RunInferenceNvidia(ctx, outputData, outputElements);
}

AIDLL_API int AiRun(void* handle, const float* inputData, float* outputData, int outputElements)
{
    if (!handle || !inputData || !outputData) { SetError("null arg"); return -1; }
    auto* ctx = reinterpret_cast<AiContext*>(handle);
    if (ctx->backend == AI_PROVIDER_DIRECTML)
        return RunInferenceDirectML(ctx, inputData, outputData, outputElements);
    memcpy(ctx->h_input, inputData, ctx->inputBytes);
    return RunInferenceNvidia(ctx, outputData, outputElements);
}

AIDLL_API void AiDestroy(void* handle)
{
    if (!handle) return;
    delete reinterpret_cast<AiContext*>(handle);
    g_liveContexts.fetch_sub(1);
}

AIDLL_API int AiShutdown()
{
    if (g_liveContexts.load() != 0) { SetError("AiShutdown called with live contexts"); return -1; }
    if (g_backend == Backend::Nvidia)
    {
        cudaError_t e = cudaDeviceReset();
        if (e != cudaSuccess) { SetError(cudaGetErrorString(e)); return -1; }
    }
    if (g_ortModule)
    {
        FreeLibrary(g_ortModule);
        g_ortModule = nullptr;
    }
    UnloadCudaRuntime();
    g_backend = Backend::None;
    return 0;
}

AIDLL_API const char* AiGetLastError()
{
    return g_lastError.c_str();
}

} // extern "C"
