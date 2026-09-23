#pragma once
#ifdef AIDLL_EXPORTS
#define AIDLL_API __declspec(dllexport)
#else
#define AIDLL_API __declspec(dllimport)
#endif
#include <stdint.h>
#include <stddef.h>

#define AI_PROVIDER_TENSORRT 0
#define AI_PROVIDER_CUDA     1
#define AI_PROVIDER_DIRECTML 2

extern "C" {

// Loads from modelData when modelSize > 0, otherwise from modelPath. provider is one of
// AI_PROVIDER_*. engineCacheDir/cachePrefix are only used by TensorRT (ignored otherwise).
// Switching provider *family* (Nvidia CUDA/TensorRT vs DirectML) requires every previous
// handle to be destroyed first — this dynamically loads a different onnxruntime.dll build
// for each family, so it can't coexist with a live session from the other one.
AIDLL_API void* AiCreate(const wchar_t* modelPath,
                         const uint8_t* modelData,
                         size_t modelSize,
                         const wchar_t* engineCacheDir,
                         const wchar_t* cachePrefix,
                         int imageSize,
                         int provider,
                         int* outOutputElements);

// Reads model metadata on a CPU-only session (Nvidia onnxruntime build). Returns UTF-8 JSON
// {"inputs":[{"name","dims"}],"outputs":[{"name","dims"}],"names":string|null},
// valid until the next AiInspect call on the same thread; nullptr on failure.
AIDLL_API const char* AiInspect(const wchar_t* modelPath, const uint8_t* modelData, size_t modelSize);

// Actual shape of the bound detection output (dynamic dims resolved). Returns the rank.
AIDLL_API int AiGetOutputShape(void* handle, int64_t* dims, int maxDims);

// Returns pointer to the pinned host input buffer — write float data here,
// then call AiRunFromPinned. Null for DirectML (no pinned host buffer; use AiRun).
AIDLL_API float* AiGetInputBuffer(void* handle);

// Run inference using whatever is already in the pinned input buffer (Nvidia providers only).
AIDLL_API int AiRunFromPinned(void* handle, float* outputData, int outputElements);

// Run inference copying from an arbitrary host pointer. Works for every provider.
AIDLL_API int AiRun(void* handle,
                    const float* inputData,
                    float* outputData,
                    int outputElements);

AIDLL_API void AiDestroy(void* handle);

// Releases everything the current backend left in the process (CUDA context reset for
// Nvidia, nothing extra for DirectML) and frees its onnxruntime module. Fails (-1) while
// any handle from AiCreate is still alive.
AIDLL_API int AiShutdown();

AIDLL_API const char* AiGetLastError();

} // extern "C"
