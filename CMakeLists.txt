cmake_minimum_required(VERSION 3.20)
project(AiInference CXX)

set(CMAKE_CXX_STANDARD 17)

set(DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies")

# ORT C API headers vendored in-repo (see thirdparty/README.md) — no dependency on any
# particular machine's NuGet cache. onnxruntime.dll and onnxruntime_directml.dll themselves
# are loaded dynamically at runtime (see EnsureBackend in AiInference.cpp), not linked.
set(ORT_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnxruntime/include")

add_library(ai_inference SHARED AiInference.cpp AiInference.h)

target_compile_definitions(ai_inference PRIVATE AIDLL_EXPORTS)

target_compile_options(ai_inference PRIVATE
    $<$<CONFIG:Release>:/O2 /Oi /Ot /GL /GS- /arch:AVX2>
)

target_link_options(ai_inference PRIVATE
    $<$<CONFIG:Release>:/LTCG /OPT:REF /OPT:ICF>
)

target_include_directories(ai_inference PRIVATE
    ${ORT_INCLUDE}
    ${DEPS_DIR}
)

target_link_directories(ai_inference PRIVATE ${DEPS_DIR})

target_link_libraries(ai_inference PRIVATE
    cudart.lib
)

set_target_properties(ai_inference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}/bin"
)
