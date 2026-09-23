# Vendored headers

Only the ONNX Runtime C/C++ API headers are vendored here — small text files, so they're
checked into git for a reproducible build across machines. The actual runtime DLLs
(`onnxruntime.dll`, `onnxruntime_directml.dll`, and everything they load) are **not**
vendored; they come from NuGet and are copied into `dependencies\` locally, or by
RatEngine's publish step for a real build.

`onnxruntime/include/` is the header set from `Microsoft.ML.OnnxRuntime.Gpu.Windows` 1.25.1
(used to build against the CUDA/TensorRT providers), plus `dml_provider_factory.h` copied in
from `Microsoft.ML.OnnxRuntime.DirectML` 1.24.4 (DirectML provider — AMD/Intel/Nvidia via
DX12). They can't both be on the include path as separate directories: `dml_provider_factory.h`
does `#include "onnxruntime_c_api.h"`, a quoted include that resolves to whichever directory
it's sitting in, so two directories each containing their own (slightly different, since the
two packages are different ORT versions) copy of that file causes duplicate/conflicting type
definitions. Copying just the one DirectML-specific header alongside the CUDA package's
headers avoids that; `dml_provider_factory.h`'s own definitions are stable enough across
versions that this is safe.

`AiInference.cpp` never links against either build's `onnxruntime.lib` — it loads whichever
`onnxruntime*.dll` the selected provider needs at runtime via `LoadLibrary`/`GetProcAddress`,
so one binary can switch between the Nvidia and DirectML backends without being import-bound
to either (see `EnsureBackend`). Because of this, it also can't use the compiled header's
`ORT_API_VERSION` macro (25, from the newer CUDA package) when calling `GetApi()` — the older
DirectML build (API version 24) would reject that request. `AiInference.cpp` hardcodes 24
instead; bump that only once both packages support a shared newer version.

To refresh headers after bumping a package version, restore a throwaway project referencing
it and copy `build(Transitive)/native/include/*.h` from the NuGet cache here.
