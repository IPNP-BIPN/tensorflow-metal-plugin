/* Copyright 2026 The TensorFlow Metal Plugin Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// The out-of-tree entry points.
//
// TensorFlow dlopens this library and looks up two symbols by name. Everything
// behind them is the same code an in-tree build registers through
// PluggableDeviceInit_Api, so this file is the whole difference between the
// two forms: there, TensorFlow holds the function pointers already and calls
// them directly; here, it resolves them out of a shared object.

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernels.h"
#include "tensorflow/core/common_runtime/metal/metal_graph.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_profiler.h"

namespace {

// The TensorFlow release this library was compiled against, from
// TF_SUPPORTED_VERSION at the repository root by way of the Makefile.
#ifndef TF_METAL_SUPPORTED_TF_VERSION
#error "TF_METAL_SUPPORTED_TF_VERSION must be defined by the build"
#endif

bool TruthyEnvironmentVariable(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return false;
  return std::strcmp(value, "0") != 0;
}

// The version of the TensorFlow that has just dlopened us, or nullptr.
//
// Looked up rather than called, for the same reason the kernels look up the
// resource variable entry points: this library is linked with TF_CAPI_WEAK,
// so an ordinary call would bind to null rather than to the real function.
const char* LoadedTensorFlowVersion() {
  using TFVersionFn = const char* (*)();
  auto* fn = reinterpret_cast<TFVersionFn>(dlsym(RTLD_DEFAULT, "TF_Version"));
  return fn == nullptr ? nullptr : fn();
}

// Compares major.minor only, so a patch release is not treated as a different
// TensorFlow.
bool SameMajorMinor(const char* a, const char* b) {
  int a_major = 0, a_minor = 0, b_major = 0, b_minor = 0;
  if (std::sscanf(a, "%d.%d", &a_major, &a_minor) != 2) return false;
  if (std::sscanf(b, "%d.%d", &b_major, &b_minor) != 2) return false;
  return a_major == b_major && a_minor == b_minor;
}

// Why the backend should not offer a device this run, or nullptr to proceed.
//
// Decided once, because it is read from three entry points and reading the
// environment twice could disagree.
const char* StandDownReason() {
  static const char* const reason = [] () -> const char* {
    static char buffer[640];

    if (TruthyEnvironmentVariable("TF_DISABLE_METAL")) {
      std::snprintf(buffer, sizeof(buffer),
                    "Metal: backend disabled by TF_DISABLE_METAL, so no GPU "
                    "device is offered. Unset it to re-enable.");
      return buffer;
    }

    if (TruthyEnvironmentVariable("TF_METAL_SKIP_VERSION_CHECK")) return nullptr;

    const char* loaded = LoadedTensorFlowVersion();
    // Nothing to compare against. A build that does not export TF_Version
    // still has a working PluggableDevice API, and standing down over a check
    // that could not run would be worse than the mismatch it guards against.
    if (loaded == nullptr) return nullptr;

    const char* built = TF_METAL_SUPPORTED_TF_VERSION;
    if (SameMajorMinor(loaded, built)) return nullptr;

    // Every struct crossing the PluggableDevice boundary carries a struct_size
    // that each side fills in from its own headers, so a plugin compiled
    // against different headers than the TensorFlow loading it goes wrong at a
    // wrong field offset: a crash, or a wrong answer, a long way from here.
    // Saying so at load is the difference between a one-line fix and an
    // afternoon.
    std::snprintf(buffer, sizeof(buffer),
                  "Metal: this plugin was built against TensorFlow %s and is "
                  "loaded into TensorFlow %s, so no GPU device is offered. "
                  "The PluggableDevice C API matches its structs by size and "
                  "the two do not interoperate. Rebuild the plugin against "
                  "this TensorFlow (pip install --force-reinstall "
                  "--no-build-isolation tensorflow-metal-plugin), or set "
                  "TF_METAL_SKIP_VERSION_CHECK=1 to load it anyway.",
                  built, loaded);
    return buffer;
  }();
  return reason;
}

// Reports no devices, which is how this plugin declines.
//
// Returning a non-OK status from SE_InitPlugin is not a way to decline:
// TensorFlow's TF_LoadPluggableDeviceLibrary turns a failed registration into
// a CHECK failure, which aborts the process. Since the plugin is loaded from
// site-packages/tensorflow-plugins during `import tensorflow`, that would turn
// "the plugin is unhappy" into "TensorFlow cannot be imported", leaving the
// user without even a working interpreter to uninstall it from. Registering
// normally and offering zero devices leaves TensorFlow running on the CPU,
// which is exactly what a user who disabled the backend asked for.
void NoDevices(const SP_Platform* platform, int* device_count,
               TF_Status* status) {
  *device_count = 0;
  TF_SetStatus(status, TF_OK, "");
}

}  // namespace

extern "C" {

void SE_InitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status) {
  tensorflow::metal::MetalInitPlugin(params, status);
  if (TF_GetCode(status) != TF_OK) return;

  const char* reason = StandDownReason();
  if (reason != nullptr) {
    std::fprintf(stderr, "%s\n", reason);
    // The platform stays registered, with nothing behind it. Everything else
    // the plugin installs (the kernels, the profiler, the graph pass) is keyed
    // to a GPU device that now does not exist, so none of it runs.
    params->platform_fns->get_device_count = NoDevices;
  }
}

void TF_InitKernel() { tensorflow::metal::RegisterAllMetalKernels(); }

// Optional: TensorFlow looks this up and carries on without it if absent. It
// is what puts a Metal row in the trace viewer next to the host's.
void TF_InitProfiler(TF_ProfilerRegistrationParams* params,
                     TF_Status* status) {
  tensorflow::metal::MetalInitProfiler(params, status);
}

// Optional, like the profiler. Fuses a bias and an activation into the
// convolution or matrix multiply in front of them, which this backend has
// kernels for and which nothing else produces for a pluggable device.
void TF_InitGraph(TP_OptimizerRegistrationParams* params, TF_Status* status) {
  tensorflow::metal::MetalInitGraph(params, status);
}

}  // extern "C"
