# Audit

Phase 0. A survey of the repository as it stands, with everything below
measured on this machine rather than read off the README.

Measured on: Apple M4 Max, macOS 26.6 (Darwin 27.0.0), Xcode Command Line
Tools SDK `MacOSX.sdk`, Python 3.12.14 at
`/opt/homebrew/opt/python@3.12/bin/python3.12`, `tensorflow==2.20.0`,
`numpy==2.5.1`, `keras==3.13.2`. Repository at `7f2e2b9`, working tree clean.

## Summary

The headline is that this is not a repository that needs rescuing. It builds
clean, all four PluggableDevice entry points are exported, 342 ops register,
the correctness sweep reports 323 verified and 0 mismatches, CPU fallback for
unregistered ops works, a Keras training step runs on the GPU, and `pip wheel`
produces a wheel with the dylib in `tensorflow-plugins/`.

The real problems are different from the ones the mission brief anticipated:

1. **The TensorFlow version is not pinned.** `install_requires` says
   `tensorflow>=2.16` and CI installs `tensorflow>=2.16`. Nothing in the tree
   asserts a version at build or load time. This is hard constraint 3 and it
   is currently unmet.
2. **The scope is roughly 3x what the brief calls for.** 342 registered ops
   across 36,098 lines of kernel code, maintained by one person. The brief's
   target list is 100% covered by about 8,000 lines of it; the other 28,000
   support ops nothing in the brief asks for.
3. **Training is slower than it looks, for a reason outside this repo.**
   TensorFlow 2.20.0 does not export the resource-variable kernel C API, so
   the plugin forces every Metal kernel to wait for the GPU before returning.
   Fifteen ops, including every optimiser, are not registered at all and run
   on the host.
4. **My benchmark numbers do not match the README's.** Two cases are now
   slower than CPU where the README reports them faster.
5. **Deployment target and API use disagree.** The build targets macOS 13.0
   but calls macOS 15.0 APIs without availability guards, 20 warnings worth.

Detail below.

## Build system, and whether a clean build succeeds

Plain GNU `make` (`Makefile`, 108 lines) driving `clang++` directly over
`src/**/*.mm` and `src/plugin_init.cc`. No bazel, no CMake. Header and library
paths are resolved from the installed TensorFlow at configure time.
`setup.py` wraps the same `make` in a `build_py` subclass so that `pip install`
compiles against the installing interpreter's TensorFlow.

**A clean build succeeds.**

```
rm -rf build && make PYTHON=/opt/homebrew/opt/python@3.12/bin/python3.12 -j8
built build/libmetal_plugin.dylib
```

165s user, 182s wall on 8 cores. 71 objects, 1.3 MB dylib. Zero errors.

One gotcha that cost time and is worth writing down: `make` with no `PYTHON=`
fails on this machine with

```
Makefile:18: *** TensorFlow was not found. Install it first, or pass TF_INCLUDE and TF_LIB explicitly.  Stop.
```

because `python3` in `make`'s `/bin/sh` resolves to the pyenv shim
(`/Users/benjamin/.pyenv/versions/3.12.12/bin/python3`), which has no
TensorFlow, while `python3` in the interactive zsh is aliased to the Homebrew
one, which does. The error message is correct and the diagnosis is not
obvious. Worth having the Makefile print which interpreter it tried.

### Warnings

20, all `-Wunguarded-availability-new`, all the same shape:

| Count | API | Available from | Files |
| ---: | --- | --- | --- |
| 10 | `MPSGraphConvolution3DOpDescriptor`, `convolution3DWithSourceTensor:`, `MPSGraphTensorNamedDataLayout{NDHWC,NCDHW,DHWIO}` and the two 3D gradients | macOS 13.2 | `metal_conv3d_ops.mm`, `metal_conv_generic_ops.mm` |
| 5 | `initWithBuffer:offset:descriptor:`, `setPreferPackedRows:`, `setMathMode:`, `reciprocalSquareRootWithTensor:name:` (x2) | macOS 15.0 | `metal_mps_graph.mm`, `metal_shader_library.mm`, `metal_elementwise_ops.mm`, `metal_batch_norm_ops.mm` |
| 1 | `MPSDataTypeBFloat16` | macOS 14.0 | `metal_mps_graph.mm` |

`-mmacosx-version-min=13.0` claims support for macOS 13 and 14. On those
systems these are null selectors and unrecognised symbols. The README says the
macOS 15 SDK is required to *build*; nothing says macOS 15 is required to
*run*, and the deployment target says otherwise. Either the target moves to
15.0 or the calls get `@available` guards. This is a latent crash on the two
OS versions the build currently claims.

## Exported entry points

`nm -gU build/libmetal_plugin.dylib`:

| Symbol | Exported | Source |
| --- | --- | --- |
| `SE_InitPlugin` | yes | [plugin_init.cc:49](src/plugin_init.cc#L49) |
| `TF_InitKernel` | yes | [plugin_init.cc:58](src/plugin_init.cc#L58) |
| `TF_InitProfiler` | yes | [plugin_init.cc:62](src/plugin_init.cc#L62) |
| `TF_InitGraph` | yes | [plugin_init.cc:75](src/plugin_init.cc#L75) |

All four. `make check-symbols` passes: both required symbols exported, and no
`tensorflow::metal` symbol left undefined.

139 symbols exported in total. 135 of those are C++ internals of
`tensorflow::metal` that have no reason to be public. Not a bug, but an
unnecessarily wide surface for a plugin whose entire contract is four C
functions; `-fvisibility=hidden` plus explicit `__attribute__((visibility("default")))`
on the four would cut it to four.

### struct_size

Set at all 14 sites where a struct crosses the boundary, always from the
header's own macro (`SP_DEVICE_STRUCT_SIZE`, `TP_OPTIMIZER_STRUCT_SIZE`, and
so on), so the value always matches the headers the object was compiled
against. Since the dylib is compiled per-install against the target
TensorFlow, this is consistent by construction.

There is **no runtime check** that the TensorFlow now loading the plugin is
the one it was compiled against. The Makefile guards the build side of this
(the `$(BUILD)/tf-include-path` stamp, added after a real incident: linking
objects built against two different TensorFlows produced a missing absl
symbol). Nothing guards the load side. A user who upgrades TensorFlow in place
without rebuilding gets undefined behaviour.

## Which TensorFlow version, and how it is pinned

**It is not pinned.** Three places disagree:

| Place | Says |
| --- | --- |
| [setup.py:98](setup.py#L98) | `install_requires=["tensorflow>=2.16"]` |
| [.github/workflows/ci.yml:41](.github/workflows/ci.yml#L41) | `pip install "tensorflow>=2.16"` |
| [README.md:49](README.md#L49) | verified against `tensorflow==2.20.0` |
| [README.md:88](README.md#L88) | install verified against `tensorflow==2.21.0` |
| [README.md:123](README.md#L123) | benchmarks measured against `tensorflow==2.21.0` |

Installed here and used for every number in this document: **2.20.0**.

Nothing has ever been built against 2.16, 2.17 or 2.18 in CI, because CI
resolves `>=2.16` to the newest release. The floor is a guess. `pyproject.toml`
deliberately keeps TensorFlow out of `[build-system] requires`, which is
correct (it avoids pip downloading a second copy into the isolated build
environment) and is separate from the pinning question.

The plugin does adapt to the installed headers in one place:
[tools/probe_stream_options.sh](tools/probe_stream_options.sh) compiles a
one-line probe for `SP_StreamOptions` and defines
`TF_METAL_NO_STREAM_OPTIONS` when absent. On 2.20.0 the probe returns `no`.

## Registered kernels

342 ops, obtained by snapshotting
`tensorflow.python.framework.kernels.get_all_registered_kernels()` before and
after `load_pluggable_device_library`, and diffing the `GPU` device type.
TensorFlow itself registers 108 `GPU` ops before the plugin loads; 448 after.
340 ops are new, 2 (`Cast`, `Identity`) gain registrations.

Source locations are the first quoted occurrence of the op name in
`src/.../kernels/`, which is where the registration list holds it. All 342
resolved.

The full table is in [Appendix: every registered kernel](#appendix-every-registered-kernel).

By file, with op counts:

| File | LOC | Ops |
| --- | ---: | ---: |
| `metal_shader_library.mm` | 2485 | 0 (the Metal source string itself) |
| `metal_rnn_ops.mm` | 1559 | 8 |
| `metal_cudnn_rnn_ops.mm` | 1466 | 11 |
| `metal_matrix_ops.mm` | 965 | 20 |
| `metal_quant_ops.mm` | 940 | 6 |
| `metal_elementwise_ops.mm` | 857 | 50 |
| `metal_sparse_manip_ops.mm` | 836 | 10 |
| `metal_batch_norm_ops.mm` | 836 | 8 |
| `metal_compare_ops.mm` | 806 | 16 |
| `metal_slice_ops.mm` | 792 | 8 |
| `metal_batch_space_ops.mm` | 727 | 5 |
| `metal_quantize_dequantize_ops.mm` | 707 | 5 |
| `metal_index_ops.mm` | 689 | 6 |
| `metal_nn_ops.mm` | 685 | 5 |
| `metal_misc_ops.mm` | 685 | 5 |
| `metal_fft_ops.mm` | 667 | 22 |
| `metal_linalg_ops.mm` | 661 | 5 |
| `metal_strided_ops.mm` | 639 | 4 |
| `metal_array_ops.mm` | 639 | 6 |
| `metal_fused_ops.mm` | 637 | 9 |
| `metal_gather_scatter_ops.mm` | 631 | 1 |
| `metal_extra_ops.mm` | 627 | 3 |
| `metal_image2_ops.mm` | 624 | 6 |
| `metal_maxpool_argmax_ops.mm` | 589 | 5 |
| `metal_alias_ops.mm` | 574 | 5 |
| `metal_search_ops.mm` | 568 | 5 |
| `metal_sparse_segment_ops.mm` | 555 | 12 |
| `metal_activation_ops.mm` | 554 | 11 |
| `metal_depthwise_ops.mm` | 548 | 4 |
| `metal_dynamic_ops.mm` | 547 | 5 |
| `metal_dilation_ops.mm` | 518 | 3 |
| `metal_training_ops.mm` | 504 | 0 on this TensorFlow (see below) |
| `metal_conv_ops.mm` | 487 | 3 |
| `metal_random_dist_ops.mm` | 477 | 7 |
| `metal_misc2_ops.mm` | 466 | 5 |
| `metal_crop_resize_ops.mm` | 448 | 3 |
| `metal_batch_norm_global_ops.mm` | 439 | 2 (both dead, see below) |
| `metal_conv3d_ops.mm` | 434 | 5 |
| `metal_pool_variant_ops.mm` | 429 | 2 |
| `metal_sparse_ops.mm` | 396 | 2 |
| `metal_random_ops.mm` | 395 | 4 |
| `metal_conv_generic_ops.mm` | 385 | 1 |
| `metal_box_proposal_ops.mm` | 366 | 1 |
| `metal_debug_ops.mm` | 362 | 2 |
| `metal_reduction_ops.mm` | 358 | 8 |
| `metal_matmul_op.mm` | 345 | 1 |
| `metal_ctc_ops.mm` | 334 | 2 |
| `metal_pooling_ops.mm` | 330 | 2 |
| `metal_image_ops.mm` | 327 | 2 |
| `metal_mps_graph.mm` | 316 | 0 (shared MPSGraph plumbing) |
| `metal_inplace_ops.mm` | 302 | 0 on this TensorFlow |
| `metal_collective_ops.mm` | 300 | 7 |
| `metal_nms_ops.mm` | 293 | 3 |
| `metal_transform_ops.mm` | 279 | 2 |
| `metal_bincount_ops.mm` | 274 | 2 |
| `metal_fill_ops.mm` | 272 | 3 |
| `metal_ref_variable_ops.mm` | 269 | 0 on this TensorFlow |
| `metal_resize_grad_ops.mm` | 260 | 2 |
| `metal_kernel_util.mm` | 253 | 0 (shared helpers) |
| `metal_volume_patch_ops.mm` | 245 | 1 |
| `metal_identity_op.mm` | 151 | 1 |
| headers (`*.h`) | 893 | |

Dtype coverage is uniform: almost every kernel is registered for `float32` and
`float16` and nothing else. `int32` and `int64` appear only on index and shape
arguments, mostly pinned to host memory.

## Coverage of the ops named in the brief

Every op on the brief's list is covered. Nothing is missing.

| Brief's item | Status |
| --- | --- |
| MatMul, BatchMatMul (V1/V2/V3) | registered, float32 + float16 |
| Conv2D, Conv2DBackpropInput, Conv2DBackpropFilter | registered, float32 + float16 |
| BiasAdd, BiasAddGrad | registered, float32 + float16 |
| Relu, ReluGrad, Sigmoid, SigmoidGrad, Tanh, TanhGrad | registered, float32 + float16 |
| Gelu | **not a TensorFlow op.** `tf.nn.gelu` decomposes into `Erf`/`Tanh`/`Mul`/`Add`/`Pow`, all of which are registered. Nothing to do. |
| Sum, Mean, Max, Min, Prod | registered, float32 + float16 |
| FusedBatchNorm V1/V2/V3 and gradients | registered |
| Softmax, SoftmaxCrossEntropyWithLogits, SparseSoftmaxCrossEntropyWithLogits | registered |
| Elementwise binary and unary | 50 ops in `metal_elementwise_ops.mm` |
| Transpose, Concat, ConcatV2, Cast, Tile, Slice, Pad, Split | registered |
| Reshape, Pack, Unpack, ExpandDims, Squeeze | **covered by TensorFlow's own `DEVICE_DEFAULT` registrations**, which apply to a pluggable device. The plugin correctly does not duplicate them. |

## CPU fallback for unregistered ops

**Verified, and it works.** Under default settings (soft placement on, which
is the TF2 eager default and is what a user gets), inside `tf.device("/GPU:0")`:

| Op | Result |
| --- | --- |
| `MatrixDeterminant` | ran on `/device:CPU:0` |
| `Svd` | ran on `/device:CPU:0` |
| `MatrixInverse` | ran on `/device:CPU:0` |
| `StringLength` | ran on `/device:CPU:0` |
| `Igamma` | ran on `/device:CPU:0` |
| `SegmentSum` | ran on `/device:CPU:0` |
| `Qr`, `Unique` | ran on `/device:GPU:0` (both are registered) |
| `inv(matmul(x, x))`, eager | ran on `/device:CPU:0`, correct |
| `inv(matmul(x, x))`, inside `tf.function` | ran, result on `/device:GPU:0` |

No crashes, no hangs, no wrong answers. A full Keras `model.fit` (Conv2D,
GlobalAveragePooling2D, Dense, Adam, sparse categorical crossentropy) trains
without error.

The test suite asserts the opposite direction as a control: with soft
placement explicitly off, an op with no GPU kernel must raise
([tests/run_tests.py:103](tests/run_tests.py#L103)). Without that control every
other numeric check could be silently passing on the CPU. Both directions are
therefore covered.

## Existing tests

| What | Where | Lines | State |
| --- | --- | ---: | --- |
| On-device checks | `tests/run_tests.py` | 367 | **passes.** 33 checks: device registration, 8 ops against CPU, soft-placement control, CudnnRNN parameter layout, dropout, float16, CheckNumerics, host round trip, an Adam-shaped variable update, profiler xplane and Metal plane, graph optimiser fusion. |
| StreamExecutor C API, no TensorFlow | `tests/stream_executor_test.mm` | 273 | **passes.** Runs as part of `make test`. Exercises `memset32` with a non-uniform pattern, which no op reaches. |
| Correctness sweep, all registered ops | `tools/op_sweep.py` + `tools/recipes.py` | 1817 | **passes.** See below. |
| Shader compilation | `tools/compile_shaders.py` | 105 | Extracts the Metal source string from `metal_shader_library.mm` and compiles it through the Metal runtime. Not part of `make test`; CI runs it as a separate step. |
| GPU vs CPU timings | `benchmarks/benchmark.py` | 145 | Runs. Numbers below. |
| CI | `.github/workflows/ci.yml` | 63 | Present and plausible. `macos-15`, Python 3.11 and 3.12. |

Nothing is dead. This is better test coverage than the brief assumes exists.

### The sweep

`make sweep` walks all 356 names in `tools/metal_ops.txt`, calls each through
TensorFlow's own dispatch with soft placement off, and compares against the
CPU kernel with identical inputs. Result on this machine:

```
mismatch                 0
gpu-error                0
match                  323
removed-from-tensorflow 19
needs-unexported-api    14
unexercised              0
no-recipe                0
duplicates               0
wrong-attr constraints   0
```

Each op is run twice and required to agree with itself, which is how an
inverse FFT that rewrote its own input was caught previously.

This already *is* the "numerical correctness harness" Phase 1 asks for, and it
is more thorough than what Phase 1 describes: it also rejects duplicate
registrations and registrations that constrain an attribute the op def does
not have, either of which makes an op look registered while being unusable.
It does not report max absolute and relative error per shape and dtype in a
table, which is the one thing Phase 1 adds.

### 19 ops that no device can run

`recipes.REMOVED_FROM_GRAPHDEF` ([tools/recipes.py:355](tools/recipes.py#L355)).
These are deprecated in their TensorFlow op def, so nothing can emit them.
The plugin still compiles and registers kernels for all of them:

`BatchFFT`, `BatchFFT2D`, `BatchFFT3D`, `BatchIFFT`, `BatchIFFT2D`,
`BatchIFFT3D`, `BatchMatrixBandPart`, `BatchMatrixDiag`,
`BatchMatrixDiagPart`, `BatchMatrixSetDiag`, `BatchMatrixTriangularSolve`,
`QuantizeAndDequantize`, `AdjustContrast`,
`BatchNormWithGlobalNormalization`, `BatchNormWithGlobalNormalizationGrad`,
`Conv3DBackpropFilter`, `Conv3DBackpropInput`, `TopK`, `TileGrad`.

This is unreachable code by definition, and it is the cleanest thing on the
delete list.

### 14 ops the plugin declines to register on a released TensorFlow

`recipes.NEEDS_UNEXPORTED_C_API` ([tools/recipes.py:375](tools/recipes.py#L375)):
`Assign`, `AssignAdd`, `AssignSub`, `ResourceApplyAdam`,
`ResourceApplyGradientDescent`, `ResourceApplyKerasMomentum`,
`ResourceApplyMomentum`, `ResourceApplyRMSProp`, `ResourceGather`,
`ResourceGatherNd`, `ResourceScatterUpdate`, `ParallelConcat`,
`_ParallelConcatStart`, `_ParallelConcatUpdate`.

They need six entry points that TensorFlow's headers declare and its binaries
do not export
([#126374](https://github.com/tensorflow/tensorflow/issues/126374)).
[metal_kernel_util.mm:195](src/tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.mm#L195)
probes with `dlsym` and skips the registrations when they are absent. Confirmed
absent on 2.20.0 here: the warning fires at load.

The consequence is not just "those ops run on the host". Because TensorFlow's
own resource-variable kernels reach device memory through a host-addressable
pointer with no knowledge of in-flight GPU work, the plugin falls back to
**forcing every Metal kernel to wait for the GPU before returning**
([metal_kernel_util.mm:213](src/tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.mm#L213),
applied at [metal_mps_graph.mm:302](src/tensorflow/core/common_runtime/metal/kernels/metal_mps_graph.mm#L302)).
That is the single largest performance fact about this backend today, and it
is not something this repository can fix. `TF_METAL_SYNCHRONOUS` overrides it
in either direction; overriding it to asynchronous on 2.20.0 reproduces `nan`
weights, per the README.

## Benchmarks

`benchmarks/benchmark.py`, TF 2.20.0, this machine, median of 10 after 3 warmups:

| Case | GPU ms | CPU ms | Speedup |
| --- | ---: | ---: | ---: |
| MatMul 1024x1024 | 1.52 | 20.43 | **13.42x** |
| MatMul 2048x2048 | 6.80 | 48.37 | **7.12x** |
| Conv2D batch 16 | 1.62 | 7.05 | **4.35x** |
| Conv2D batch 64 | 6.84 | 25.79 | **3.77x** |
| MatMul 512x512 | 1.02 | 2.51 | 2.45x |
| CNN train step, SGD, batch 128 | 47.19 | 93.06 | 1.97x |
| ReduceSum 4096x4096 | 0.99 | 1.69 | 1.70x |
| Elementwise 4096x4096 | 6.04 | 9.12 | 1.51x |
| CNN forward batch 32 | 10.31 | 10.53 | 1.02x |
| CNN forward batch 128 | 11.61 | 6.95 | **0.60x** |

**These do not match the README's table**, which was measured against 2.21.0.
Notable disagreements:

| Case | README | Measured here |
| --- | ---: | ---: |
| MatMul 2048x2048 | 6.7x | 7.12x |
| MatMul 1024x1024 | 1.6x | 13.42x |
| Elementwise 4096x4096 | 0.5x | 1.51x |
| ReduceSum 4096x4096 | 0.6x | 1.70x |
| CNN forward batch 128 | 1.1x | **0.60x** |
| CNN train step batch 128 | 1.5x | 1.97x |

Most move in the plugin's favour. One does not: `CNN forward batch 128` is now
measurably slower than the CPU. Two readings are possible, a 2.20 versus 2.21
difference or run-to-run noise on a laptop, and the audit cannot tell them
apart from one run each. Either way the README's table is stale enough that it
should not be quoted until re-measured on a pinned version.

There is no `BENCHMARKS.md`. Phase 2 asks for one.

## Packaging

Works. `pip wheel . --no-deps --no-build-isolation` produces

```
tensorflow_metal_plugin-0.2.0-cp312-cp312-macosx_27_0_arm64.whl   497 KB
  tensorflow-plugins/__init__.py
  tensorflow-plugins/libmetal_plugin.dylib     1394984
  ...dist-info/
```

The dylib lands in `tensorflow-plugins/`, which is exactly what TensorFlow
scans at import. `BinaryDistribution.has_ext_modules` correctly forces a
platform tag rather than `py3-none-any`.

Two notes:

- With pip's default build isolation the build fails here, and **it is not
  this repository's fault**: a stale
  `sphinxcontrib_jsmath-1.0.1-py3.7-nspkg.pth` in the Homebrew site-packages
  corrupts `sys.path` in the isolated subprocess, which then cannot
  `import json`. Local environment damage. Worth knowing before it gets
  diagnosed as a packaging bug.
- The platform tag is `macosx_27_0`, taken from the building machine. Since
  there is deliberately no prebuilt wheel (the dylib must be compiled against
  the target TensorFlow), this is harmless, but it does mean the artefact
  produced by `pip wheel` is not redistributable, which is by design and
  should stay documented.

The distribution name is `tensorflow-metal-plugin`. Phase 3 asks for a name
"clearly distinct from `tensorflow-metal`". One hyphenated suffix away from
Apple's abandoned package is not distinct enough, and PyPI normalises both to
similar-looking names. Worth renaming before anything is published.

## Line count by subsystem

| Subsystem | Lines | Share |
| --- | ---: | ---: |
| Kernels (`src/.../kernels/`, 68 files) | 36,098 | 80.4% |
| of which: the Metal shader source string | 2,485 | 5.5% |
| of which: RNN families (`rnn_ops` + `cudnn_rnn_ops`) | 3,025 | 6.7% |
| of which: quantisation (`quant_ops` + `quantize_dequantize_ops`) | 1,647 | 3.7% |
| Device runtime (`src/.../metal/*.mm,*.h`, 12 files) | 2,854 | 6.4% |
| of which: stream executor | 1,081 | 2.4% |
| of which: graph optimiser | 630 | 1.4% |
| of which: profiler | 368 | 0.8% |
| of which: stream | 538 | 1.2% |
| Test and sweep tooling (`tools/`, `tests/`) | 2,653 | 5.9% |
| of which: `op_sweep.py` + `recipes.py` | 1,817 | 4.0% |
| Benchmarks | 145 | 0.3% |
| Packaging (`setup.py`, `pyproject.toml`, `MANIFEST.in`) | 141 | 0.3% |
| Entry points (`plugin_init.cc`) | 84 | 0.2% |
| Docs (`README.md`, `docs/ops.md`) | 686 | 1.5% |
| CI | 63 | 0.1% |
| **Total tracked source** | **~44,900** | |

The brief estimated 20k. It is closer to 45k, of which 36k is kernels.

## What should be deleted

Ordered by how defensible the deletion is. Items over ~500 lines are marked,
per the "ask me before" rule, and nothing has been touched.

### Delete outright, no discussion needed

1. **Six tracked `.DS_Store` files** (`.DS_Store`, `src/.DS_Store`,
   `src/tensorflow/.DS_Store`, `src/tensorflow/core/.DS_Store`,
   `src/tensorflow/core/common_runtime/.DS_Store`,
   `src/tensorflow/core/common_runtime/metal/.DS_Store`), about 8 KB of Finder
   metadata in git. Add `.DS_Store` to `.gitignore`.

2. **`docs/ops.md`, 374 lines.** It documents the in-tree bazel build that no
   longer exists:
   ```
   ./configure    # answer yes to "Metal GPU"
   bazel build --config=metal //tensorflow/tools/pip_package:wheel
   ```
   and opens with "built into TensorFlow rather than installed as a separate
   plugin", which is the opposite of what this repository now is. The README
   links to it for the dtype table. Replace with a generated table, since the
   registry can be dumped mechanically (this audit's appendix is that dump).

3. **The 19 kernels for ops TensorFlow has removed.** Unreachable by
   construction. Concentrated enough to be surgical:
   `metal_batch_norm_global_ops.mm` (439 lines) is *entirely* these two ops and
   can go as a file; the rest are individual registrations inside
   `metal_fft_ops.mm`, `metal_matrix_ops.mm`,
   `metal_quantize_dequantize_ops.mm`, `metal_conv3d_ops.mm`,
   `metal_image2_ops.mm`, `metal_search_ops.mm`, `metal_strided_ops.mm`,
   `metal_linalg_ops.mm`.

4. **The stale header comment on `.github/workflows/ci.yml`**, which says the
   file is "Kept alongside the plugin rather than in .github/workflows". It is
   in `.github/workflows`.

### Propose to delete, needs your decision (all over 500 lines)

Everything here is *working, tested code*. The argument for cutting it is
purely the staffing constraint in the brief: none of it is on the brief's op
list, and each is a subsystem that can break independently under an MPSGraph
update with no user to notice.

| Candidate | Lines | Ops | Why |
| --- | ---: | ---: | --- |
| `metal_cudnn_rnn_ops.mm` | 1466 | 11 | The `CudnnRNN` family. Keras 3 does not route to these ops on a non-CUDA device; it decomposes LSTM/GRU into matmuls and elementwise ops, which are registered. This is a compatibility shim for a code path nothing on macOS takes. Carries the most intricate logic in the tree (parameter buffer layout, `skip_input`, projections, dropout reserve space) and four dedicated tests. |
| `metal_rnn_ops.mm` | 1559 | 8 | `BlockLSTM`/`GRUBlockCell` and gradients. Same argument: a fused path Keras 3 does not emit. |
| `metal_quant_ops.mm` + `metal_quantize_dequantize_ops.mm` | 1647 | 11 | `FakeQuant*` and `QuantizeAndDequantize*`. Quantisation-aware training. Not on the brief's list, and 5 of the 11 are covered by newer op versions or are removed. |
| `metal_sparse_manip_ops.mm` | 836 | 10 | Sparse and ragged tensor manipulation. Not on the brief's list; these are typically CPU-bound shape work where the GPU has little to win. |
| `metal_fft_ops.mm` | 667 | 22 | The Fourier transforms, 6 of the 22 already dead. Real use exists (audio models) but it is a self-contained subsystem with a history of an in-place bug. |
| `metal_linalg_ops.mm` | 661 | 5 | `Qr`, `Lu`, `SelfAdjointEigV2`, `MatrixTriangularSolve`. Numerically the most delicate code here and the least likely to be exercised by the target workloads. |
| `metal_image2_ops.mm` | 624 | 6 | HSV colour-space conversion and contrast/saturation adjustment. Data-augmentation ops, almost always run in the input pipeline on CPU. |
| `metal_collective_ops.mm` | 300 | 7 | NCCL collectives "over one device". A single-GPU backend implementing a multi-GPU communication API is a stub whose only correct behaviour is a copy. Under 500 lines, listed here because cutting it is a scope decision rather than a cleanup. |

Cutting everything in that table removes about 7,760 lines and 80 ops, leaving
roughly 262 ops and 28,300 kernel lines. That is still far more than the
brief's target list. A more aggressive cut to the brief's actual list is
possible but would remove things users would plausibly miss (`GatherV2`,
`OneHot`, `StridedSlice`, the image resizes, `TopKV2`, `CropAndResize`), and I
would want your call on where the line goes rather than guessing.

### Not deletable, for the record

`metal_shader_library.mm` (2485 lines) looks like the biggest single target
and is not one: it is the Metal source string backing the elementwise, fill,
random and comparison kernels. It shrinks only as those shrink.

## What I would do next, if you approve

Phase 1 as written is largely already done. The gap between here and its stated
goal is narrow:

1. Pin a TensorFlow version. 2.20.0 is installed and everything above passes
   against it; 2.21.0 is what the README quotes and is the current release.
   **This needs your decision** (it is on the "ask me before" list), and it is
   the single highest-value change in this document.
2. Add a load-time version check so a TensorFlow upgraded in place fails
   loudly instead of undefined.
3. Extend `op_sweep.py` to emit the per-op max absolute and relative error
   table Phase 1 asks for. The comparison already happens; only the reporting
   is missing.
4. Resolve the deployment-target contradiction: either `-mmacosx-version-min=15.0`
   or `@available` guards on the 16 macOS 15 and macOS 13.2 call sites.
5. Do the uncontroversial deletions in section "Delete outright".
6. Re-measure the benchmarks on the pinned version and write `BENCHMARKS.md`.

Stopping here as instructed.

## Appendix: every registered kernel

342 ops. Dtypes are the `T` or `dtype` attribute constraint as TensorFlow holds
it after registration. "Host-memory args" are inputs or outputs the
registration pins to host memory. The file and line are where the op name
appears in the registration list.

| Op | dtypes | Host-memory args | file:line |
| --- | --- | --- | --- |
| `Abs` | float16, float32 |  | [metal_elementwise_ops.mm:280](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L280) |
| `Acos` | float16, float32 |  | [metal_elementwise_ops.mm:297](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L297) |
| `Acosh` | float16, float32 |  | [metal_elementwise_ops.mm:302](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L302) |
| `Add` | float16, float32 |  | [metal_elementwise_ops.mm:94](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L94) |
| `AddN` | float16, float32 |  | [metal_array_ops.mm:623](src/tensorflow/core/common_runtime/metal/kernels/metal_array_ops.mm#L623) |
| `AddV2` | float16, float32 |  | [metal_elementwise_ops.mm:782](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L782) |
| `AdjustContrast` | float32 |  | [metal_image2_ops.mm:416](src/tensorflow/core/common_runtime/metal/kernels/metal_image2_ops.mm#L416) |
| `AdjustContrastv2` | float32 |  | [metal_image2_ops.mm:619](src/tensorflow/core/common_runtime/metal/kernels/metal_image2_ops.mm#L619) |
| `AdjustHue` | float32 |  | [metal_image2_ops.mm:502](src/tensorflow/core/common_runtime/metal/kernels/metal_image2_ops.mm#L502) |
| `AdjustSaturation` | float32 |  | [metal_image2_ops.mm:502](src/tensorflow/core/common_runtime/metal/kernels/metal_image2_ops.mm#L502) |
| `All` | (no T/dtype constraint) | reduction_indices | [metal_reduction_ops.mm:58](src/tensorflow/core/common_runtime/metal/kernels/metal_reduction_ops.mm#L58) |
| `Any` | (no T/dtype constraint) | reduction_indices | [metal_reduction_ops.mm:57](src/tensorflow/core/common_runtime/metal/kernels/metal_reduction_ops.mm#L57) |
| `ApproxTopK` | float16, float32 |  | [metal_search_ops.mm:447](src/tensorflow/core/common_runtime/metal/kernels/metal_search_ops.mm#L447) |
| `ApproximateEqual` | float16, float32 |  | [metal_compare_ops.mm:87](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L87) |
| `ArgMax` | float16, float32 | dimension | [metal_compare_ops.mm:434](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L434) |
| `ArgMin` | float16, float32 | dimension | [metal_compare_ops.mm:434](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L434) |
| `Asin` | float16, float32 |  | [metal_elementwise_ops.mm:296](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L296) |
| `Asinh` | float16, float32 |  | [metal_elementwise_ops.mm:301](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L301) |
| `Atan` | float16, float32 |  | [metal_elementwise_ops.mm:298](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L298) |
| `Atan2` | float16, float32 |  | [metal_elementwise_ops.mm:105](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L105) |
| `Atanh` | float16, float32 |  | [metal_elementwise_ops.mm:303](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L303) |
| `AvgPool` | float16, float32 |  | [metal_array_ops.mm:520](src/tensorflow/core/common_runtime/metal/kernels/metal_array_ops.mm#L520) |
| `AvgPoolGrad` | float16, float32 | orig_input_shape | [metal_depthwise_ops.mm:435](src/tensorflow/core/common_runtime/metal/kernels/metal_depthwise_ops.mm#L435) |
| `BatchFFT` | (no T/dtype constraint) |  | [metal_fft_ops.mm:647](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L647) |
| `BatchFFT2D` | (no T/dtype constraint) |  | [metal_fft_ops.mm:648](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L648) |
| `BatchFFT3D` | (no T/dtype constraint) |  | [metal_fft_ops.mm:649](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L649) |
| `BatchIFFT` | (no T/dtype constraint) |  | [metal_fft_ops.mm:650](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L650) |
| `BatchIFFT2D` | (no T/dtype constraint) |  | [metal_fft_ops.mm:651](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L651) |
| `BatchIFFT3D` | (no T/dtype constraint) |  | [metal_fft_ops.mm:652](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L652) |
| `BatchMatMul` | float16, float32 |  | [metal_activation_ops.mm:397](src/tensorflow/core/common_runtime/metal/kernels/metal_activation_ops.mm#L397) |
| `BatchMatMulV2` | float16, float32 |  | [metal_activation_ops.mm:543](src/tensorflow/core/common_runtime/metal/kernels/metal_activation_ops.mm#L543) |
| `BatchMatMulV3` | (no T/dtype constraint) |  | [metal_activation_ops.mm:546](src/tensorflow/core/common_runtime/metal/kernels/metal_activation_ops.mm#L546) |
| `BatchMatrixBandPart` | float16, float32 | num_lower, num_upper | [metal_matrix_ops.mm:923](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L923) |
| `BatchMatrixDiag` | float16, float32 |  | [metal_matrix_ops.mm:925](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L925) |
| `BatchMatrixDiagPart` | float16, float32 |  | [metal_matrix_ops.mm:927](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L927) |
| `BatchMatrixSetDiag` | float16, float32 |  | [metal_matrix_ops.mm:929](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L929) |
| `BatchMatrixTriangularSolve` | float32 |  | [metal_linalg_ops.mm:625](src/tensorflow/core/common_runtime/metal/kernels/metal_linalg_ops.mm#L625) |
| `BatchNormWithGlobalNormalization` | float32 |  | [metal_batch_norm_global_ops.mm:432](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_norm_global_ops.mm#L432) |
| `BatchNormWithGlobalNormalizationGrad` | float32 |  | [metal_batch_norm_global_ops.mm:434](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_norm_global_ops.mm#L434) |
| `BatchToSpace` | float16, float32 | crops | [metal_batch_space_ops.mm:448](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_space_ops.mm#L448) |
| `BatchToSpaceND` | float16, float32 | block_shape, crops | [metal_batch_space_ops.mm:327](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_space_ops.mm#L327) |
| `Betainc` | float32 |  | [metal_misc2_ops.mm:436](src/tensorflow/core/common_runtime/metal/kernels/metal_misc2_ops.mm#L436) |
| `BiasAdd` | float16, float32 |  | [metal_fused_ops.mm:111](src/tensorflow/core/common_runtime/metal/kernels/metal_fused_ops.mm#L111) |
| `BiasAddGrad` | float16, float32 |  | [metal_nn_ops.mm:320](src/tensorflow/core/common_runtime/metal/kernels/metal_nn_ops.mm#L320) |
| `BiasAddV1` | float16, float32 |  | [metal_alias_ops.mm:156](src/tensorflow/core/common_runtime/metal/kernels/metal_alias_ops.mm#L156) |
| `Bincount` | float32, int32 | size | [metal_bincount_ops.mm:266](src/tensorflow/core/common_runtime/metal/kernels/metal_bincount_ops.mm#L266) |
| `BlockLSTM` | float32 | seq_len_max | [metal_rnn_ops.mm:1548](src/tensorflow/core/common_runtime/metal/kernels/metal_rnn_ops.mm#L1548) |
| `BlockLSTMGrad` | float32 | seq_len_max | [metal_rnn_ops.mm:1550](src/tensorflow/core/common_runtime/metal/kernels/metal_rnn_ops.mm#L1550) |
| `BlockLSTMGradV2` | float32 | seq_len_max | [metal_rnn_ops.mm:1554](src/tensorflow/core/common_runtime/metal/kernels/metal_rnn_ops.mm#L1554) |
| `BlockLSTMV2` | float32 | seq_len_max | [metal_rnn_ops.mm:1552](src/tensorflow/core/common_runtime/metal/kernels/metal_rnn_ops.mm#L1552) |
| `Bucketize` | float16, float32 |  | [metal_alias_ops.mm:303](src/tensorflow/core/common_runtime/metal/kernels/metal_alias_ops.mm#L303) |
| `CTCLoss` | (no T/dtype constraint) | labels_indices, labels_values, sequence_length | [metal_ctc_ops.mm:329](src/tensorflow/core/common_runtime/metal/kernels/metal_ctc_ops.mm#L329) |
| `CTCLossV2` | (no T/dtype constraint) | labels_indices, labels_values, sequence_length | [metal_ctc_ops.mm:330](src/tensorflow/core/common_runtime/metal/kernels/metal_ctc_ops.mm#L330) |
| `Cast` | (no T/dtype constraint) |  | [metal_elementwise_ops.mm:637](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L637) |
| `Ceil` | float16, float32 |  | [metal_elementwise_ops.mm:283](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L283) |
| `CheckNumerics` | float16, float32 |  | [metal_misc_ops.mm:671](src/tensorflow/core/common_runtime/metal/kernels/metal_misc_ops.mm#L671) |
| `CheckNumericsV2` | float16, float32 |  | [metal_misc_ops.mm:673](src/tensorflow/core/common_runtime/metal/kernels/metal_misc_ops.mm#L673) |
| `ClipByValue` | float16, float32 |  | [metal_index_ops.mm:556](src/tensorflow/core/common_runtime/metal/kernels/metal_index_ops.mm#L556) |
| `Concat` | float16, float32 | concat_dim | [metal_array_ops.mm:631](src/tensorflow/core/common_runtime/metal/kernels/metal_array_ops.mm#L631) |
| `ConcatV2` | float16, float32 | axis | [metal_array_ops.mm:629](src/tensorflow/core/common_runtime/metal/kernels/metal_array_ops.mm#L629) |
| `Conj` | complex64 |  | [metal_alias_ops.mm:380](src/tensorflow/core/common_runtime/metal/kernels/metal_alias_ops.mm#L380) |
| `ConjugateTranspose` | float16, float32 | perm | [metal_alias_ops.mm:247](src/tensorflow/core/common_runtime/metal/kernels/metal_alias_ops.mm#L247) |
| `Conv` | float16, float32 |  | [metal_conv_generic_ops.mm:247](src/tensorflow/core/common_runtime/metal/kernels/metal_conv_generic_ops.mm#L247) |
| `Conv2D` | float16, float32 |  | [metal_conv_ops.mm:181](src/tensorflow/core/common_runtime/metal/kernels/metal_conv_ops.mm#L181) |
| `Conv2DBackpropFilter` | float16, float32 | filter_sizes | [metal_conv_ops.mm:365](src/tensorflow/core/common_runtime/metal/kernels/metal_conv_ops.mm#L365) |
| `Conv2DBackpropInput` | float16, float32 | input_sizes | [metal_conv_ops.mm:271](src/tensorflow/core/common_runtime/metal/kernels/metal_conv_ops.mm#L271) |
| `Conv3D` | float16, float32 |  | [metal_conv3d_ops.mm:242](src/tensorflow/core/common_runtime/metal/kernels/metal_conv3d_ops.mm#L242) |
| `Conv3DBackpropFilter` | float16, float32 | filter_sizes | [metal_conv3d_ops.mm:428](src/tensorflow/core/common_runtime/metal/kernels/metal_conv3d_ops.mm#L428) |
| `Conv3DBackpropFilterV2` | float16, float32 | filter_sizes | [metal_conv3d_ops.mm:423](src/tensorflow/core/common_runtime/metal/kernels/metal_conv3d_ops.mm#L423) |
| `Conv3DBackpropInput` | float16, float32 | input_sizes | [metal_conv3d_ops.mm:426](src/tensorflow/core/common_runtime/metal/kernels/metal_conv3d_ops.mm#L426) |
| `Conv3DBackpropInputV2` | float16, float32 | input_sizes | [metal_conv3d_ops.mm:421](src/tensorflow/core/common_runtime/metal/kernels/metal_conv3d_ops.mm#L421) |
| `Cos` | float16, float32 |  | [metal_elementwise_ops.mm:294](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L294) |
| `Cosh` | float16, float32 |  | [metal_elementwise_ops.mm:300](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L300) |
| `CropAndResize` | float32 | crop_size | [metal_crop_resize_ops.mm:439](src/tensorflow/core/common_runtime/metal/kernels/metal_crop_resize_ops.mm#L439) |
| `CropAndResizeGradBoxes` | float32 |  | [metal_crop_resize_ops.mm:443](src/tensorflow/core/common_runtime/metal/kernels/metal_crop_resize_ops.mm#L443) |
| `CropAndResizeGradImage` | float32 | image_size | [metal_crop_resize_ops.mm:441](src/tensorflow/core/common_runtime/metal/kernels/metal_crop_resize_ops.mm#L441) |
| `Cross` | float16, float32 |  | [metal_alias_ops.mm:444](src/tensorflow/core/common_runtime/metal/kernels/metal_alias_ops.mm#L444) |
| `CudnnRNN` | float16, float32 |  | [metal_cudnn_rnn_ops.mm:1446](src/tensorflow/core/common_runtime/metal/kernels/metal_cudnn_rnn_ops.mm#L1446) |
| `CudnnRNNBackprop` | float16, float32 |  | [metal_cudnn_rnn_ops.mm:1456](src/tensorflow/core/common_runtime/metal/kernels/metal_cudnn_rnn_ops.mm#L1456) |
| `CudnnRNNBackpropV2` | float16, float32 |  | [metal_cudnn_rnn_ops.mm:1458](src/tensorflow/core/common_runtime/metal/kernels/metal_cudnn_rnn_ops.mm#L1458) |
| `CudnnRNNBackpropV3` | float16, float32 |  | [metal_cudnn_rnn_ops.mm:1460](src/tensorflow/core/common_runtime/metal/kernels/metal_cudnn_rnn_ops.mm#L1460) |
| `CudnnRNNCanonicalToParams` | float16, float32 | input_size, num_layers, num_units | [metal_cudnn_rnn_ops.mm:1438](src/tensorflow/core/common_runtime/metal/kernels/metal_cudnn_rnn_ops.mm#L1438) |
| `CudnnRNNCanonicalToParamsV2` | float16, float32 | input_size, num_layers, num_units | [metal_cudnn_rnn_ops.mm:1440](src/tensorflow/core/common_runtime/metal/kernels/metal_cudnn_rnn_ops.mm#L1440) |
| `CudnnRNNParamsSize` | float16, float32 | input_size, num_layers, num_units | [metal_cudnn_rnn_ops.mm:1436](src/tensorflow/core/common_runtime/metal/kernels/metal_cudnn_rnn_ops.mm#L1436) |
| `CudnnRNNParamsToCanonical` | float16, float32 | input_size, num_layers, num_units | [metal_cudnn_rnn_ops.mm:1442](src/tensorflow/core/common_runtime/metal/kernels/metal_cudnn_rnn_ops.mm#L1442) |
| `CudnnRNNParamsToCanonicalV2` | float16, float32 | input_size, num_layers, num_units | [metal_cudnn_rnn_ops.mm:1444](src/tensorflow/core/common_runtime/metal/kernels/metal_cudnn_rnn_ops.mm#L1444) |
| `CudnnRNNV2` | float16, float32 |  | [metal_cudnn_rnn_ops.mm:1448](src/tensorflow/core/common_runtime/metal/kernels/metal_cudnn_rnn_ops.mm#L1448) |
| `CudnnRNNV3` | float16, float32 |  | [metal_cudnn_rnn_ops.mm:1454](src/tensorflow/core/common_runtime/metal/kernels/metal_cudnn_rnn_ops.mm#L1454) |
| `Cumprod` | float16, float32 | axis | [metal_index_ops.mm:483](src/tensorflow/core/common_runtime/metal/kernels/metal_index_ops.mm#L483) |
| `Cumsum` | float16, float32 | axis | [metal_index_ops.mm:483](src/tensorflow/core/common_runtime/metal/kernels/metal_index_ops.mm#L483) |
| `CumulativeLogsumexp` | float16, float32 | axis | [metal_extra_ops.mm:208](src/tensorflow/core/common_runtime/metal/kernels/metal_extra_ops.mm#L208) |
| `DebugNumericSummaryV2` | (no T/dtype constraint) |  | [metal_debug_ops.mm:349](src/tensorflow/core/common_runtime/metal/kernels/metal_debug_ops.mm#L349) |
| `DenseBincount` | float32, int32 | size | [metal_bincount_ops.mm:268](src/tensorflow/core/common_runtime/metal/kernels/metal_bincount_ops.mm#L268) |
| `DepthToSpace` | float16, float32 |  | [metal_matrix_ops.mm:812](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L812) |
| `DepthwiseConv2dNative` | float16, float32 |  | [metal_depthwise_ops.mm:533](src/tensorflow/core/common_runtime/metal/kernels/metal_depthwise_ops.mm#L533) |
| `DepthwiseConv2dNativeBackpropFilter` | float16, float32 | filter_sizes | [metal_depthwise_ops.mm:538](src/tensorflow/core/common_runtime/metal/kernels/metal_depthwise_ops.mm#L538) |
| `DepthwiseConv2dNativeBackpropInput` | float16, float32 | input_sizes | [metal_depthwise_ops.mm:535](src/tensorflow/core/common_runtime/metal/kernels/metal_depthwise_ops.mm#L535) |
| `Diag` | float16, float32 |  | [metal_matrix_ops.mm:542](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L542) |
| `DiagPart` | float16, float32 |  | [metal_matrix_ops.mm:542](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L542) |
| `Dilation2D` | float16, float32 |  | [metal_dilation_ops.mm:195](src/tensorflow/core/common_runtime/metal/kernels/metal_dilation_ops.mm#L195) |
| `Dilation2DBackpropFilter` | float32 |  | [metal_dilation_ops.mm:495](src/tensorflow/core/common_runtime/metal/kernels/metal_dilation_ops.mm#L495) |
| `Dilation2DBackpropInput` | float32 |  | [metal_dilation_ops.mm:494](src/tensorflow/core/common_runtime/metal/kernels/metal_dilation_ops.mm#L494) |
| `Div` | float16, float32 |  | [metal_elementwise_ops.mm:97](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L97) |
| `DynamicPartition` | float32, int32, int64 | partitions | [metal_dynamic_ops.mm:534](src/tensorflow/core/common_runtime/metal/kernels/metal_dynamic_ops.mm#L534) |
| `DynamicStitch` | float32, int32, int64 | indices | [metal_dynamic_ops.mm:537](src/tensorflow/core/common_runtime/metal/kernels/metal_dynamic_ops.mm#L537) |
| `Elu` | float16, float32 |  | [metal_fused_ops.mm:117](src/tensorflow/core/common_runtime/metal/kernels/metal_fused_ops.mm#L117) |
| `EluGrad` | float16, float32 |  | [metal_activation_ops.mm:88](src/tensorflow/core/common_runtime/metal/kernels/metal_activation_ops.mm#L88) |
| `Empty` | float32, int32 | shape | [metal_misc2_ops.mm:446](src/tensorflow/core/common_runtime/metal/kernels/metal_misc2_ops.mm#L446) |
| `Equal` | float16, float32, int32, int64 |  | [metal_compare_ops.mm:79](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L79) |
| `Erf` | float16, float32 |  | [metal_elementwise_ops.mm:287](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L287) |
| `EuclideanNorm` | float16, float32 | reduction_indices | [metal_reduction_ops.mm:59](src/tensorflow/core/common_runtime/metal/kernels/metal_reduction_ops.mm#L59) |
| `Exp` | float16, float32 |  | [metal_elementwise_ops.mm:275](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L275) |
| `Expm1` | float16, float32 |  | [metal_elementwise_ops.mm:292](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L292) |
| `ExtractImagePatches` | float16, float32 |  | [metal_misc_ops.mm:438](src/tensorflow/core/common_runtime/metal/kernels/metal_misc_ops.mm#L438) |
| `ExtractVolumePatches` | float32 |  | [metal_volume_patch_ops.mm:240](src/tensorflow/core/common_runtime/metal/kernels/metal_volume_patch_ops.mm#L240) |
| `FFT` | (no T/dtype constraint) |  | [metal_fft_ops.mm:640](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L640) |
| `FFT2D` | (no T/dtype constraint) |  | [metal_fft_ops.mm:641](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L641) |
| `FFT3D` | (no T/dtype constraint) |  | [metal_fft_ops.mm:642](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L642) |
| `FFTND` | (no T/dtype constraint) | axes, fft_length | [metal_fft_ops.mm:660](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L660) |
| `FakeQuantWithMinMaxArgs` | (no T/dtype constraint) |  | [metal_quant_ops.mm:924](src/tensorflow/core/common_runtime/metal/kernels/metal_quant_ops.mm#L924) |
| `FakeQuantWithMinMaxArgsGradient` | (no T/dtype constraint) |  | [metal_quant_ops.mm:926](src/tensorflow/core/common_runtime/metal/kernels/metal_quant_ops.mm#L926) |
| `FakeQuantWithMinMaxVars` | (no T/dtype constraint) |  | [metal_quant_ops.mm:928](src/tensorflow/core/common_runtime/metal/kernels/metal_quant_ops.mm#L928) |
| `FakeQuantWithMinMaxVarsGradient` | (no T/dtype constraint) |  | [metal_quant_ops.mm:930](src/tensorflow/core/common_runtime/metal/kernels/metal_quant_ops.mm#L930) |
| `FakeQuantWithMinMaxVarsPerChannel` | (no T/dtype constraint) |  | [metal_quant_ops.mm:932](src/tensorflow/core/common_runtime/metal/kernels/metal_quant_ops.mm#L932) |
| `FakeQuantWithMinMaxVarsPerChannelGradient` | (no T/dtype constraint) |  | [metal_quant_ops.mm:934](src/tensorflow/core/common_runtime/metal/kernels/metal_quant_ops.mm#L934) |
| `Fill` | float16, float32 | dims | [metal_fill_ops.mm:260](src/tensorflow/core/common_runtime/metal/kernels/metal_fill_ops.mm#L260) |
| `Floor` | float16, float32 |  | [metal_elementwise_ops.mm:282](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L282) |
| `FloorDiv` | float16, float32 |  | [metal_elementwise_ops.mm:102](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L102) |
| `FloorMod` | float16, float32 |  | [metal_elementwise_ops.mm:103](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L103) |
| `FusedBatchNorm` | float32 |  | [metal_batch_norm_ops.mm:271](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_norm_ops.mm#L271) |
| `FusedBatchNormGrad` | float32 |  | [metal_batch_norm_ops.mm:547](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_norm_ops.mm#L547) |
| `FusedBatchNormGradV2` | float16, float32 |  | [metal_batch_norm_ops.mm:818](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_norm_ops.mm#L818) |
| `FusedBatchNormGradV3` | float16, float32 |  | [metal_batch_norm_ops.mm:826](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_norm_ops.mm#L826) |
| `FusedBatchNormV2` | float16, float32 |  | [metal_batch_norm_ops.mm:814](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_norm_ops.mm#L814) |
| `FusedBatchNormV3` | float16, float32 |  | [metal_batch_norm_ops.mm:816](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_norm_ops.mm#L816) |
| `GRUBlockCell` | float32 |  | [metal_rnn_ops.mm:1208](src/tensorflow/core/common_runtime/metal/kernels/metal_rnn_ops.mm#L1208) |
| `GRUBlockCellGrad` | float32 |  | [metal_rnn_ops.mm:1335](src/tensorflow/core/common_runtime/metal/kernels/metal_rnn_ops.mm#L1335) |
| `GatherNd` | (no T/dtype constraint) |  | [metal_gather_scatter_ops.mm:245](src/tensorflow/core/common_runtime/metal/kernels/metal_gather_scatter_ops.mm#L245) |
| `GatherV2` | (no T/dtype constraint) | axis | [metal_index_ops.mm:198](src/tensorflow/core/common_runtime/metal/kernels/metal_index_ops.mm#L198) |
| `GenerateBoundingBoxProposals` | (no T/dtype constraint) | min_size, nms_threshold, pre_nms_topn | [metal_box_proposal_ops.mm:361](src/tensorflow/core/common_runtime/metal/kernels/metal_box_proposal_ops.mm#L361) |
| `Greater` | float16, float32, int32, int64 |  | [metal_compare_ops.mm:83](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L83) |
| `GreaterEqual` | float16, float32, int32, int64 |  | [metal_compare_ops.mm:84](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L84) |
| `HSVToRGB` | float32 |  | [metal_image2_ops.mm:357](src/tensorflow/core/common_runtime/metal/kernels/metal_image2_ops.mm#L357) |
| `HistogramFixedWidth` | int32, int64 | nbins | [metal_search_ops.mm:251](src/tensorflow/core/common_runtime/metal/kernels/metal_search_ops.mm#L251) |
| `IFFT` | (no T/dtype constraint) |  | [metal_fft_ops.mm:643](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L643) |
| `IFFT2D` | (no T/dtype constraint) |  | [metal_fft_ops.mm:644](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L644) |
| `IFFT3D` | (no T/dtype constraint) |  | [metal_fft_ops.mm:645](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L645) |
| `IFFTND` | (no T/dtype constraint) | axes, fft_length | [metal_fft_ops.mm:661](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L661) |
| `IRFFT` | (no T/dtype constraint) | fft_length | [metal_fft_ops.mm:656](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L656) |
| `IRFFT2D` | (no T/dtype constraint) | fft_length | [metal_fft_ops.mm:657](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L657) |
| `IRFFT3D` | (no T/dtype constraint) | fft_length | [metal_fft_ops.mm:658](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L658) |
| `IRFFTND` | (no T/dtype constraint) | axes, fft_length | [metal_fft_ops.mm:663](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L663) |
| `Identity` | int32 |  | [metal_identity_op.mm:145](src/tensorflow/core/common_runtime/metal/kernels/metal_identity_op.mm#L145) |
| `ImageProjectiveTransformV2` | float32 | output_shape | [metal_transform_ops.mm:272](src/tensorflow/core/common_runtime/metal/kernels/metal_transform_ops.mm#L272) |
| `ImageProjectiveTransformV3` | float32 | fill_value, output_shape | [metal_transform_ops.mm:274](src/tensorflow/core/common_runtime/metal/kernels/metal_transform_ops.mm#L274) |
| `InTopK` | int32, int64 |  | [metal_compare_ops.mm:559](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L559) |
| `InTopKV2` | int32, int64 | k | [metal_compare_ops.mm:742](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L742) |
| `L2Loss` | float16, float32 |  | [metal_matrix_ops.mm:706](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L706) |
| `LRN` | float32 |  | [metal_misc_ops.mm:270](src/tensorflow/core/common_runtime/metal/kernels/metal_misc_ops.mm#L270) |
| `LRNGrad` | float32 |  | [metal_extra_ops.mm:305](src/tensorflow/core/common_runtime/metal/kernels/metal_extra_ops.mm#L305) |
| `LSTMBlockCell` | float32 |  | [metal_rnn_ops.mm:1541](src/tensorflow/core/common_runtime/metal/kernels/metal_rnn_ops.mm#L1541) |
| `LSTMBlockCellGrad` | float32 |  | [metal_rnn_ops.mm:384](src/tensorflow/core/common_runtime/metal/kernels/metal_rnn_ops.mm#L384) |
| `LeakyRelu` | float16, float32 |  | [metal_fused_ops.mm:119](src/tensorflow/core/common_runtime/metal/kernels/metal_fused_ops.mm#L119) |
| `LeakyReluGrad` | float16, float32 |  | [metal_activation_ops.mm:87](src/tensorflow/core/common_runtime/metal/kernels/metal_activation_ops.mm#L87) |
| `Less` | float16, float32, int32, int64 |  | [metal_compare_ops.mm:81](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L81) |
| `LessEqual` | float16, float32, int32, int64 |  | [metal_compare_ops.mm:82](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L82) |
| `LinSpace` | float16, float32 | num | [metal_matrix_ops.mm:626](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L626) |
| `Log` | float16, float32 |  | [metal_elementwise_ops.mm:276](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L276) |
| `Log1p` | float16, float32 |  | [metal_elementwise_ops.mm:291](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L291) |
| `LogSoftmax` | float16, float32 |  | [metal_activation_ops.mm:95](src/tensorflow/core/common_runtime/metal/kernels/metal_activation_ops.mm#L95) |
| `LogicalAnd` | (no T/dtype constraint) |  | [metal_compare_ops.mm:85](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L85) |
| `LogicalNot` | (no T/dtype constraint) |  | [metal_compare_ops.mm:248](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L248) |
| `LogicalOr` | (no T/dtype constraint) |  | [metal_compare_ops.mm:86](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L86) |
| `LowerBound` | float16, float32 |  | [metal_search_ops.mm:143](src/tensorflow/core/common_runtime/metal/kernels/metal_search_ops.mm#L143) |
| `Lu` | float32 |  | [metal_linalg_ops.mm:638](src/tensorflow/core/common_runtime/metal/kernels/metal_linalg_ops.mm#L638) |
| `MatMul` | float16, float32 |  | [metal_matmul_op.mm:322](src/tensorflow/core/common_runtime/metal/kernels/metal_matmul_op.mm#L322) |
| `MatrixBandPart` | float16, float32 | num_lower, num_upper | [metal_matrix_ops.mm:170](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L170) |
| `MatrixDiag` | float16, float32 |  | [metal_matrix_ops.mm:357](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L357) |
| `MatrixDiagPart` | float16, float32 |  | [metal_matrix_ops.mm:260](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L260) |
| `MatrixDiagPartV2` | float16, float32 | k, padding_value | [metal_matrix_ops.mm:948](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L948) |
| `MatrixDiagPartV3` | float16, float32 | k, padding_value | [metal_matrix_ops.mm:950](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L950) |
| `MatrixDiagV2` | float16, float32 | k, num_cols, num_rows, padding_value | [metal_matrix_ops.mm:944](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L944) |
| `MatrixDiagV3` | float16, float32 | k, num_cols, num_rows, padding_value | [metal_matrix_ops.mm:946](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L946) |
| `MatrixSetDiag` | float16, float32 |  | [metal_matrix_ops.mm:433](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L433) |
| `MatrixSetDiagV2` | float16, float32 | k | [metal_matrix_ops.mm:938](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L938) |
| `MatrixSetDiagV3` | float16, float32 | k | [metal_matrix_ops.mm:940](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L940) |
| `MatrixTriangularSolve` | float32 |  | [metal_linalg_ops.mm:623](src/tensorflow/core/common_runtime/metal/kernels/metal_linalg_ops.mm#L623) |
| `Max` | float16, float32 | reduction_indices | [metal_reduction_ops.mm:54](src/tensorflow/core/common_runtime/metal/kernels/metal_reduction_ops.mm#L54) |
| `MaxPool` | float16, float32 |  | [metal_pooling_ops.mm:160](src/tensorflow/core/common_runtime/metal/kernels/metal_pooling_ops.mm#L160) |
| `MaxPoolGrad` | float16, float32 |  | [metal_pooling_ops.mm:228](src/tensorflow/core/common_runtime/metal/kernels/metal_pooling_ops.mm#L228) |
| `MaxPoolGradGrad` | float16, float32 |  | [metal_maxpool_argmax_ops.mm:580](src/tensorflow/core/common_runtime/metal/kernels/metal_maxpool_argmax_ops.mm#L580) |
| `MaxPoolGradGradV2` | float16, float32 | ksize, strides | [metal_maxpool_argmax_ops.mm:583](src/tensorflow/core/common_runtime/metal/kernels/metal_maxpool_argmax_ops.mm#L583) |
| `MaxPoolGradGradWithArgmax` | float16, float32 |  | [metal_maxpool_argmax_ops.mm:578](src/tensorflow/core/common_runtime/metal/kernels/metal_maxpool_argmax_ops.mm#L578) |
| `MaxPoolGradV2` | float16, float32 | ksize, strides | [metal_pool_variant_ops.mm:312](src/tensorflow/core/common_runtime/metal/kernels/metal_pool_variant_ops.mm#L312) |
| `MaxPoolGradWithArgmax` | float16, float32 |  | [metal_maxpool_argmax_ops.mm:576](src/tensorflow/core/common_runtime/metal/kernels/metal_maxpool_argmax_ops.mm#L576) |
| `MaxPoolV2` | float16, float32 | ksize, strides | [metal_pool_variant_ops.mm:249](src/tensorflow/core/common_runtime/metal/kernels/metal_pool_variant_ops.mm#L249) |
| `MaxPoolWithArgmax` | float16, float32 |  | [metal_maxpool_argmax_ops.mm:574](src/tensorflow/core/common_runtime/metal/kernels/metal_maxpool_argmax_ops.mm#L574) |
| `Maximum` | float16, float32 |  | [metal_elementwise_ops.mm:98](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L98) |
| `Mean` | float16, float32 | reduction_indices | [metal_reduction_ops.mm:53](src/tensorflow/core/common_runtime/metal/kernels/metal_reduction_ops.mm#L53) |
| `Min` | float16, float32 | reduction_indices | [metal_reduction_ops.mm:55](src/tensorflow/core/common_runtime/metal/kernels/metal_reduction_ops.mm#L55) |
| `Minimum` | float16, float32 |  | [metal_elementwise_ops.mm:99](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L99) |
| `MirrorPad` | float16, float32 | paddings | [metal_slice_ops.mm:773](src/tensorflow/core/common_runtime/metal/kernels/metal_slice_ops.mm#L773) |
| `MirrorPadGrad` | float16, float32 | paddings | [metal_slice_ops.mm:402](src/tensorflow/core/common_runtime/metal/kernels/metal_slice_ops.mm#L402) |
| `Mod` | float16, float32 |  | [metal_elementwise_ops.mm:104](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L104) |
| `Mul` | float16, float32 |  | [metal_elementwise_ops.mm:96](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L96) |
| `Multinomial` | (no T/dtype constraint) | num_samples | [metal_random_dist_ops.mm:458](src/tensorflow/core/common_runtime/metal/kernels/metal_random_dist_ops.mm#L458) |
| `NcclAllReduce` | float16, float32, float64, int32, int64 |  | [metal_collective_ops.mm:277](src/tensorflow/core/common_runtime/metal/kernels/metal_collective_ops.mm#L277) |
| `NcclBroadcast` | float16, float32, float64, int32, int64 |  | [metal_collective_ops.mm:279](src/tensorflow/core/common_runtime/metal/kernels/metal_collective_ops.mm#L279) |
| `NcclReduce` | float16, float32, float64, int32, int64 |  | [metal_collective_ops.mm:281](src/tensorflow/core/common_runtime/metal/kernels/metal_collective_ops.mm#L281) |
| `Neg` | float16, float32 |  | [metal_elementwise_ops.mm:272](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L272) |
| `NonMaxSuppressionV2` | float32 | iou_threshold, max_output_size | [metal_nms_ops.mm:284](src/tensorflow/core/common_runtime/metal/kernels/metal_nms_ops.mm#L284) |
| `NonMaxSuppressionV3` | float32 | iou_threshold, max_output_size, score_threshold | [metal_nms_ops.mm:286](src/tensorflow/core/common_runtime/metal/kernels/metal_nms_ops.mm#L286) |
| `NonMaxSuppressionV4` | float32 | iou_threshold, max_output_size, score_threshold | [metal_nms_ops.mm:288](src/tensorflow/core/common_runtime/metal/kernels/metal_nms_ops.mm#L288) |
| `NotEqual` | float16, float32, int32, int64 |  | [metal_compare_ops.mm:80](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L80) |
| `OneHot` | float16, float32 | depth | [metal_index_ops.mm:291](src/tensorflow/core/common_runtime/metal/kernels/metal_index_ops.mm#L291) |
| `OnesLike` | float16, float32 |  | [metal_fill_ops.mm:266](src/tensorflow/core/common_runtime/metal/kernels/metal_fill_ops.mm#L266) |
| `Pad` | float16, float32 | paddings | [metal_slice_ops.mm:298](src/tensorflow/core/common_runtime/metal/kernels/metal_slice_ops.mm#L298) |
| `PadV2` | float16, float32 | paddings | [metal_slice_ops.mm:770](src/tensorflow/core/common_runtime/metal/kernels/metal_slice_ops.mm#L770) |
| `ParallelDynamicStitch` | float32, int32, int64 | indices | [metal_dynamic_ops.mm:540](src/tensorflow/core/common_runtime/metal/kernels/metal_dynamic_ops.mm#L540) |
| `ParameterizedTruncatedNormal` | float32 | shape | [metal_random_dist_ops.mm:448](src/tensorflow/core/common_runtime/metal/kernels/metal_random_dist_ops.mm#L448) |
| `PopulationCount` | int32, int64 |  | [metal_extra_ops.mm:140](src/tensorflow/core/common_runtime/metal/kernels/metal_extra_ops.mm#L140) |
| `Pow` | float16, float32 |  | [metal_elementwise_ops.mm:100](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L100) |
| `Prod` | float16, float32 | reduction_indices | [metal_reduction_ops.mm:56](src/tensorflow/core/common_runtime/metal/kernels/metal_reduction_ops.mm#L56) |
| `Qr` | float32 |  | [metal_linalg_ops.mm:627](src/tensorflow/core/common_runtime/metal/kernels/metal_linalg_ops.mm#L627) |
| `QuantizeAndDequantize` | float32 |  | [metal_quantize_dequantize_ops.mm:693](src/tensorflow/core/common_runtime/metal/kernels/metal_quantize_dequantize_ops.mm#L693) |
| `QuantizeAndDequantizeV2` | float32 | input_max, input_min | [metal_quantize_dequantize_ops.mm:695](src/tensorflow/core/common_runtime/metal/kernels/metal_quantize_dequantize_ops.mm#L695) |
| `QuantizeAndDequantizeV3` | float32 | input_max, input_min, num_bits | [metal_quantize_dequantize_ops.mm:697](src/tensorflow/core/common_runtime/metal/kernels/metal_quantize_dequantize_ops.mm#L697) |
| `QuantizeAndDequantizeV4` | float32 | input_max, input_min | [metal_quantize_dequantize_ops.mm:700](src/tensorflow/core/common_runtime/metal/kernels/metal_quantize_dequantize_ops.mm#L700) |
| `QuantizeAndDequantizeV4Grad` | float32 | input_max, input_min | [metal_quantize_dequantize_ops.mm:702](src/tensorflow/core/common_runtime/metal/kernels/metal_quantize_dequantize_ops.mm#L702) |
| `RFFT` | (no T/dtype constraint) | fft_length | [metal_fft_ops.mm:653](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L653) |
| `RFFT2D` | (no T/dtype constraint) | fft_length | [metal_fft_ops.mm:654](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L654) |
| `RFFT3D` | (no T/dtype constraint) | fft_length | [metal_fft_ops.mm:655](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L655) |
| `RFFTND` | (no T/dtype constraint) | axes, fft_length | [metal_fft_ops.mm:662](src/tensorflow/core/common_runtime/metal/kernels/metal_fft_ops.mm#L662) |
| `RGBToHSV` | float32 |  | [metal_image2_ops.mm:220](src/tensorflow/core/common_runtime/metal/kernels/metal_image2_ops.mm#L220) |
| `RaggedBincount` | float32, int32 | size | [metal_misc2_ops.mm:459](src/tensorflow/core/common_runtime/metal/kernels/metal_misc2_ops.mm#L459) |
| `RaggedFillEmptyRows` | float32 |  | [metal_sparse_manip_ops.mm:827](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_manip_ops.mm#L827) |
| `RaggedFillEmptyRowsGrad` | float32 |  | [metal_sparse_manip_ops.mm:831](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_manip_ops.mm#L831) |
| `RandomGamma` | float32 | shape | [metal_random_dist_ops.mm:466](src/tensorflow/core/common_runtime/metal/kernels/metal_random_dist_ops.mm#L466) |
| `RandomStandardNormal` | int32, int64 | shape | [metal_random_ops.mm:355](src/tensorflow/core/common_runtime/metal/kernels/metal_random_ops.mm#L355) |
| `RandomUniform` | int32, int64 | shape | [metal_random_ops.mm:351](src/tensorflow/core/common_runtime/metal/kernels/metal_random_ops.mm#L351) |
| `RandomUniformInt` | int32, int64 | shape | [metal_random_ops.mm:371](src/tensorflow/core/common_runtime/metal/kernels/metal_random_ops.mm#L371) |
| `RealDiv` | float16, float32 |  | [metal_elementwise_ops.mm:784](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L784) |
| `Reciprocal` | float16, float32 |  | [metal_elementwise_ops.mm:281](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L281) |
| `Relu` | float16, float32 |  | [metal_fused_ops.mm:113](src/tensorflow/core/common_runtime/metal/kernels/metal_fused_ops.mm#L113) |
| `Relu6` | float16, float32 |  | [metal_fused_ops.mm:115](src/tensorflow/core/common_runtime/metal/kernels/metal_fused_ops.mm#L115) |
| `Relu6Grad` | float16, float32 |  | [metal_activation_ops.mm:92](src/tensorflow/core/common_runtime/metal/kernels/metal_activation_ops.mm#L92) |
| `ReluGrad` | float16, float32 |  | [metal_nn_ops.mm:144](src/tensorflow/core/common_runtime/metal/kernels/metal_nn_ops.mm#L144) |
| `ResizeBilinear` | float16, float32 | size | [metal_image_ops.mm:137](src/tensorflow/core/common_runtime/metal/kernels/metal_image_ops.mm#L137) |
| `ResizeBilinearGrad` | float32 |  | [metal_resize_grad_ops.mm:251](src/tensorflow/core/common_runtime/metal/kernels/metal_resize_grad_ops.mm#L251) |
| `ResizeNearestNeighbor` | float16, float32 | size | [metal_image_ops.mm:321](src/tensorflow/core/common_runtime/metal/kernels/metal_image_ops.mm#L321) |
| `ResizeNearestNeighborGrad` | float32 | size | [metal_resize_grad_ops.mm:255](src/tensorflow/core/common_runtime/metal/kernels/metal_resize_grad_ops.mm#L255) |
| `Reverse` | float16, float32 | dims | [metal_misc_ops.mm:205](src/tensorflow/core/common_runtime/metal/kernels/metal_misc_ops.mm#L205) |
| `ReverseSequence` | float16, float32 |  | [metal_batch_space_ops.mm:580](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_space_ops.mm#L580) |
| `ReverseV2` | float16, float32 | axis | [metal_slice_ops.mm:529](src/tensorflow/core/common_runtime/metal/kernels/metal_slice_ops.mm#L529) |
| `Rint` | float16, float32 |  | [metal_elementwise_ops.mm:285](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L285) |
| `Roll` | float16, float32 | axis, shift | [metal_strided_ops.mm:511](src/tensorflow/core/common_runtime/metal/kernels/metal_strided_ops.mm#L511) |
| `Round` | float16, float32 |  | [metal_elementwise_ops.mm:284](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L284) |
| `Rsqrt` | float16, float32 |  | [metal_elementwise_ops.mm:274](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L274) |
| `RsqrtGrad` | float16, float32 |  | [metal_elementwise_ops.mm:482](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L482) |
| `Select` | float16, float32, int32, int64 |  | [metal_compare_ops.mm:306](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L306) |
| `SelectV2` | float16, float32, int32, int64 |  | [metal_compare_ops.mm:769](src/tensorflow/core/common_runtime/metal/kernels/metal_compare_ops.mm#L769) |
| `SelfAdjointEigV2` | float32 |  | [metal_linalg_ops.mm:628](src/tensorflow/core/common_runtime/metal/kernels/metal_linalg_ops.mm#L628) |
| `Selu` | float16, float32 |  | [metal_elementwise_ops.mm:290](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L290) |
| `SeluGrad` | float16, float32 |  | [metal_activation_ops.mm:89](src/tensorflow/core/common_runtime/metal/kernels/metal_activation_ops.mm#L89) |
| `Sigmoid` | float16, float32 |  | [metal_fused_ops.mm:121](src/tensorflow/core/common_runtime/metal/kernels/metal_fused_ops.mm#L121) |
| `SigmoidGrad` | float16, float32 |  | [metal_elementwise_ops.mm:480](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L480) |
| `Sign` | float16, float32 |  | [metal_elementwise_ops.mm:286](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L286) |
| `Sin` | float16, float32 |  | [metal_elementwise_ops.mm:293](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L293) |
| `Sinh` | float16, float32 |  | [metal_elementwise_ops.mm:299](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L299) |
| `Slice` | float16, float32 | begin, size | [metal_slice_ops.mm:177](src/tensorflow/core/common_runtime/metal/kernels/metal_slice_ops.mm#L177) |
| `Snapshot` | float16, float32, int32, int64 |  | [metal_misc2_ops.mm:437](src/tensorflow/core/common_runtime/metal/kernels/metal_misc2_ops.mm#L437) |
| `Softmax` | float16, float32 |  | [metal_nn_ops.mm:386](src/tensorflow/core/common_runtime/metal/kernels/metal_nn_ops.mm#L386) |
| `SoftmaxCrossEntropyWithLogits` | float16, float32 |  | [metal_nn_ops.mm:669](src/tensorflow/core/common_runtime/metal/kernels/metal_nn_ops.mm#L669) |
| `Softplus` | float16, float32 |  | [metal_elementwise_ops.mm:288](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L288) |
| `SoftplusGrad` | float16, float32 |  | [metal_activation_ops.mm:90](src/tensorflow/core/common_runtime/metal/kernels/metal_activation_ops.mm#L90) |
| `Softsign` | float16, float32 |  | [metal_activation_ops.mm:93](src/tensorflow/core/common_runtime/metal/kernels/metal_activation_ops.mm#L93) |
| `SoftsignGrad` | float16, float32 |  | [metal_activation_ops.mm:94](src/tensorflow/core/common_runtime/metal/kernels/metal_activation_ops.mm#L94) |
| `SpaceToBatch` | float16, float32 | paddings | [metal_batch_space_ops.mm:448](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_space_ops.mm#L448) |
| `SpaceToBatchND` | float16, float32 | block_shape, paddings | [metal_batch_space_ops.mm:199](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_space_ops.mm#L199) |
| `SpaceToDepth` | float16, float32 |  | [metal_matrix_ops.mm:812](src/tensorflow/core/common_runtime/metal/kernels/metal_matrix_ops.mm#L812) |
| `SparseBincount` | float32, int32 | dense_shape, size | [metal_misc2_ops.mm:456](src/tensorflow/core/common_runtime/metal/kernels/metal_misc2_ops.mm#L456) |
| `SparseConcat` | float32 |  | [metal_sparse_manip_ops.mm:822](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_manip_ops.mm#L822) |
| `SparseFillEmptyRows` | float32 |  | [metal_sparse_manip_ops.mm:823](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_manip_ops.mm#L823) |
| `SparseFillEmptyRowsGrad` | float32 |  | [metal_sparse_manip_ops.mm:825](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_manip_ops.mm#L825) |
| `SparseReorder` | float32 |  | [metal_sparse_manip_ops.mm:816](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_manip_ops.mm#L816) |
| `SparseReshape` | (no T/dtype constraint) |  | [metal_sparse_manip_ops.mm:814](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_manip_ops.mm#L814) |
| `SparseSegmentMean` | float32 |  | [metal_sparse_segment_ops.mm:508](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_segment_ops.mm#L508) |
| `SparseSegmentMeanGrad` | float32 | output_dim0 | [metal_sparse_segment_ops.mm:520](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_segment_ops.mm#L520) |
| `SparseSegmentMeanGradV2` | float32 | dense_output_dim0 | [metal_sparse_segment_ops.mm:526](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_segment_ops.mm#L526) |
| `SparseSegmentMeanWithNumSegments` | float32 | num_segments | [metal_sparse_segment_ops.mm:514](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_segment_ops.mm#L514) |
| `SparseSegmentSqrtN` | float32 |  | [metal_sparse_segment_ops.mm:510](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_segment_ops.mm#L510) |
| `SparseSegmentSqrtNGrad` | float32 | output_dim0 | [metal_sparse_segment_ops.mm:522](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_segment_ops.mm#L522) |
| `SparseSegmentSqrtNGradV2` | float32 | dense_output_dim0 | [metal_sparse_segment_ops.mm:528](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_segment_ops.mm#L528) |
| `SparseSegmentSqrtNWithNumSegments` | float32 | num_segments | [metal_sparse_segment_ops.mm:516](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_segment_ops.mm#L516) |
| `SparseSegmentSum` | float32 |  | [metal_sparse_segment_ops.mm:506](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_segment_ops.mm#L506) |
| `SparseSegmentSumGrad` | float32 | output_dim0 | [metal_sparse_segment_ops.mm:518](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_segment_ops.mm#L518) |
| `SparseSegmentSumGradV2` | float32 | dense_output_dim0 | [metal_sparse_segment_ops.mm:524](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_segment_ops.mm#L524) |
| `SparseSegmentSumWithNumSegments` | float32 | num_segments | [metal_sparse_segment_ops.mm:512](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_segment_ops.mm#L512) |
| `SparseSlice` | float32 |  | [metal_sparse_manip_ops.mm:818](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_manip_ops.mm#L818) |
| `SparseSliceGrad` | float32 |  | [metal_sparse_manip_ops.mm:819](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_manip_ops.mm#L819) |
| `SparseSoftmaxCrossEntropyWithLogits` | float16, float32 |  | [metal_nn_ops.mm:675](src/tensorflow/core/common_runtime/metal/kernels/metal_nn_ops.mm#L675) |
| `SparseSplit` | float32 |  | [metal_sparse_manip_ops.mm:821](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_manip_ops.mm#L821) |
| `SparseTensorDenseMatMul` | float32 | a_shape | [metal_sparse_ops.mm:389](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_ops.mm#L389) |
| `SparseToDense` | float32 | output_shape | [metal_sparse_ops.mm:386](src/tensorflow/core/common_runtime/metal/kernels/metal_sparse_ops.mm#L386) |
| `Split` | float16, float32 | split_dim | [metal_slice_ops.mm:661](src/tensorflow/core/common_runtime/metal/kernels/metal_slice_ops.mm#L661) |
| `SplitV` | float16, float32 | size_splits, split_dim | [metal_slice_ops.mm:782](src/tensorflow/core/common_runtime/metal/kernels/metal_slice_ops.mm#L782) |
| `Sqrt` | float16, float32 |  | [metal_elementwise_ops.mm:273](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L273) |
| `SqrtGrad` | float16, float32 |  | [metal_elementwise_ops.mm:481](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L481) |
| `Square` | float16, float32 |  | [metal_elementwise_ops.mm:277](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L277) |
| `SquaredDifference` | float16, float32 |  | [metal_elementwise_ops.mm:101](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L101) |
| `StatelessMultinomial` | (no T/dtype constraint) | num_samples, seed | [metal_random_dist_ops.mm:461](src/tensorflow/core/common_runtime/metal/kernels/metal_random_dist_ops.mm#L461) |
| `StatelessParameterizedTruncatedNormal` | float32 | seed, shape | [metal_random_dist_ops.mm:450](src/tensorflow/core/common_runtime/metal/kernels/metal_random_dist_ops.mm#L450) |
| `StatelessRandomGammaV2` | float32 | seed, shape | [metal_random_dist_ops.mm:468](src/tensorflow/core/common_runtime/metal/kernels/metal_random_dist_ops.mm#L468) |
| `StatelessRandomGammaV3` | float32 | alg, counter, key, shape | [metal_random_dist_ops.mm:471](src/tensorflow/core/common_runtime/metal/kernels/metal_random_dist_ops.mm#L471) |
| `StridedSlice` | float16, float32 | begin, end, strides | [metal_strided_ops.mm:215](src/tensorflow/core/common_runtime/metal/kernels/metal_strided_ops.mm#L215) |
| `StridedSliceGrad` | float16, float32 | begin, end, shape, strides | [metal_strided_ops.mm:313](src/tensorflow/core/common_runtime/metal/kernels/metal_strided_ops.mm#L313) |
| `Sub` | float16, float32 |  | [metal_elementwise_ops.mm:95](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L95) |
| `Sum` | float16, float32 | reduction_indices | [metal_reduction_ops.mm:52](src/tensorflow/core/common_runtime/metal/kernels/metal_reduction_ops.mm#L52) |
| `Tan` | float16, float32 |  | [metal_elementwise_ops.mm:295](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L295) |
| `Tanh` | float16, float32 |  | [metal_fused_ops.mm:123](src/tensorflow/core/common_runtime/metal/kernels/metal_fused_ops.mm#L123) |
| `TanhGrad` | float16, float32 |  | [metal_elementwise_ops.mm:479](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L479) |
| `Tile` | float16, float32 | multiples | [metal_array_ops.mm:402](src/tensorflow/core/common_runtime/metal/kernels/metal_array_ops.mm#L402) |
| `TileGrad` | float16, float32 | multiples | [metal_strided_ops.mm:424](src/tensorflow/core/common_runtime/metal/kernels/metal_strided_ops.mm#L424) |
| `TopK` | float16, float32 |  | [metal_search_ops.mm:365](src/tensorflow/core/common_runtime/metal/kernels/metal_search_ops.mm#L365) |
| `TopKV2` | float16, float32 | k | [metal_index_ops.mm:407](src/tensorflow/core/common_runtime/metal/kernels/metal_index_ops.mm#L407) |
| `Transpose` | float16, float32 | perm | [metal_array_ops.mm:132](src/tensorflow/core/common_runtime/metal/kernels/metal_array_ops.mm#L132) |
| `TruncatedNormal` | int32, int64 | shape | [metal_random_ops.mm:360](src/tensorflow/core/common_runtime/metal/kernels/metal_random_ops.mm#L360) |
| `Unique` | float32, int32, int64 |  | [metal_dynamic_ops.mm:527](src/tensorflow/core/common_runtime/metal/kernels/metal_dynamic_ops.mm#L527) |
| `UniqueWithCounts` | float32, int32, int64 |  | [metal_dynamic_ops.mm:529](src/tensorflow/core/common_runtime/metal/kernels/metal_dynamic_ops.mm#L529) |
| `UpperBound` | float16, float32 |  | [metal_search_ops.mm:143](src/tensorflow/core/common_runtime/metal/kernels/metal_search_ops.mm#L143) |
| `Xdivy` | float16, float32 |  | [metal_elementwise_ops.mm:106](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L106) |
| `Xlogy` | float16, float32 |  | [metal_elementwise_ops.mm:107](src/tensorflow/core/common_runtime/metal/kernels/metal_elementwise_ops.mm#L107) |
| `ZerosLike` | float16, float32 |  | [metal_fill_ops.mm:264](src/tensorflow/core/common_runtime/metal/kernels/metal_fill_ops.mm#L264) |
| `_FusedBatchNormEx` | float16, float32 |  | [metal_batch_norm_ops.mm:824](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_norm_ops.mm#L824) |
| `_FusedBatchNormGradEx` | float16, float32 |  | [metal_batch_norm_ops.mm:829](src/tensorflow/core/common_runtime/metal/kernels/metal_batch_norm_ops.mm#L829) |
| `_FusedConv2D` | float16, float32 |  | [metal_fused_ops.mm:629](src/tensorflow/core/common_runtime/metal/kernels/metal_fused_ops.mm#L629) |
| `_FusedMatMul` | float16, float32 |  | [metal_fused_ops.mm:631](src/tensorflow/core/common_runtime/metal/kernels/metal_fused_ops.mm#L631) |
| `_NcclBroadcastRecv` | float16, float32, float64, int32, int64 | shape | [metal_collective_ops.mm:288](src/tensorflow/core/common_runtime/metal/kernels/metal_collective_ops.mm#L288) |
| `_NcclBroadcastSend` | float16, float32, float64, int32, int64 |  | [metal_collective_ops.mm:292](src/tensorflow/core/common_runtime/metal/kernels/metal_collective_ops.mm#L292) |
| `_NcclReduceRecv` | float16, float32, float64, int32, int64 |  | [metal_collective_ops.mm:290](src/tensorflow/core/common_runtime/metal/kernels/metal_collective_ops.mm#L290) |
| `_NcclReduceSend` | float16, float32, float64, int32, int64 |  | [metal_collective_ops.mm:294](src/tensorflow/core/common_runtime/metal/kernels/metal_collective_ops.mm#L294) |
| `_TensorToHashBucketFast` | int16, int32, int64, int8 |  | [metal_debug_ops.mm:355](src/tensorflow/core/common_runtime/metal/kernels/metal_debug_ops.mm#L355) |


## Addendum: what Phase 1 changed, and one thing this audit got wrong

Added after Phase 1 ran, so that reading this document does not leave you
acting on findings that have since been fixed or corrected.

**One recommendation above is wrong.** The "Delete outright" section proposes
deleting `docs/ops.md`. That judgement came from its first 45 lines, which
describe a bazel in-tree build that no longer exists. The rest of the file, its
Design and Limitations sections, is accurate and is the only written account of
why the unified-memory decision is load-bearing, what the recurrent parameter
buffer's layout costs, and why the graph pass refuses to fuse across a tensor
with more than one consumer. Phase 1 fixed the front matter and generated the
op table instead. Do not delete it.

**Two problems this audit missed**, both found while fixing what it did find:

- `TF_DISABLE_METAL=1` did not disable the backend, it aborted the process.
  `SE_InitPlugin` reported the refusal as a failed status, and TensorFlow turns
  a failed plugin registration into a CHECK failure. Since the plugin loads
  during `import tensorflow`, the documented way to switch the backend off made
  TensorFlow unimportable. The audit tested that unregistered ops fall back to
  the CPU, and did not test the documented off switch.
- Every dylib built here was stamped `minos 27.0`. `-mmacosx-version-min` was
  in `CXXFLAGS`, which does not reach the link, so `ld` used the build
  machine's macOS. The audit read the deployment target out of the Makefile and
  did not check what ended up in the binary.

Both are fixed. The lesson for the next audit is the same in both cases: check
the artefact, not the setting that was supposed to produce it.

**Fixed in Phase 1:** the version pin (2.20.0, one source of truth in
`TF_SUPPORTED_VERSION`, enforced at load), the 19 dead kernels, the 20
availability warnings and the deployment target, the `.DS_Store` files, the
stale CI comment, the hand-maintained op table (now generated), and the
per-op error reporting the harness was missing.

**Still open**, in the order I would take them:

1. The eight large subsystems in "Propose to delete", which are still awaiting
   a decision and are untouched.
2. The benchmark table. Phase 1 found the measurement method was the problem,
   not the numbers: timing all of the GPU and then all of the CPU turned
   machine drift into apparent speedup, and three consecutive runs put the
   training step at 0.9x, 4.3x and 1.7x. The harness now interleaves the two
   devices and reports the spread of paired ratios. Short cases still move
   between runs on this laptop and the table says which.
3. The distribution name, still one hyphen from Apple's abandoned package.
4. Phase 4's weekly job against the newest TensorFlow, which is what the
   version pin makes meaningful: it is now possible to say "the pin is 2.20 and
   2.22 still loads" rather than floating and never noticing.

## Addendum: what Phases 2 to 4 changed

Added after the scope cut, the rename and the weekly job, for the same reason
as the addendum above: this document is a survey of a tree that no longer
looks like this.

**The eight subsystems in "Propose to delete" are gone**, all of them, at
`8e1d734`. The audit proposed seven of them and hedged on the eighth; the
decision taken was to cut every one except the Fourier transforms, which stay
because `tf.signal` is a real workload on this hardware and the subsystem is
self-contained enough for one person to own. Kernel sources go from 61 files
to 53, kernel code from about 36,000 lines to 27,997, and registered ops from
323 to 268. The sweep reports 268 verified and 0 mismatches against the CPU.

The cut left one thing behind that deleting files did not catch: the QR, the
Jacobi eigensolver and the LAPACK pivot replay are Metal source in the shared
shader library, not in `metal_linalg_ops.mm`, so they survived their only
caller and went on being compiled at every load. Removed at `6092fbe`. A
shader is dispatched by a name assembled at the call site, so searching for
the name finds dozens of false orphans and misses the real one; what
identifies a dead shader is the family it belonged to.

Nothing a default TensorFlow program does breaks. Soft placement is on unless
a program turns it off, so every cut op runs on the host with the same answer,
more slowly. The audit had already verified that fallback works, which is what
made the cut safe to take.

**The distribution is `metal-pluggable-device`**, at `f1efb1b`. The audit
called `tensorflow-metal-plugin` "one hyphenated suffix away from Apple's
abandoned package", which it was. The import path is unchanged, since
TensorFlow scans `site-packages/tensorflow-plugins` and that is still where
the dylib lands. The wheel is 407 KB, down from the 497 KB the audit measured.

That rename was reverted in 0.4.1. It rested on nothing having been published,
which was wrong: `tensorflow-metal-plugin` 0.1.0 to 0.2.0 were already on PyPI,
and a new name would have left those users on a release pinned to nothing.

**CI compares arithmetic now**, at `3c4a6ef`. The audit listed the sweep among
the existing tests and did not say that CI never ran it: the build, the symbol
check, the shader compile and the on-device harness all pass against a kernel
that returns plausible wrong numbers. Thirty seconds a push closes that. The
same step regenerates the two generated tables and fails on a diff, so
"generated" stays true.

**A weekly job builds against the newest TensorFlow**, at `aac463f`. It moves
the pin to whatever is newest for the length of the run and then does what
`ci.yml` does against the pinned release, sweep included, so the tree can say
"the pin is 2.20 and 2.22 still works" instead of finding out from a user.

**One thing the audit missed**, found while cutting: `tools/metal_ops.txt`, a
hand-maintained list of 337 op names, is what the sweep iterates. The audit
described the sweep as thorough and never noticed that its input is a file
someone has to remember to edit. Cutting 55 ops made it disagree with the
registry and the sweep failed on ops that were no longer registered, which is
how it surfaced. `make kernels` now writes it too, from the registry plus the
fourteen ops that need kernel C API entry points no released TensorFlow
exports, so the sweep's input cannot drift from what is registered. The
generated file matched the hand-pruned one exactly, which is the only evidence
available that the hand-maintained version was still correct.

**Still open**, and both need Benjamin rather than a commit:

1. Publishing to PyPI. The name is free of the collision and the sdist and
   wheel both build; nothing has been published under either name.
2. Whether to cut further, to the brief's own op list. That would remove
   `GatherV2`, `OneHot`, `StridedSlice`, the image resizes, `TopKV2` and
   `CropAndResize`, which users would plausibly miss. The audit declined to
   guess where that line goes and so does this addendum.

## Addendum: what changed on 2026-09-16 and 17

The third and last of these. Everything above was written against a tree whose
central limitation has since been lifted by somebody else.

**The missing kernel C API came back.**
[tensorflow/tensorflow#126377](https://github.com/tensorflow/tensorflow/pull/126377)
merged on 2026-09-10, after both 2.20.0 and 2.21.0 had shipped without it. The
audit treated the missing entry points as the standing fact that made the
whole backend synchronous, which it was; it is now a fact about two releases
rather than about TensorFlow. Verified against `tf-nightly 2.22.0-dev20260914`:
all six symbols resolve, the plugin registers the fifteen ops and turns the
synchronous mode off by itself, and nothing in this repository had to change
for that. It is worth 1.34x on a training step and 1.71x on CNN forward at
batch 32, which also stops that case losing to the CPU.

**The benchmark table the audit quoted does not reproduce.** Measured again on
the same machine against the same TensorFlow with no kernel changed in
between, MatMul 2048x2048 came out at 3.55x where the committed table said
6.43x, and the CPU time at 12.01 ms where it said 48.05 ms. Two suites in one
session agree to within a few percent, so neither table is noisy; they
measured different machine conditions. The generated file and the README now
say that the reported range samples one session and that the variance between
sessions is larger.

**Three cases lose to the CPU outright**, which the audit's table did not
show: the 4096x4096 elementwise chain at 0.47x, the 4096x4096 reduction at
0.67x, and CNN forward at batch 32 at 0.80x. A new tool, `benchmarks/crossover.py`,
walks each family from 32x32 to 4096x4096 and finds that the first two never
win at any size, while MatMul crosses at 512x512. That is the cost of eager
dispatch around a memory-bound op rather than a fault in those kernels, and
the conclusion is to keep them: the alternative to a slow GPU op mid-graph is
the same CPU kernel plus two transfers.

**Two things the audit could not have found, both surfaced by measuring:**

- Every boolean op made MPSGraph try the Neural Engine and print nine lines of
  `ANE I/O op can only do F16 MemRef <-> F32 Tensor cast` per compiled graph.
  The logical operators are float arithmetic now and it is silent. While
  checking that, the GPU was confirmed to be doing all of the work: 100%
  active residency at 47.3 W under load, with the Neural Engine rail reporting
  nothing.
- The sweep exempted fourteen ops from a list written by hand, so on the
  TensorFlow that fixed them it would have reported "needs unexported api"
  about a TensorFlow that exports them, and stopped measuring fourteen ops on
  the day they started working. The exemption is a runtime question now, eight
  of the fourteen have recipes and are verified against the CPU, and the six
  that cannot be reached from eager are counted separately.

**Still open**, and all four need Benjamin: publishing to PyPI, cutting further
to the brief's op list, moving the pin to 2.22.0 when it ships, and whether the
three losing cases change anything. They are issues 2, 3, 4 and 10.
