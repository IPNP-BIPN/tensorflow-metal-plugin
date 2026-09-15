# Metal PluggableDevice backend

How the backend works. For installing and using it, see the
[README](../README.md); for what it registers, see
[kernels.md](kernels.md).

## Status

Working. 323 ops are registered and every one of them has been run on a real
GPU and compared against the CPU kernel for the same op; `make sweep` is that
comparison and [op_errors.md](op_errors.md) is how far apart the answers were.

Registering an op is not the same as running it well. What has been measured
end to end is a convolutional classifier: convolutions and their gradients,
pooling, activations, softmax cross entropy, reductions, weight initialisation
and the SGD update. The Adam update runs on the host on a released
TensorFlow, for the reason under [Limitations](#limitations).

Ops with no kernel here are not an error. TensorFlow places them on the CPU
through its ordinary soft placement, and a model that uses one runs, slower,
rather than failing.

## Building

```
make
make check-symbols
make test
```

See the [README](../README.md#build) for what that needs and for installing
the result.

## Design

The backend is a StreamExecutor C API plugin in a shared object that
TensorFlow `dlopen`s from `site-packages/tensorflow-plugins`, and
`src/plugin_init.cc` exports the four symbols it looks up by name:
`SE_InitPlugin`, `TF_InitKernel`, and the optional `TF_InitProfiler` and
`TF_InitGraph`. That reuses the whole of
`tensorflow/core/common_runtime/pluggable_device` unchanged, with no patch to
TensorFlow.

### Device type

Devices are registered under device type `GPU` with platform name `METAL`, so
they appear as `/device:GPU:0`. Existing user code, Keras, `tf.distribute` and
the placement rules therefore work without modification. There is no clash with
the CUDA GPU device because CUDA is never built on macOS.

### Unified memory is the load-bearing decision

Every allocation is `MTLResourceStorageModeShared`, and what core receives as
the device address is the buffer's `contents` pointer: a real, host-addressable
address in the same physical memory the GPU reads.

This is what makes the backend work at all. Core does not treat a device
address as opaque. The BFC allocator carves sub-allocations out of a region by
pointer arithmetic, and kernels receive pointers into the middle of buffers. A
plugin returning an `id<MTLBuffer>` in that field breaks the first time core
adds an offset to it.

It also removes the transfer cost that dominates small-model performance on
Mac. Host/device copies are `memcpy`, not staged blits.

`MetalBufferRegistry` goes the other way, recovering the `(buffer, offset)`
pair a Metal encoder needs from an arbitrary interior pointer.

The backend accepts only devices reporting `hasUnifiedMemory`, and that is a
correctness gate. On a discrete GPU, shared-storage buffers are host-side
staging that needs explicit `didModifyRange:` and `synchronizeResource:`, so
the zero-copy transfers would read stale data. Devices without it are skipped
and logged by name.

### Stream ordering

StreamExecutor's stream is strictly ordered. An `MTLCommandQueue` only
guarantees that command buffers are *scheduled* in commit order; their
execution may overlap. `SP_Stream_st` restores the contract with a per-stream
`MTLSharedEvent` used as a sequence counter: command buffer N waits for N-1 and
signals N. That serialises a stream without serialising the device.

`OrderedCommandBuffer` is the only way to obtain a command buffer, so nothing
can bypass the ordering. Its `CommitWithHostCompletion` variant signals the
sequence from the host after a completion block runs, which is what lets the
`memcpy`-based transfers take part in stream order without racing the next
command buffer.

### Kernels

Registered through the Kernel C API. Elementwise arithmetic and `Cast` are
Metal compute shaders compiled at runtime from an embedded source string;
`MatMul` uses `MPSMatrixMultiplication`.

Convolutions, pooling, activations, softmax, the cross entropies, batch
normalisation and the reductions go through `MPSGraph`; `MatMul` uses `MPSMatrix` directly because a
2-D multiply needs less machinery.

The `MPSGraph` path is zero-copy in both directions, which is not obvious and
is worth stating: `MPSGraphTensorData`'s `MTLBuffer` initialiser assumes a
tensor starts at the beginning of its buffer, which BFC sub-allocation
guarantees it does not. `MPSNDArray`'s `initWithBuffer:offset:descriptor:`
aliases a buffer at a byte offset, and `MPSGraphTensorData` accepts an
`MPSNDArray`, so operands are fed and results written in place with no staging
buffer anywhere.

Arithmetic goes through `MPSGraph` too, which is what gives it full NumPy
broadcasting rather than the scalar-only broadcasting a hand-written shader
would have to implement by hand.

Random number generation and the optimiser updates are compute shaders rather
than `MPSGraph`. Graphs here are cached by shape, so a seed baked into a graph
would make every call after the first return the same tensor; and MPSGraph
parameterises Adam differently from TensorFlow, which would train subtly
differently rather than fail.

## Supported ops

The table is generated from the kernel registry rather than written by hand,
because a hand-written one is a claim about the kernels rather than a
description of them, and this one had drifted: it still named ops that had
been deleted. See [kernels.md](kernels.md), refreshed with `make kernels`.

Resource variables (`VarHandleOp`, `ReadVariableOp`, `AssignVariableOp` and
the rest), `Reshape`, `Const`, `Shape`, `StridedSlice`, `Pack`, `Unpack`,
`ExpandDims` and `Squeeze` need no kernel here: TensorFlow registers them for
`DEVICE_DEFAULT`, which any device type inherits when it has no kernel of its
own. Registering them again here would add nothing.

## Limitations

* **Registered is not the same as accelerated.** `TensorArray` and the CSR
  sparse matrix ops are registered with their tensors pinned to host memory,
  because their kernels run the host's arithmetic over a resource the kernel C
  API cannot reach. On a unified memory device the pinning costs a memcpy
  rather than a transfer, but the arithmetic itself is on the CPU. This is the
  same missing C API as
  [#126374](https://github.com/tensorflow/tensorflow/issues/126374) and is not
  fixable from inside a plugin; the proposed fix is
  [#126377](https://github.com/tensorflow/tensorflow/pull/126377).
* **The recurrent parameter buffer's layout is this backend's own.** That is
  allowed because the buffer is opaque and the canonical conversions are the
  only defined way in and out of it, but a checkpoint holding a buffer written
  by cuDNN will not load; one holding canonical weights will. Dropout's masks
  are likewise this backend's own sequence, since nothing outside cuDNN
  defines that one either: what is guaranteed is the rate, the inverted
  scaling, the placement between layers, and reproducibility from the seed.
  Everything else cuDNN accepts is implemented, including `skip_input`, a
  recurrent projection, all four cell types, both directions, any number of
  layers and per-sequence lengths.
* **`ParallelConcat` is registered but always fails**, which is what every
  device does, CUDA included: the graph rewrite replaces the op with an
  allocation and one update per stacked value, so reaching the kernel means
  the rewrite did not run. Both ops it is replaced by are implemented. This is
  correct behaviour rather than a gap.
* **The graph pass fuses a bias and an activation, and nothing else.** A
  folded batch normalisation would need more inputs than the fused kernels
  read, and fusing across a tensor with more than one consumer would leave the
  other consumers pointing at a node that no longer exists, so both are
  refused. The pass also turns TensorFlow's layout optimizer off, because
  MPSGraph takes NHWC and NCHW alike and its transposes are pure loss here.

  Most of the fusing is TensorFlow's own remapper, which does run for a
  pluggable device named GPU. What it leaves behind, and this pass picks up,
  is a convolution followed by a bias with no activation, and anything
  starting from a `MatMul`: the remapper gates those on
  `BlasLtMatmulEnabled()`, which reads `TF_USE_CUBLASLT`, a switch named after
  a CUDA library and one a Metal device can never satisfy.
* **The profiler reports command buffers, not kernels.** One event per
  submission, named after the node that submitted it. A command buffer that
  the runtime issues on its own, a copy or a fill, has no node to name and
  appears as `unnamed`.
* **`MatMul` takes rank-2 tensors only**, which is the op definition rather
  than a restriction: anything higher is `BatchMatMulV2` or `V3`, which this
  backend implements separately.
* **The max pooling ops that carry indices are NHWC only**, which is again the
  op definition: none of them has a `data_format` attribute.
* **Single device.** Every Apple silicon Mac reports one GPU; multi-device has
  had no testing.

## Ops the CUDA build registers and this one does not

The five TensorRT ops, `TRTEngineOp` and the four that manage its resource.
They are gated behind `if_tensorrt`, TensorRT does not build on macOS, and the
ops therefore do not exist to be registered for.

Nineteen more were registered here until they were removed: TensorFlow
deprecated them in their own op defs, so no graph a current TensorFlow builds
can contain one and no device can run them. They were the six `BatchFFT`
spellings, the four `BatchMatrix` ones, `BatchMatrixTriangularSolve`,
`QuantizeAndDequantize`, `AdjustContrast`, the two
`BatchNormWithGlobalNormalization` ops, the two v1 `Conv3D` gradients, `TopK`
and `TileGrad`.

Everything else CUDA registers for a GPU is registered here, by one of four
routes, and it is worth knowing which because they are not equally fast:

* **A Metal kernel**, for most of them. These are the ops in the table above.
* **`DEVICE_DEFAULT`**, TensorFlow's own registration for a device with no
  kernel of its own. `Reshape`, `Const`, `Shape`, the resource variable ops
  and the input pipeline all arrive this way, with no code here.
* **An unguarded `DEVICE_GPU` registration** in TensorFlow's own kernels. The
  Metal platform's device type is `GPU`, so a registration that is not inside
  a `GOOGLE_CUDA` guard already applies to it.
* **`PLUGGABLE_DEVICE_SUPPORTED_MACOS`**, a macro TensorFlow's kernels already
  carry for exactly this situation and that nothing had ever defined. A Metal
  build defines it, which turns on the host-memory registrations for
  `TensorArray` and the CSR sparse matrix ops. That is the pinned-memory case
  under [Limitations](#limitations).

Two groups are registered but do less than the CUDA kernel of the same name,
and say so rather than pretending otherwise:

* **The NCCL collectives** reduce across devices, and every Apple silicon Mac
  reports one GPU. `NcclAllReduce`, `NcclBroadcast` and `NcclReduce` copy their
  input to their output, which is what reducing over one device means, and
  fail at construction when asked for more than one.
* **`ParallelConcat`** fails with TensorFlow's own message, as it does on
  every device including CUDA.

## Files

| File | Role |
| --- | --- |
| `metal_buffer_registry.{h,mm}` | Device address to `MTLBuffer` mapping, allocator stats |
| `metal_stream.{h,mm}` | Streams, events, timers, `OrderedCommandBuffer`, per-device state |
| `metal_stream_executor.{h,mm}` | The `SP_StreamExecutor` callback table |
| `metal_platform.{h,mm}` | `SP_Platform`, device discovery, plugin entry point |
| `metal_profiler.{h,mm}` | The pluggable profiler: op labels, GPU timings, XSpace |
| `metal_graph.{h,mm}` | The graph optimizer: bias and activation fusion |
| `../plugin_init.cc` | The four exported entry points |
| `kernels/metal_mps_graph.{h,mm}` | MPSGraph bridge, graph cache, zero-copy tensor aliasing |
| `kernels/metal_shader_library.{h,mm}` | Embedded Metal source and pipeline cache |
| `kernels/metal_*_ops.mm` | Op kernels, grouped by family |

Everything under `src/tensorflow/core/common_runtime/metal/`.
