# tensorflow-metal-plugin

A Metal GPU backend for TensorFlow on Apple silicon, built as an out-of-tree
PluggableDevice. It loads into a stock TensorFlow wheel and adds
`/physical_device:GPU:0`.

This is the out-of-tree form of the backend proposed in
[tensorflow/tensorflow#126254](https://github.com/tensorflow/tensorflow/pull/126254).
The sources are the same; the only difference is this repository exports
`SE_InitPlugin` and `TF_InitKernel` from a shared object, where the in-tree
form hands the same function pointers to `RegisterPluggableDevicePlugin`.

## Status

Working, and every op it registers has been run on a real GPU and checked.
One significant limitation is not this project's to fix: see
[What a released TensorFlow cannot do](#what-a-released-tensorflow-cannot-do).

`make sweep` calls all 356 registered ops through TensorFlow's own dispatch,
once on the GPU and once on the CPU with identical inputs, with soft placement
off so that a missing kernel raises rather than answering from the host:

| | |
| --- | --- |
| Verified against the CPU kernel, or against a property where there is no CPU kernel | 323 |
| Removed from TensorFlow, so no device can run them | 17 |
| Broken in TensorFlow itself, on every device | 2 |
| Need kernel C API entry points a released TensorFlow does not export | 14 |
| **Unaccounted for** | **0** |

Every op is also run twice and required to give the same answer, which is how
an inverse transform that rewrote its own input was caught. The sweep
separately enumerates every registration TensorFlow holds for these ops and
rejects any that is duplicated or that constrains an attribute the op does not
have, since either makes an op unusable while looking registered.

The two broken in TensorFlow are `TopK` and `TileGrad`, whose own CPU
registrations constrain `index_type` and `Tmultiples`, attributes those ops do
not have. Reproduced on a stock 2.18 and 2.20 with no plugin loaded.

Verified on an Apple M4 Max, macOS 26.6, against the stock
`tensorflow==2.20.0` wheel for Python 3.12:

```
before: ['/physical_device:CPU:0']
after : ['/physical_device:CPU:0', '/physical_device:GPU:0']
Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
```

`MatMul`, `Conv2D`, `Softmax`, `Relu`, `MaxPool2D` and `ReduceSum` match the
CPU kernels with soft placement disabled, so a missing GPU kernel raises
instead of quietly producing a correct answer on the wrong device.

## Build

Needs the macOS 15 SDK or later and a Python with TensorFlow installed. The
backend aliases an `MTLBuffer` through `MPSNDArray` with packed rows, and both
`initWithBuffer:offset:descriptor:` and `preferPackedRows` arrived in that SDK;
an older one does not declare them and the build stops rather than degrading. The
header and library paths come from that TensorFlow, so the plugin is built
against exactly the one it will be loaded into.

```
make                                  # or: make PYTHON=/path/to/venv/bin/python
make check-symbols
make test
```

Then either point TensorFlow at it directly:

```python
from tensorflow.python.framework import load_library
load_library.load_pluggable_device_library("build/libmetal_plugin.dylib")
```

or install it so that `import tensorflow` finds it:

```
make install
```

`TF_DISABLE_METAL=1` keeps the backend out of the process without
uninstalling it.

## What a released TensorFlow cannot do

Six entry points of the kernel C API are declared in the headers a released
TensorFlow ships and are exported by no binary in it:

```
TF_AssignRefVariable
TF_AssignUpdateVariable
TF_GetInputTensorFromVariable
TF_MaybeLockVariableInputMutexesInOrder
TF_ReleaseVariableInputLockHolder
TF_OpKernelConstruction_GetAttrTensorShape
TF_OpKernelContext_ForwardRefInputToRefOutput
```

Checked against `tensorflow==2.20.0` on macOS arm64: absent from
`libtensorflow_framework.2.dylib`, from `libtensorflow_cc.2.dylib`, and from
every pywrap module, and unresolvable by `dlsym` inside a live process.
`TF_AllocateOutput` and `TF_NewKernelBuilder`, from the same header set, are
exported normally, so this is not a matter of the whole C API being private.

Fifteen ops need them, and the plugin does not register those when the symbols
are missing, logging one warning instead:

| Family | Ops |
| --- | --- |
| Optimisers | `ResourceApplyAdam`, `ResourceApplyGradientDescent`, `ResourceApplyMomentum`, `ResourceApplyKerasMomentum`, `ResourceApplyRMSProp` |
| Resource gather and scatter | `ResourceGather`, `ResourceGatherNd`, `ResourceScatterUpdate`, `GatherNd` |
| Reference variables | `Assign`, `AssignAdd`, `AssignSub` |
| Parallel stacking | `ParallelConcat`, `_ParallelConcatStart`, `_ParallelConcatUpdate` |

The optimisers are the whole of that list that matters: **without them there is
no training on the GPU**, only inference and manual gradient work. They run on
the host instead, which is correct and slow.

This is not something an out-of-tree plugin can work around. The functions are
not merely unexported on macOS; nothing in the wheel defines them where a
plugin can reach. Fixing it means TensorFlow exporting them, which is a change
to TensorFlow, not to this repository.

It is also the sharpest argument for the in-tree form, where the same code
links these functions directly and all fifteen ops work. That trade is the
subject of the discussion on
[#126254](https://github.com/tensorflow/tensorflow/pull/126254).

## Why this exists

Apple's `tensorflow-metal` last shipped 1.2.0 on 2025-01-31, publishes no
wheel past cp312, has no sdist, and its repository was archived in 2021. TF
master requires Python 3.10 or later and classifies up to cp313, so on a
current Python there is no GPU path for TensorFlow on a Mac at all.

## Op coverage

The backend registers every op TensorFlow registers for `DEVICE_GPU`, less the
five TensorRT ops that `if_tensorrt` excludes from a macOS build, and less the
fifteen above when the C API entry points they need are missing. The table of
Metal kernels with their dtypes is in
[docs/ops.md](docs/ops.md).

## Layout

```
src/plugin_init.cc                          the two exported entry points
src/tensorflow/core/common_runtime/metal/   the backend, verbatim from the
                                            TensorFlow tree
tools/                                      build probes and the symbol check
tests/                                      on-device checks against CPU
```

The backend sources keep their TensorFlow paths so that syncing them from the
tree is a copy rather than a patch. Two macros, `TF_METAL_OUT_OF_TREE` and
`TF_METAL_NO_STREAM_OPTIONS`, are the whole of what the out-of-tree build
turns on; both are no-ops in the tree.

## Licence

Apache 2.0, the same as TensorFlow.
