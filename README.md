# tensorflow-metal-plugin

A Metal GPU backend for TensorFlow on Apple silicon, built as an out-of-tree
PluggableDevice. It loads into a stock TensorFlow wheel and adds
`/physical_device:GPU:0`.

Not `tensorflow-metal`, which is Apple's package and has not shipped since
January 2025, three TensorFlow minor releases ago. This one is unaffiliated
with Apple and is named so that the two cannot be confused at a pip prompt.

This is the out-of-tree form of the backend proposed in
[tensorflow/tensorflow#126384](https://github.com/tensorflow/tensorflow/pull/126384).
The sources are the same; the only difference is this repository exports
`SE_InitPlugin`, `TF_InitKernel`, `TF_InitProfiler` and `TF_InitGraph` from a
shared object, where the in-tree form hands the same function pointers to
`RegisterPluggableDevicePlugin`.

Four modules, then: the device and its memory, the kernels, a profiler that
puts Metal work on the TensorFlow timeline, and a graph pass that fuses a bias
and an activation into the convolution or matrix multiply in front of them.

## Status

Working, and every op it registers has been run on a real GPU and checked.
One significant limitation is not this project's to fix: see
[What a released TensorFlow cannot do](#what-a-released-tensorflow-cannot-do).

`make sweep` calls all 282 ops this plugin registers, or would register if a
released TensorFlow exported the entry points they need, through TensorFlow's
own dispatch: once on the GPU and once on the CPU with identical inputs, with
soft placement off so that a missing kernel raises rather than answering from
the host.

| | |
| --- | --- |
| Verified against the CPU kernel, or against a property where there is no CPU kernel | 268 |
| Need kernel C API entry points the pinned TensorFlow does not export, [fixed upstream](#it-is-fixed-upstream-and-not-yet-in-a-release) for 2.22 | 14 |
| **Unaccounted for** | **0** |

How far apart the two answers were, per op, is in
[docs/op_errors.md](docs/op_errors.md). 247 of the 268 carry a row there; the
other 21 have no CPU kernel or no deterministic answer and are checked against
a property instead.

Every op is also run twice and required to give the same answer, which is how
an inverse transform that rewrote its own input was caught. The sweep
separately enumerates every registration TensorFlow holds for these ops and
rejects any that is duplicated or that constrains an attribute the op does not
have, since either makes an op unusable while looking registered.

Nineteen further ops were registered here until they were removed: TensorFlow
deprecates them in their own op defs, so no graph a current TensorFlow builds
can contain one. Eight whole subsystems were removed after that, on purpose
and while working, for the reasons under [Op coverage](#op-coverage). See
[docs/ops.md](docs/ops.md#ops-the-cuda-build-registers-and-this-one-does-not).

Verified on an Apple M4 Max, macOS 27.0, against the stock
`tensorflow==2.21.0` wheel for Python 3.12:

```
before: ['/physical_device:CPU:0']
after : ['/physical_device:CPU:0', '/physical_device:GPU:0']
Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
```

`MatMul`, `Conv2D`, `Softmax`, `Relu`, `MaxPool2D` and `ReduceSum` match the
CPU kernels with soft placement disabled, so a missing GPU kernel raises
instead of quietly producing a correct answer on the wrong device.

## Install

```
pip install "tensorflow==2.21.*"
pip install tensorflow-metal-plugin
```

Requires macOS 15 or later on Apple silicon, and **TensorFlow 2.21**. One
minor release, deliberately: the PluggableDevice C API matches the structs
crossing it by size rather than negotiating a version, so a plugin compiled
against one release and loaded into another goes wrong at a field offset
rather than at a version check. The plugin checks at load and offers no device
if the two disagree, rather than failing in a way that says nothing about the
cause. `TF_SUPPORTED_VERSION` at the repository root is the pin, and setup.py
and CI both read it.

The two commands are in that order for a reason, and the second one fails
without the first. There is no prebuilt wheel: a PluggableDevice is compiled
against the TensorFlow it will be loaded into, and it records that
TensorFlow's location in its own load path, so there is nothing to build
against until TensorFlow is installed. Installing both in a single
`pip install` does not work either, since pip builds this package before it
installs the dependency.

The shared object is built at install time against the
TensorFlow of the interpreter doing the installing, and lands in
`site-packages/tensorflow-plugins`, which TensorFlow scans at import. Nothing
has to be loaded by hand:

```python
>>> import tensorflow as tf
>>> tf.config.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Verified on a clean environment with `tensorflow==2.21.0`, Python 3.12, macOS
27.0 on an M4 Max.

## Training works, and what it costs today

`model.fit(optimizer="adam")` trains and the loss goes down. Getting there
needed a correction worth stating plainly, because it changes the speed.

TensorFlow's own kernels for resource variables reach a tensor through its
data pointer. On a unified memory device that pointer is host-addressable, so
those kernels read and write device memory from the host with no idea that GPU
work is in flight against it. A plugin is supposed to implement those ops
itself, through `tensorflow/c/kernels_experimental.h`, and order them on its
own stream. Since 2.20.0 no shipped binary defines those entry points
([#126374](https://github.com/tensorflow/tensorflow/issues/126374)), so the
ops fall back to the host and race.

The symptom was the worst kind: an optimiser read a slot variable mid-write,
took the square root of whatever was there, and produced `nan` weights with no
error raised. `model.fit` reported `[nan, nan, nan]` and carried on.

While those entry points are missing, every Metal kernel waits for the GPU
before returning, which closes the window. It is announced in a warning at
load, and `TF_METAL_SYNCHRONOUS` forces it either way.

Where the entry points do exist, the plugin implements `AssignVariableOp`,
`AssignAddVariableOp` and `AssignSubVariableOp` itself, on the device, and
nothing touches a variable from the host at all. That is what makes running
asynchronously safe rather than merely faster: without those kernels, an
asynchronous run reproduced the same `nan` on TensorFlow 2.19.1, where the C
API is present. With them, and measured on that 2.19.1, a training step was
11.48 ms against 21.04 ms on the CPU, and correct.

Everything in [BENCHMARKS.md](BENCHMARKS.md) is measured with the wait in
place, since that is what a supported TensorFlow does today.

## Is it faster than the CPU

Sometimes, and by how much depends entirely on the shape of the work.
[BENCHMARKS.md](BENCHMARKS.md) is the full table, regenerated by
`make benchmark`. On an M4 Max against TensorFlow 2.21.0, five runs of the
whole suite, 25 paired repetitions per case per run:

| | GPU | CPU | speedup | across runs |
| --- | ---: | ---: | ---: | ---: |
| MatMul 2048x2048 | 3.85 ms | 12.57 ms | **3.77x** | 3.14..5.94 |
| MatMul 1024x1024 | 1.00 ms | 2.02 ms | **2.77x** | 2.02..2.87 |
| Conv2D, batch 64, 64x64x32 to 64 | 4.67 ms | 9.79 ms | **2.10x** | 2.06..3.31 |
| Conv2D, batch 16 | 1.36 ms | 2.48 ms | **1.83x** | 1.57..3.08 |
| CNN forward, batch 128 | 5.58 ms | 5.87 ms | 1.18x | 1.03..1.45 |
| MatMul 512x512 | 0.35 ms | 0.36 ms | 1.15x | 1.04..1.62 |
| CNN training step, SGD, batch 128 | 18.48 ms | 19.97 ms | 1.12x | 1.06..1.45 |
| CNN forward, batch 32 | 4.33 ms | 3.54 ms | 0.89x | 0.82..0.94 |
| ReduceSum 4096x4096 | 0.52 ms | 0.31 ms | 0.67x | 0.56..0.71 |
| Elementwise 4096x4096 | 3.10 ms | 1.58 ms | 0.53x | 0.44..0.59 |

The range is across whole runs and it is the honest figure, not a defect in
the measurement. Both devices run the identical graph on the identical data in
one process and are timed alternately rather than in two phases, because
timing all of one and then all of the other turns thermal drift into apparent
speedup; that alone moved a training step between 0.9x and 4.3x on three
consecutive runs of an earlier version of this script. What is left after
fixing that is variance between whole runs, which more repetitions inside a
run do not narrow, so the suite is run five times and the spread is reported.

**These numbers replace a table that claimed far more, and the reason to
distrust both is worth more than either.** Two runs of this same script,
on this same machine, against TensorFlow 2.20.0, with nothing changed in
the kernels between them, put MatMul 2048x2048 at 6.43x in one and 3.55x in
the other, and the CPU at 48.05 ms in one and 12.01 ms in the other. A CPU four
times slower is not a measurement of this backend, it is a measurement of what
else the machine was doing. Neither table's "across runs" range hinted at it,
because that range samples one session and the variance that matters sits
between sessions. Re-measure on your own machine rather than quoting either.

The bottom three rows lose to the CPU here, and the pattern is the ordinary
one: the GPU wins where there is arithmetic to do per byte moved and loses
where there is not. A 4096x4096 elementwise chain moves 67 MB and does three
floating point operations per element, so it is bound by memory on a machine
whose CPU shares that same memory.

Every one of these is measured with the kernel C API for resource variables
missing, which is to say with every Metal kernel waiting for the GPU before
returning. That is not free: with `TF_METAL_SYNCHRONOUS=0` forcing the
asynchronous path in the same session, the training step goes from 17.81 ms to
12.32 ms and CNN forward at batch 32 from 3.97 ms to 2.30 ms, which is 1.45x
and 1.73x. Nobody should run that way, because the races it allows are exactly
what the waiting prevents, but it is what the missing entry points cost. See
[What a released TensorFlow cannot do](#what-a-released-tensorflow-cannot-do).

The convolution numbers owe as much to the graph pass as to the kernels. It
folds the bias and the activation into the convolution, and it turns
TensorFlow's layout optimizer off, which was inserting an NHWC to NCHW
transpose on either side of every convolution: on a 4x16x16x8 case those cost
42.8 and 44.2 microseconds around a 43.0 microsecond convolution. MPSGraph
takes either layout, so the rewrite was pure loss.

## Build

Needs the macOS 15 SDK or later and a Python with TensorFlow 2.21 installed.

macOS 15 is also the runtime requirement, and the build says so: the backend
aliases an `MTLBuffer` through `MPSNDArray` with
`initWithBuffer:offset:descriptor:`, which arrived in macOS 15 and which
nothing here falls back from, so the deployment target is 15.0 and dyld will
decline to load the library on anything older rather than meeting an
unrecognised selector partway through a convolution.

The header and library paths come from the installed TensorFlow, so the plugin
is built against exactly the one it will be loaded into. If `make` reports
that TensorFlow was not found, check which interpreter it used: `make`
resolves a bare `python3` through `/bin/sh`, which need not be the one an
interactive shell gives you.

```
make                                  # or: make PYTHON=/path/to/venv/bin/python
make check-symbols
make test                             # on-device checks against the CPU
make test-load                        # how the plugin declines to load
make test-install                     # pip install it, then use it as a user
make sweep                            # every op, against the CPU
make kernels                          # regenerate docs/kernels.md and the sweep's op list
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

Not both in one interpreter. TensorFlow loads everything in
`site-packages/tensorflow-plugins` at import, so once the package is installed
the plugin is already there, and loading the dylib by hand on top of it
registers the `METAL` platform a second time. TensorFlow treats that as a
CHECK failure rather than an error it can return, and the process aborts. The
test scripts here notice an already-registered plugin and test that one
instead; `make kernels` refuses, because the table it writes is the difference
the plugin makes and there is no before to measure.

`TF_DISABLE_METAL=1` keeps the backend out of the process without
uninstalling it: the platform still registers, and offers no device, so
TensorFlow carries on with the CPU.

`TF_METAL_SKIP_VERSION_CHECK=1` loads the plugin into a TensorFlow it was not
built against, which it otherwise declines to do.

Two CI jobs, and they answer different questions. `ci.yml` builds and tests
against the pinned release on every push, because the pin is what a user
installs. `newest-tensorflow.yml` runs weekly against whatever TensorFlow is
newest, with the pin moved to match for the length of the run, and says
whether moving the pin would work: whether it still compiles, still loads, and
still agrees with the CPU op for op. A failure there breaks nothing anyone has
installed, it is the week's notice that the next release needs work first.

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

This is a regression, not a standing limitation. All fourteen symbols of
`tensorflow/c/kernels_experimental.cc` are exported by `libtensorflow_framework`
in 2.19.1 and 2.18.1, and absent from every binary in the 2.20.0 wheel, with
none added in exchange. The headers still declare them. Filed upstream as
[tensorflow/tensorflow#126374](https://github.com/tensorflow/tensorflow/issues/126374).

### It is fixed upstream, and not yet in a release

[tensorflow/tensorflow#126377](https://github.com/tensorflow/tensorflow/pull/126377)
was merged on 2026-09-10. The exports are back. They are not in 2.20.0 or in
2.21.0, both of which shipped before the merge, so everything above is still
what a user gets today; they are in `tf-nightly` and will be in 2.22.0.

Verified here rather than taken on trust. Against `tf-nightly 2.22.0-dev20260914`
on macOS arm64, all six symbols resolve, the plugin builds unchanged, the
on-device harness passes, the sweep reports 268 verified with 0 mismatches,
and the warning about the missing entry points is simply not printed: the
plugin registers the fifteen ops and turns the synchronous mode off by itself,
with no change to this repository.

What that is worth, measured in one session against the same CPU:

| | 2.20.0 | tf-nightly 2.22 | |
| --- | ---: | ---: | --- |
| CNN training step, SGD, batch 128 | 17.87 ms, 1.05x | **13.31 ms, 1.39x** | 1.34x faster |
| CNN forward, batch 128 | 4.80 ms, 1.12x | **3.50 ms, 1.49x** | 1.37x faster |
| CNN forward, batch 32 | 3.99 ms, 0.80x | **2.34 ms, 1.41x** | 1.71x, and it stops losing to the CPU |
| MatMul 2048x2048 | 3.38 ms, 3.55x | 3.90 ms, 3.46x | unchanged |

The shape of that is the point: a single large op cannot hide anything behind
a wait that happens once, and a graph of many small ops pays the wait at every
one. Moving the pin to 2.22.0 when it ships is
[#4](https://github.com/IPNP-BIPN/tensorflow-metal-plugin/issues/4).

It was also the sharpest argument for the in-tree form, where the same code
links these functions directly and all fifteen ops work. That trade is the
subject of the discussion on
[#126254](https://github.com/tensorflow/tensorflow/pull/126254).

## Why this exists

Apple's `tensorflow-metal` last shipped 1.2.0 on 2025-01-31, publishes no
wheel past cp312, has no sdist, and its repository was archived in 2021. TF
master requires Python 3.10 or later and classifies up to cp313, so on a
current Python there is no GPU path for TensorFlow on a Mac at all.

| | tensorflow-metal-plugin | tensorflow-metal |
| --- | --- | --- |
| Latest release | 0.4.1 | 1.2.0, 2025-01-31 |
| TensorFlow releases shipped since | none yet | three: 2.19, 2.20, 2.21 |
| Vendor | none, one maintainer and whoever joins | Apple, which has moved to MLX |
| Source | in this repository, Apache-2.0 | closed, a binary wheel only |
| Distribution | sdist, compiled at install against your TensorFlow | prebuilt wheels, none past cp312 |
| Supported TensorFlow | exactly one release at a time, checked at load | not stated per release |
| Ops on the GPU | 268, listed in [docs/kernels.md](docs/kernels.md) | not documented |
| Unregistered op | falls back to the CPU, verified op by op | same TensorFlow mechanism, not measured here |
| Numerics | every op compared against the CPU each push, [docs/op_errors.md](docs/op_errors.md) | not published |

The comparison is about maintenance, not quality. `tensorflow-metal` was good
and is in many places faster; what it is not any more is maintained, and a
closed binary that nobody updates cannot be fixed by whoever needs it fixed.

## Looking for maintainers

One person cannot own all of this, which is why the scope was cut rather than
completed. Named places where a second pair of hands would change what this
package can promise:

* **The MPSGraph bridge** (`kernels/metal_mps_graph.{h,mm}`, the graph cache
  and the zero-copy aliasing through `MPSNDArray`). This is the hottest code
  in the tree and the least redundant: a mistake here is wrong numbers in
  every op that goes through MPSGraph, not one.
* **The Fourier transforms** (`kernels/metal_fft_ops.mm`). Kept when the other
  seven candidates were cut, because `tf.signal` is a real workload. It is
  self-contained, it has a history of an in-place bug, and it is the one
  subsystem here whose owner could be someone who only cares about audio.
* **CTC loss** (`kernels/metal_ctc_ops.mm`). Log-space forward-backward, one
  thread per sequence, and no CPU comparison for the V2 form because the two
  versions disagree on where the blank class goes.
* **The graph pass** (`metal_graph.{h,mm}`). It fuses a bias and an activation
  and turns the layout optimizer off. What it does not do is interact
  predictably with TensorFlow's own remapper across releases, and nothing
  tells us when that changes except the weekly job.
* **The profiler** (`metal_profiler.{h,mm}`). Reports command buffers rather
  than kernels, so anything the runtime issues on its own shows up as
  `unnamed`. Making that name the op would make the timeline usable.
* **Reference variables and the rewritten ops.** `Assign`, `AssignAdd` and
  `AssignSub` take a reference variable, which eager refuses to call at all,
  and `ParallelConcat` is meant to be replaced by a graph rewrite before its
  kernel is reached. Six ops whose behaviour is stated here rather than
  measured, because the sweep is an eager harness. Reaching them would take a
  v1 graph and a session.
* **Hardware that is not an M4 Max.** Every number in
  [BENCHMARKS.md](BENCHMARKS.md) comes from one laptop. M1, M2, M3, the Ultra
  parts and the Mac Studio thermal envelope are all unmeasured.

Each of those is an open issue: the five subsystems are
[#7](https://github.com/IPNP-BIPN/tensorflow-metal-plugin/issues/7), the
upstream C API is
[#5](https://github.com/IPNP-BIPN/tensorflow-metal-plugin/issues/5), and
hardware that is not this laptop is
[#6](https://github.com/IPNP-BIPN/tensorflow-metal-plugin/issues/6), which
needs one command and a paste.
[#1](https://github.com/IPNP-BIPN/tensorflow-metal-plugin/issues/1) tracks the
rest.

Comment saying which one, or take it and send a pull request. The correctness
sweep is the gate: `make sweep` has to stay at 0 mismatches, and `make
kernels` regenerates the tables CI checks are current.

## Op coverage

268 ops, each one with its dtypes and its registration site in
[docs/kernels.md](docs/kernels.md), which is generated from the registry
rather than maintained by hand.

That is on purpose less than TensorFlow registers for `DEVICE_GPU`. Eight
subsystems were removed because one maintainer cannot answer for them: the
`CudnnRNN` family and the fused `BlockLSTM`/`GRUBlockCell` cells, which Keras 3
does not emit on a non-CUDA device; quantisation-aware training; the sparse and
ragged manipulations; the NCCL collectives, which reduce across devices on a
machine that reports one GPU; the HSV and contrast adjustments, which belong to
an input pipeline; and `Qr`, `Lu`, `SelfAdjointEigV2` and
`MatrixTriangularSolve`, the most delicate numerics here and the least likely
to be on a hot path.

**Nothing an unmodified program does breaks.** With soft placement on, which
is the TF2 eager default, an op with no Metal kernel runs on the host and the
answer is the same. What changes is speed on those ops, and that a program
which has explicitly turned soft placement off now raises where it used to
run on the GPU.

## Layout

```
src/plugin_init.cc                          the exported entry points
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
