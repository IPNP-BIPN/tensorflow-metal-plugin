"""Runs every op the backend registers and compares it to the CPU kernel.

The point is the difference between registered and working. A kernel that
returns zeros, or that TensorFlow refuses to dispatch, is registered and
broken, and only going through TensorFlow's own dispatch with real inputs
shows it.

Each op is called twice with identical inputs, once pinned to the GPU and once
to the CPU, and the results compared. Soft placement is off, so an op with no
GPU kernel raises rather than quietly answering from the host. When the CPU
call fails too, the inputs were wrong rather than the kernel, and the op is
reported as unexercised rather than as a failure: a sweep that counts its own
bad recipes as bugs is worse than no sweep.
"""

import argparse
import os
import sys
import traceback

import collections

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import kernels
from tensorflow.python.framework import load_library
from tensorflow.python.framework import op_def_registry

import recipes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MATCH, MISMATCH, GPU_ERROR, UNEXERCISED, NO_RECIPE = (
    "match", "mismatch", "gpu-error", "unexercised", "no-recipe")


def load_ops(path):
  with open(path) as handle:
    return [line.strip() for line in handle if line.strip()]


# Shapes small enough to be quick and awkward enough to catch stride bugs: a
# non-square matrix, an odd innermost dimension, a batch that is not one.
RNG = np.random.default_rng(7)


def f32(*shape):
  return RNG.standard_normal(shape, dtype=np.float32)


def positive(*shape):
  return np.abs(f32(*shape)) + 0.5


IMAGE = f32(2, 8, 9, 3)
MAT = f32(6, 5)
VEC = f32(7)


# Inputs are chosen so that an op's mathematical domain is respected: values
# in (0.05, 0.95) are valid for the logarithms, the roots, the inverse
# trigonometric functions and the reciprocals alike, so one array serves the
# whole unary family without a per-op table of domains.
SAFE = (RNG.random((6, 5)).astype(np.float32) * 0.9 + 0.05)
SAFE2 = (RNG.random((6, 5)).astype(np.float32) * 0.9 + 0.05)
SIGNED = RNG.standard_normal((6, 5), dtype=np.float32)
BOOL = RNG.random((6, 5)) > 0.5
INT = RNG.integers(0, 5, (6, 5)).astype(np.int32)


def dtype_for(op_def, attr_name="T", prefer=(tf.float32, tf.int32, tf.bool)):
  for attr in op_def.attr:
    if attr.name != attr_name or attr.type != "type":
      continue
    allowed = list(attr.allowed_values.list.type)
    if not allowed:
      return tf.float32
    for candidate in prefer:
      if candidate.as_datatype_enum in allowed:
        return candidate
    return tf.as_dtype(allowed[0])
  return None


def tensor(dtype, positive=True):
  if dtype is None or dtype == tf.float32:
    return tf.constant(SAFE if positive else SIGNED)
  if dtype.is_bool:
    return tf.constant(BOOL)
  if dtype.is_integer:
    return tf.constant(INT.astype(dtype.as_numpy_dtype))
  if dtype.is_floating:
    return tf.constant((SAFE if positive else SIGNED).astype(
        dtype.as_numpy_dtype))
  return None


# One recipe per input signature, which is what an op def actually varies. The
# long tail of structured ops is handled by name below.
BY_SIGNATURE = {
    ("x",): lambda d: {"x": tensor(dtype_for(d))},
    ("input",): lambda d: {"input": tensor(dtype_for(d))},
    ("features",): lambda d: {"features": tensor(dtype_for(d), positive=False)},
    ("x", "y"): lambda d: {"x": tensor(dtype_for(d)),
                           "y": tensor(dtype_for(d))},
    ("input", "reduction_indices"): lambda d: {
        "input": tensor(dtype_for(d)), "reduction_indices": [1]},
    ("gradients", "features"): lambda d: {
        "gradients": tf.constant(SIGNED),
        "features": tf.constant(SIGNED)},
    ("y", "dy"): lambda d: {"y": tf.constant(SAFE), "dy": tf.constant(SAFE2)},
    ("value", "bias"): lambda d: {"value": tf.constant(SAFE),
                                  "bias": tf.constant(SAFE[0])},
    ("x", "axis"): lambda d: {"x": tensor(dtype_for(d)), "axis": 1},
    ("input", "dimension"): lambda d: {"input": tf.constant(SIGNED),
                                       "dimension": 1},
    ("diagonal",): lambda d: {"diagonal": tf.constant(SAFE[0])},
    ("input", "num_lower", "num_upper"): lambda d: {
        "input": tf.constant(SAFE), "num_lower": 1, "num_upper": 1},
    ("input", "diagonal"): lambda d: {"input": tf.constant(SAFE[:5, :5]),
                                      "diagonal": tf.constant(SAFE[0][:5])},
    ("matrix", "rhs"): lambda d: {
        "matrix": tf.constant(np.tril(SAFE[:5, :5]) + 2.0 * np.eye(5,
                                                                   dtype=np.float32)),
        "rhs": tf.constant(SAFE[:5, :3])},
    ("t", "clip_value_min", "clip_value_max"): lambda d: {
        "t": tf.constant(SIGNED), "clip_value_min": -0.5,
        "clip_value_max": 0.5},
    ("condition", "t", "e"): lambda d: {
        "condition": tf.constant(BOOL), "t": tf.constant(SAFE),
        "e": tf.constant(SAFE2)},
    ("data", "segment_ids"): lambda d: {
        "data": tf.constant(SAFE), "segment_ids": tf.constant([0, 0, 1, 1, 2, 2])},
    ("data", "segment_ids", "num_segments"): lambda d: {
        "data": tf.constant(SAFE),
        "segment_ids": tf.constant([0, 0, 1, 1, 2, 2]), "num_segments": 3},
    ("params", "indices"): lambda d: {
        "params": tf.constant(SAFE), "indices": tf.constant([0, 2, 1])},
    ("params", "indices", "axis"): lambda d: {
        "params": tf.constant(SAFE), "indices": tf.constant([0, 2, 1]),
        "axis": 0},
    ("tensor", "shape"): lambda d: {"tensor": tf.constant(SAFE),
                                    "shape": [5, 6]},
    ("input", "shape"): lambda d: {"input": tf.constant(SAFE), "shape": [5, 6]},
    ("input", "perm"): lambda d: {"input": tf.constant(SAFE), "perm": [1, 0]},
    ("x", "perm"): lambda d: {"x": tf.constant(SAFE), "perm": [1, 0]},
    ("input", "multiples"): lambda d: {"input": tf.constant(SAFE),
                                       "multiples": [2, 1]},
    ("input", "paddings"): lambda d: {"input": tf.constant(SAFE),
                                      "paddings": [[1, 1], [2, 0]]},
    ("input", "paddings", "constant_values"): lambda d: {
        "input": tf.constant(SAFE), "paddings": [[1, 1], [2, 0]],
        "constant_values": 0.0},
    ("input", "begin", "size"): lambda d: {
        "input": tf.constant(SAFE), "begin": [1, 1], "size": [3, 2]},
    ("input", "axis"): lambda d: {"input": tf.constant(SAFE), "axis": [1]},
    ("input", "dims"): lambda d: {"input": tf.constant(SAFE),
                                  "dims": [False, True]},
    ("tensor", "mask"): lambda d: {"tensor": tf.constant(SAFE),
                                   "mask": tf.constant(BOOL[0])},
}

# Ops whose inputs are structural enough that only a hand-written call will do.
IMAGE = RNG.standard_normal((2, 8, 9, 3), dtype=np.float32)
FILTER = RNG.standard_normal((3, 3, 3, 4), dtype=np.float32)


def by_name():
  conv = {"input": tf.constant(IMAGE), "filter": tf.constant(FILTER),
          "strides": [1, 1, 1, 1], "padding": "SAME"}
  pool = {"input": tf.constant(IMAGE), "ksize": [1, 2, 2, 1],
          "strides": [1, 2, 2, 1], "padding": "VALID"}
  recipes = {
      "Conv2D": conv,
      "DepthwiseConv2dNative": dict(conv, filter=tf.constant(FILTER)),
      "MaxPool": pool,
      "AvgPool": pool,
      "Relu": {"features": tf.constant(IMAGE)},
      "Softmax": {"logits": tf.constant(SAFE)},
      "LogSoftmax": {"logits": tf.constant(SAFE)},
      "BiasAdd": {"value": tf.constant(IMAGE),
                  "bias": tf.constant(IMAGE[0, 0, 0])},
      "MatMul": {"a": tf.constant(SAFE), "b": tf.constant(SAFE.T)},
      "BatchMatMulV2": {"x": tf.constant(IMAGE), "y": tf.constant(
          np.transpose(IMAGE, (0, 1, 3, 2)).copy())},
      "Fill": {"dims": [3, 4], "value": 2.0},
      "OnesLike": {"x": tf.constant(SAFE)},
      "ZerosLike": {"x": tf.constant(SAFE)},
      "Cast": {"x": tf.constant(SAFE), "DstT": tf.float16},
      "Concat": {"concat_dim": 0, "values": [tf.constant(SAFE),
                                             tf.constant(SAFE2)]},
      "ConcatV2": {"values": [tf.constant(SAFE), tf.constant(SAFE2)],
                   "axis": 0},
      "Pack": {"values": [tf.constant(SAFE), tf.constant(SAFE2)], "axis": 0},
      "Split": {"split_dim": 0, "value": tf.constant(SAFE), "num_split": 2},
      "AddN": {"inputs": [tf.constant(SAFE), tf.constant(SAFE2)]},
      "Transpose": {"x": tf.constant(SAFE), "perm": [1, 0]},
      "ConjugateTranspose": {"x": tf.constant(SAFE), "perm": [1, 0]},
      "Reshape": {"tensor": tf.constant(SAFE), "shape": [5, 6]},
      "ExpandDims": {"input": tf.constant(SAFE), "dim": 0},
      "Squeeze": {"input": tf.constant(SAFE[None])},
      "Tile": {"input": tf.constant(SAFE), "multiples": [2, 1]},
      "TopKV2": {"input": tf.constant(SAFE), "k": 3},
      "LRN": {"input": tf.constant(IMAGE)},
      "L2Loss": {"t": tf.constant(SAFE)},
      "Where": {"input": tf.constant(BOOL)},
      "Unique": {"x": tf.constant([1, 2, 2, 3, 1], dtype=tf.int32)},
      "InvertPermutation": {"x": tf.constant([2, 0, 1], dtype=tf.int32)},
  }
  return recipes


# Built after the plugin is loaded, never at import: creating a tensor
# initialises the eager context, and a device registered after that is not
# picked up. The first version of this sweep built its inputs at import and
# reported all 111 exercised ops as having no GPU kernel, which was the sweep
# describing its own mistake.
NAMED = {}
RANDOM = {}


def synthesize(op_def):
  """A call for an op, or None when no recipe covers it."""
  if op_def.name in RANDOM:
    return dict(RANDOM[op_def.name])
  if op_def.name in NAMED:
    return dict(NAMED[op_def.name])
  signature = tuple(a.name for a in op_def.input_arg)
  builder = BY_SIGNATURE.get(signature)
  if builder is None:
    return None
  for attr in op_def.attr:
    if attr.type == "type" or attr.HasField("default_value"):
      continue
    return None  # a required attr this recipe says nothing about
  for arg in op_def.input_arg:
    if arg.is_ref or arg.type_list_attr or arg.number_attr:
      return None
    dtype = tf.as_dtype(arg.type) if arg.type else None
    if dtype in (tf.resource, tf.variant, tf.string):
      return None
  return builder(op_def)


def duplicate_registrations(ops):
  """Ops registered twice for the GPU with the same constraints.

  TensorFlow will not dispatch an op whose registrations tie, so a duplicate
  is not a harmless extra: the op cannot run at all. This is cheap to check
  and invisible from the outside until something tries to use the op, which
  is how twelve of them, Identity among them, stayed broken.
  """
  found = {}
  for name in ops:
    try:
      registered = kernels.get_registered_kernels_for_op(name).kernel
    except Exception:  # pylint: disable=broad-except
      continue
    seen = collections.Counter()
    for kernel in registered:
      if kernel.device_type != "GPU":
        continue
      seen[(tuple(sorted((c.name, tuple(c.allowed_values.list.type))
                         for c in kernel.constraint)),
            tuple(sorted(kernel.host_memory_arg)))] += 1
    extra = sum(count - 1 for count in seen.values() if count > 1)
    if extra:
      found[name] = extra
  return found


def run(op_name, kwargs, device):
  with tf.device(device):
    return getattr(tf.raw_ops, op_name)(**kwargs)


def flatten(result):
  if isinstance(result, (list, tuple)):
    return [np.asarray(r) for r in result]
  if hasattr(result, "numpy"):
    return [result.numpy()]
  return [np.asarray(result)]


# Outputs the op def calls scratch: their contents are the kernel's own
# business, are documented as opaque, and differ between implementations on
# purpose. TensorFlow's CPU kernel leaves the batch-norm reserve spaces zero
# during inference; this backend writes the statistics into them. What has to
# agree is the gradient that reads them, which is exercised on its own.
OPAQUE_OUTPUTS = {
    "FusedBatchNorm": (3, 4),
    "FusedBatchNormV2": (3, 4),
    "FusedBatchNormV3": (3, 4),
    "_FusedBatchNormEx": (3, 4),
}


def compare(cpu, gpu, op_name=""):
  if len(cpu) != len(gpu):
    return False, "different output counts"
  skip = OPAQUE_OUTPUTS.get(op_name, ())
  worst = 0.0
  for index, (a, b) in enumerate(zip(cpu, gpu)):
    if index in skip:
      continue
    if a.shape != b.shape:
      return False, f"shape {b.shape} vs {a.shape}"
    if a.dtype.kind in "fc":
      if not np.all(np.isfinite(a)):
        continue
      worst = max(worst, float(np.max(np.abs(a - b))) if a.size else 0.0)
      if not np.allclose(a, b, rtol=1e-3, atol=1e-3, equal_nan=True):
        # Which output, because an op with six of them says nothing useful
        # otherwise.
        return False, f"output {index} differs by {worst:.3e}"
    elif not np.array_equal(a, b):
      return False, f"output {index} differs in value"
  return True, f"max diff {worst:.2e}"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ops", default=os.path.join(HERE, "metal_ops.txt"))
  parser.add_argument("--plugin",
                      default=os.path.join(ROOT, "build", "libmetal_plugin.dylib"))
  parser.add_argument("--only", default=None)
  args = parser.parse_args()

  load_library.load_pluggable_device_library(args.plugin)
  tf.config.set_soft_device_placement(False)
  devices = [d.name for d in tf.config.list_physical_devices("GPU")]
  if not devices:
    print("no GPU device after loading the plugin, nothing to sweep")
    return 1
  print(f"sweeping against {devices[0]}")
  global NAMED, RANDOM
  NAMED = by_name()
  NAMED.update(recipes.build())
  RANDOM = recipes.nondeterministic()

  ops = load_ops(args.ops)
  if args.only:
    ops = [o for o in ops if args.only in o]

  duplicates = duplicate_registrations(ops)
  if duplicates:
    print(f"\n=== duplicate GPU registrations ({len(duplicates)})")
    for name in sorted(duplicates):
      print(f"  {name:38s} {duplicates[name]} extra, so the op cannot run")

  results = {}
  details = {}
  for name in ops:
    op_def = op_def_registry.get(name)
    if op_def is None:
      results[name] = NO_RECIPE
      details[name] = "not in the op registry"
      continue
    kwargs = synthesize(op_def)
    if kwargs is None:
      results[name] = NO_RECIPE
      details[name] = "no generic call"
      continue
    if name in RANDOM:
      try:
        gpu = flatten(run(name, kwargs, "/GPU:0"))
      except Exception as error:  # pylint: disable=broad-except
        results[name] = GPU_ERROR
        details[name] = str(error).splitlines()[0][:110]
        continue
      # A random op cannot be compared to the CPU, so it is asked for the two
      # things a broken generator gets wrong: values that are not finite, and
      # values that are all the same.
      bad = []
      for array in gpu:
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
          bad.append("not finite")
        if array.size > 1 and np.all(array == array.flat[0]):
          bad.append("constant")
      results[name] = MISMATCH if bad else MATCH
      details[name] = ", ".join(bad) if bad else "random, finite and varying"
      continue
    try:
      cpu = flatten(run(name, kwargs, "/CPU:0"))
    except Exception as error:  # pylint: disable=broad-except
      results[name] = UNEXERCISED
      details[name] = f"cpu: {str(error).splitlines()[0][:90]}"
      continue
    try:
      gpu = flatten(run(name, kwargs, "/GPU:0"))
    except Exception as error:  # pylint: disable=broad-except
      results[name] = GPU_ERROR
      details[name] = str(error).splitlines()[0][:110]
      continue
    ok, detail = compare(cpu, gpu, name)
    results[name] = MATCH if ok else MISMATCH
    details[name] = detail

  order = [MISMATCH, GPU_ERROR, MATCH, UNEXERCISED, NO_RECIPE]
  counts = {k: 0 for k in order}
  for value in results.values():
    counts[value] += 1

  if os.environ.get("SWEEP_VERBOSE"):
    for kind in (UNEXERCISED, NO_RECIPE):
      named = sorted(n for n, v in results.items() if v == kind)
      print(f"\n=== {kind} ({len(named)})")
      for n in named:
        print(f"  {n:38s} {details[n]}")

  for kind in (MISMATCH, GPU_ERROR):
    named = sorted(n for n, v in results.items() if v == kind)
    if named:
      print(f"\n=== {kind} ({len(named)})")
      for n in named:
        print(f"  {n:38s} {details[n]}")

  print("\n=== summary")
  for kind in order:
    print(f"  {kind:14s} {counts[kind]}")
  print(f"  {'duplicates':14s} {len(duplicates)}")
  return 1 if counts[MISMATCH] or counts[GPU_ERROR] or duplicates else 0


if __name__ == "__main__":
  sys.exit(main())
