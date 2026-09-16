# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Calls for the ops whose inputs are too structured to guess.

Every recipe is a dict of keyword arguments for tf.raw_ops. The sweep runs the
same call on both devices and compares, so a recipe only has to be valid, not
clever: the shapes are small, deliberately not square, and deliberately not a
multiple of four in the innermost dimension, because that is where stride and
alignment mistakes hide.

Ops whose output is random are listed in NONDETERMINISTIC instead: comparing
them to the CPU is meaningless, so the sweep checks shape, dtype and
finiteness, and that the values are not all identical.
"""

import ctypes

import numpy as np
import tensorflow as tf

RNG = np.random.default_rng(11)


def build():
  """Built after the plugin is loaded: making a tensor freezes the context.

  Everything here is built on the host. Once the plugin is loaded the GPU is
  the default device, so an innocent-looking `x[:2]` while assembling inputs
  would run on the device under test, and a recipe that fails to build is
  indistinguishable from a kernel that fails to run.
  """
  with tf.device("/CPU:0"):
    return _build()


def _build():
  f = lambda *s: tf.constant(RNG.standard_normal(s, dtype=np.float32))
  u = lambda *s: tf.constant(
      (RNG.random(s).astype(np.float32) * 0.9 + 0.05))
  image = f(2, 7, 9, 3)          # NHWC, odd spatial sizes
  image5 = f(2, 4, 5, 6, 3)      # NDHWC
  small = u(6, 5)
  square = tf.constant(np.tril(RNG.random((5, 5)).astype(np.float32))
                       + 2.0 * np.eye(5, dtype=np.float32))
  filt = f(3, 3, 3, 4)
  filt3 = f(2, 3, 3, 3, 4)
  nhwc = dict(strides=[1, 1, 1, 1], padding="SAME")
  pool = dict(ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
  pooled = tf.nn.max_pool2d(image, 2, 2, "VALID")
  bn = dict(scale=f(3), offset=f(3), mean=u(3), variance=u(3))
  boxes = tf.constant([[0.0, 0.0, 0.6, 0.6], [0.1, 0.1, 0.9, 0.9],
                       [0.5, 0.5, 1.0, 1.0]], dtype=tf.float32)
  scores = tf.constant([0.9, 0.75, 0.6], dtype=tf.float32)
  sparse_indices = tf.constant([[0, 0], [1, 2], [2, 1], [3, 3]],
                               dtype=tf.int64)
  sparse_values = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
  sparse_shape = tf.constant([4, 4], dtype=tf.int64)

  recipes = {
      # Convolution, forward and both gradients, in two and three dimensions.
      "Conv": {"input": image, "filter": filt, **nhwc},
      "Conv3D": {"input": image5, "filter": filt3,
                 "strides": [1, 1, 1, 1, 1], "padding": "SAME"},
      "Conv2DBackpropInput": {
          "input_sizes": [2, 7, 9, 3], "filter": filt,
          "out_backprop": tf.nn.conv2d(image, filt, 1, "SAME"), **nhwc},
      "Conv2DBackpropFilter": {
          "input": image, "filter_sizes": [3, 3, 3, 4],
          "out_backprop": tf.nn.conv2d(image, filt, 1, "SAME"), **nhwc},
      "Conv3DBackpropInputV2": {
          "input_sizes": [2, 4, 5, 6, 3], "filter": filt3,
          "out_backprop": tf.nn.conv3d(image5, filt3, [1, 1, 1, 1, 1],
                                       "SAME"),
          "strides": [1, 1, 1, 1, 1], "padding": "SAME"},
      "Conv3DBackpropFilterV2": {
          "input": image5, "filter_sizes": [2, 3, 3, 3, 4],
          "out_backprop": tf.nn.conv3d(image5, filt3, [1, 1, 1, 1, 1],
                                       "SAME"),
          "strides": [1, 1, 1, 1, 1], "padding": "SAME"},
      "DepthwiseConv2dNativeBackpropInput": {
          "input_sizes": [2, 7, 9, 3], "filter": filt,
          "out_backprop": tf.nn.depthwise_conv2d(
              image, filt, [1, 1, 1, 1], "SAME"), **nhwc},
      "DepthwiseConv2dNativeBackpropFilter": {
          "input": image, "filter_sizes": [3, 3, 3, 4],
          "out_backprop": tf.nn.depthwise_conv2d(
              image, filt, [1, 1, 1, 1], "SAME"), **nhwc},

      # Pooling and its gradients.
      "MaxPoolV2": {"input": image, "ksize": [1, 2, 2, 1],
                    "strides": [1, 2, 2, 1], "padding": "VALID"},
      "MaxPoolGrad": {"orig_input": image, "orig_output": pooled,
                      "grad": tf.ones_like(pooled), **pool},
      "MaxPoolGradV2": {"orig_input": image, "orig_output": pooled,
                        "grad": tf.ones_like(pooled),
                        "ksize": [1, 2, 2, 1], "strides": [1, 2, 2, 1],
                        "padding": "VALID"},
      "MaxPoolGradGrad": {"orig_input": image, "orig_output": pooled,
                          "grad": tf.ones_like(image), **pool},
      "MaxPoolGradGradV2": {"orig_input": image, "orig_output": pooled,
                            "grad": tf.ones_like(image),
                            "ksize": [1, 2, 2, 1], "strides": [1, 2, 2, 1],
                            "padding": "VALID"},
      "MaxPoolWithArgmax": {"input": image, **pool},
      "AvgPoolGrad": {"orig_input_shape": [2, 7, 9, 3],
                      "grad": tf.nn.avg_pool2d(image, 2, 2, "VALID"), **pool},

      # Normalisation.
      "FusedBatchNorm": {"x": image, **bn, "is_training": False},
      "FusedBatchNormV2": {"x": image, **bn, "is_training": False},
      "FusedBatchNormV3": {"x": image, **bn, "is_training": False},
      "BatchNormWithGlobalNormalization": {
          "t": image, "m": u(3), "v": u(3), "beta": f(3), "gamma": f(3),
          "variance_epsilon": 1e-3, "scale_after_normalization": True},

      # Images.
      "AdjustContrast": {"images": image, "contrast_factor": 1.5,
                         "min_value": -1.0, "max_value": 1.0},
      "ResizeBilinear": {"images": image, "size": [5, 6]},
      "ResizeNearestNeighbor": {"images": image, "size": [5, 6]},
      "ResizeBilinearGrad": {
          "grads": f(2, 5, 6, 3), "original_image": image},
      "ResizeNearestNeighborGrad": {"grads": f(2, 5, 6, 3), "size": [7, 9]},
      "CropAndResize": {"image": image, "boxes": boxes[:2],
                        "box_ind": tf.constant([0, 1], dtype=tf.int32),
                        "crop_size": [4, 5]},
      "ExtractImagePatches": {"images": image, "ksizes": [1, 2, 2, 1],
                              "strides": [1, 1, 1, 1], "rates": [1, 1, 1, 1],
                              "padding": "VALID"},
      "ExtractVolumePatches": {"input": image5, "ksizes": [1, 2, 2, 2, 1],
                               "strides": [1, 1, 1, 1, 1], "padding": "VALID"},
      "Dilation2D": {"input": image, "filter": f(2, 2, 3),
                     "strides": [1, 1, 1, 1], "rates": [1, 1, 1, 1],
                     "padding": "SAME"},
      "ImageProjectiveTransformV2": {
          "images": image, "transforms": tf.constant(
              [[1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 2,
              dtype=tf.float32),
          "output_shape": [7, 9], "interpolation": "BILINEAR"},
      "ImageProjectiveTransformV3": {
          "images": image, "transforms": tf.constant(
              [[1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 2,
              dtype=tf.float32),
          "output_shape": [7, 9], "fill_value": 0.0,
          "interpolation": "BILINEAR"},

      # Layout.
      "DepthToSpace": {"input": f(2, 4, 6, 8), "block_size": 2},
      "SpaceToDepth": {"input": f(2, 4, 6, 3), "block_size": 2},
      "BatchToSpace": {"input": f(8, 2, 3, 3), "crops": [[0, 0], [0, 0]],
                       "block_size": 2},
      "SpaceToBatch": {"input": f(2, 4, 6, 3), "paddings": [[0, 0], [0, 0]],
                       "block_size": 2},
      "BatchToSpaceND": {"input": f(8, 2, 3, 3), "block_shape": [2, 2],
                         "crops": [[0, 0], [0, 0]]},
      "SpaceToBatchND": {"input": f(2, 4, 6, 3), "block_shape": [2, 2],
                         "paddings": [[0, 0], [0, 0]]},
      "MirrorPad": {"input": small, "paddings": [[1, 1], [2, 2]],
                    "mode": "REFLECT"},
      "MirrorPadGrad": {"input": f(8, 9), "paddings": [[1, 1], [2, 2]],
                        "mode": "REFLECT"},
      "Reverse": {"tensor": small, "dims": [False, True]},
      "ReverseV2": {"tensor": small, "axis": [1]},
      "ReverseSequence": {"input": small, "seq_lengths": tf.constant(
          [5, 4, 3, 2, 1, 5], dtype=tf.int64), "seq_dim": 1, "batch_dim": 0},
      "Roll": {"input": small, "shift": [2], "axis": [1]},
      "StridedSlice": {"input": small, "begin": [0, 1], "end": [5, 4],
                       "strides": [2, 1]},
      "SplitV": {"value": small, "size_splits": [2, 4], "axis": 0,
                 "num_split": 2},
      "OneHot": {"indices": tf.constant([0, 2, 1], dtype=tf.int32),
                 "depth": 4, "on_value": 1.0, "off_value": 0.0, "axis": -1},
      "LinSpace": {"start": 0.0, "stop": 1.0, "num": 7},

      # Matrix diagonals.
      "MatrixDiagV2": {"diagonal": f(3, 5), "k": 0, "num_rows": -1,
                       "num_cols": -1, "padding_value": 0.0},
      "MatrixDiagV3": {"diagonal": f(3, 5), "k": 0, "num_rows": -1,
                       "num_cols": -1, "padding_value": 0.0},
      "MatrixDiagPartV2": {"input": f(3, 5, 5), "k": 0,
                           "padding_value": 0.0},
      "MatrixDiagPartV3": {"input": f(3, 5, 5), "k": 0,
                           "padding_value": 0.0},
      "MatrixSetDiagV2": {"input": f(3, 5, 5), "diagonal": f(3, 5), "k": 0},
      "MatrixSetDiagV3": {"input": f(3, 5, 5), "diagonal": f(3, 5), "k": 0},

      # Search and counting.
      "TopK": {"input": small, "k": 3},
      "LowerBound": {"sorted_inputs": tf.constant([[1.0, 3.0, 5.0, 7.0]]),
                     "values": tf.constant([[2.0, 5.0, 8.0]])},
      "UpperBound": {"sorted_inputs": tf.constant([[1.0, 3.0, 5.0, 7.0]]),
                     "values": tf.constant([[2.0, 5.0, 8.0]])},
      "Bucketize": {"input": small, "boundaries": [0.2, 0.5, 0.8]},
      "HistogramFixedWidth": {"values": small, "value_range": [0.0, 1.0],
                              "nbins": 5},
      "InTopK": {"predictions": u(4, 5),
                 "targets": tf.constant([0, 1, 2, 3], dtype=tf.int32), "k": 2},
      "InTopKV2": {"predictions": u(4, 5),
                   "targets": tf.constant([0, 1, 2, 3], dtype=tf.int32),
                   "k": 2},
      "Bincount": {"arr": tf.constant([0, 1, 1, 3], dtype=tf.int32),
                   "size": 5, "weights": tf.constant([], dtype=tf.float32)},
      "DenseBincount": {"input": tf.constant([[0, 1], [1, 3]],
                                             dtype=tf.int32),
                        "size": 5, "weights": tf.constant([],
                                                          dtype=tf.float32),
                        "binary_output": False},

      # Losses and activations with gradients.
      "SoftmaxCrossEntropyWithLogits": {"features": f(4, 5),
                                        "labels": tf.nn.softmax(f(4, 5))},
      "SparseSoftmaxCrossEntropyWithLogits": {
          "features": f(4, 5),
          "labels": tf.constant([0, 1, 2, 3], dtype=tf.int32)},
      "BiasAddGrad": {"out_backprop": image},
      "EluGrad": {"gradients": f(6, 5), "outputs": f(6, 5)},
      "SeluGrad": {"gradients": f(6, 5), "outputs": f(6, 5)},
      "LRNGrad": {"input_grads": image, "input_image": image,
                  "output_image": tf.nn.local_response_normalization(image)},
      "CheckNumerics": {"tensor": small, "message": "check"},
      "CheckNumericsV2": {"tensor": small, "message": "check"},

      # Odds and ends.
      "Atan2": {"y": small, "x": small},
      "Cross": {"a": f(4, 3), "b": f(4, 3)},
      "Betainc": {"a": u(6, 5), "b": u(6, 5), "x": u(6, 5)},
      "Empty": {"shape": [3, 4], "dtype": tf.float32, "init": True},
      "DynamicPartition": {"data": small,
                           "partitions": tf.constant([0, 1, 0, 1, 0, 1],
                                                     dtype=tf.int32),
                           "num_partitions": 2},
      "DynamicStitch": {
          "indices": [tf.constant([0, 2], dtype=tf.int32),
                      tf.constant([1, 3], dtype=tf.int32)],
          "data": [f(2, 5), f(2, 5)]},
      "ParallelDynamicStitch": {
          "indices": [tf.constant([0, 2], dtype=tf.int32),
                      tf.constant([1, 3], dtype=tf.int32)],
          "data": [f(2, 5), f(2, 5)]},
      "NonMaxSuppressionV2": {"boxes": boxes, "scores": scores,
                              "max_output_size": 3,
                              "iou_threshold": 0.5},
      "NonMaxSuppressionV3": {"boxes": boxes, "scores": scores,
                              "max_output_size": 3, "iou_threshold": 0.5,
                              "score_threshold": 0.0},
      "NonMaxSuppressionV4": {"boxes": boxes, "scores": scores,
                              "max_output_size": 3, "iou_threshold": 0.5,
                              "score_threshold": 0.0},

      # Quantisation.

      # Signal. The real transforms take real input and return complex, and
      # the inverses the other way, so each needs its own shape.
      "RFFT": {"input": f(2, 8), "fft_length": [8]},
      "RFFT2D": {"input": f(2, 4, 8), "fft_length": [4, 8]},
      "RFFT3D": {"input": f(2, 4, 4, 8), "fft_length": [4, 4, 8]},
      "IRFFT": {"input": tf.signal.rfft(f(2, 8)), "fft_length": [8]},
      "IRFFT2D": {"input": tf.signal.rfft2d(f(2, 4, 8)),
                  "fft_length": [4, 8]},
      "IRFFT3D": {"input": tf.signal.rfft3d(f(2, 4, 4, 8)),
                  "fft_length": [4, 4, 8]},

      # Sparse. The dense-output ops are the ones this backend implements.
      "SparseToDense": {"sparse_indices": sparse_indices,
                        "output_shape": sparse_shape,
                        "sparse_values": sparse_values, "default_value": 0.0},
      "SparseTensorDenseMatMul": {"a_indices": sparse_indices,
                                  "a_values": sparse_values,
                                  "a_shape": sparse_shape, "b": f(4, 3)},
  }

  # The segment families share a shape, so they are generated rather than
  # written out eight times over.
  seg_data = f(6, 4)
  seg_idx = tf.constant([0, 1, 2, 3, 4, 5], dtype=tf.int32)
  seg_ids = tf.constant([0, 0, 1, 1, 2, 2], dtype=tf.int32)
  for stem in ("Sum", "Mean", "SqrtN"):
    recipes[f"SparseSegment{stem}"] = {
        "data": seg_data, "indices": seg_idx, "segment_ids": seg_ids}
    recipes[f"SparseSegment{stem}WithNumSegments"] = {
        "data": seg_data, "indices": seg_idx, "segment_ids": seg_ids,
        "num_segments": 3}
    recipes[f"SparseSegment{stem}Grad"] = {
        "grad": f(3, 4), "indices": seg_idx, "segment_ids": seg_ids,
        "output_dim0": 6}
    recipes[f"SparseSegment{stem}GradV2"] = {
        "grad": f(3, 4), "indices": seg_idx, "segment_ids": seg_ids,
        "dense_output_dim0": 6}
  return recipes


# Ops that need the kernel C API entry points for resource variables. The
# plugin registers them only when those entry points are there, so whether
# this set is an exemption or a list of ops to sweep depends on the
# TensorFlow underneath rather than on anything here: see
# resource_variable_api_available() below.
NEEDS_UNEXPORTED_C_API = {
    "Assign", "AssignAdd", "AssignSub",
    "ResourceApplyAdam", "ResourceApplyGradientDescent",
    "ResourceApplyKerasMomentum", "ResourceApplyMomentum",
    "ResourceApplyRMSProp",
    "ResourceGather", "ResourceGatherNd", "ResourceScatterUpdate",
    "ParallelConcat", "_ParallelConcatStart", "_ParallelConcatUpdate",
}

# The same six symbols the plugin looks for at load, looked up the same way.
_RESOURCE_VARIABLE_ENTRY_POINTS = (
    "TF_AssignVariable",
    "TF_GetInputTensorFromVariable",
    "TF_MaybeLockVariableInputMutexesInOrder",
    "TF_ReleaseVariableInputLockHolder",
    "TF_OpKernelConstruction_GetAttrTensorShape",
    "TF_OpKernelContext_ForwardRefInputToRefOutput",
)


def resource_variable_api_available():
  """Whether the loaded TensorFlow exports the resource variable entry points.

  Asked rather than assumed. These were declared in the headers and exported
  by nothing from 2.20 until tensorflow/tensorflow#126377 was merged on
  2026-09-10, which is why NEEDS_UNEXPORTED_C_API exists at all. A sweep that
  keeps exempting them after that fix ships would stop measuring fourteen ops
  on the day they start working, and would say "needs unexported api" about a
  TensorFlow that exports them.

  TensorFlow is already loaded into this process, so its symbols are reachable
  through the global handle, which is where the plugin's own C++ check looks.
  """
  handle = ctypes.CDLL(None)
  for name in _RESOURCE_VARIABLE_ENTRY_POINTS:
    if not hasattr(handle, name):
      return False
  return True


def more():
  with tf.device("/CPU:0"):
    return _more()


def _more():
  f = lambda *s: tf.constant(RNG.standard_normal(s, dtype=np.float32))
  u = lambda *s: tf.constant(RNG.random(s).astype(np.float32) * 0.9 + 0.05)
  small = u(6, 5)
  complex_1d = tf.complex(f(2, 8), f(2, 8))
  complex_2d = tf.complex(f(2, 4, 8), f(2, 4, 8))
  complex_3d = tf.complex(f(2, 4, 4, 8), f(2, 4, 4, 8))
  boolean = tf.constant(RNG.random((6, 5)) > 0.5)
  image = f(2, 7, 9, 3)

  recipes = {
      # Reductions name their axis input "axis" in the generated wrappers even
      # though the op def calls it reduction_indices.
      "Sum": {"input": small, "axis": [1]},
      "Mean": {"input": small, "axis": [1]},
      "Max": {"input": small, "axis": [1]},
      "Min": {"input": small, "axis": [1]},
      "Prod": {"input": small, "axis": [1]},
      "All": {"input": boolean, "axis": [1]},
      "Any": {"input": boolean, "axis": [1]},
      "EuclideanNorm": {"input": small, "axis": [1]},

      "LogicalAnd": {"x": boolean, "y": tf.constant(RNG.random((6, 5)) > 0.5)},
      "LogicalOr": {"x": boolean, "y": tf.constant(RNG.random((6, 5)) > 0.5)},
      "LogicalNot": {"x": boolean},
      "Select": {"condition": boolean, "x": small, "y": u(6, 5)},
      "Conj": {"input": complex_1d},

      "AvgPool": {"value": image, "ksize": [1, 2, 2, 1],
                  "strides": [1, 2, 2, 1], "padding": "VALID"},
      "TopK": {"input": small, "k": 3},
      "Split": {"axis": 0, "value": small, "num_split": 2},
      "TileGrad": {"input": f(12, 5), "multiples": [2, 1]},
      "UniqueWithCounts": {"x": tf.constant([1, 2, 2, 3, 1], dtype=tf.int32)},
      "DiagPart": {"input": f(5, 5)},
      "GatherV2": {"params": small, "indices": tf.constant([0, 2, 1],
                                                           dtype=tf.int32),
                   "axis": 0},
      "GatherNd": {"params": small,
                   "indices": tf.constant([[0, 1], [2, 3]], dtype=tf.int32)},
      "BatchMatMul": {"x": f(2, 4, 5), "y": f(2, 5, 3)},
      "BatchMatMulV3": {"x": f(2, 4, 5), "y": f(2, 5, 3), "Tout": tf.float32},

      # The complex transforms, forward and inverse, in one to three
      # dimensions.
      "FFT": {"input": complex_1d},
      "IFFT": {"input": complex_1d},
      "FFT2D": {"input": complex_2d},
      "IFFT2D": {"input": complex_2d},
      "FFT3D": {"input": complex_3d},
      "IFFT3D": {"input": complex_3d},
      "FFTND": {"input": complex_2d, "fft_length": [4, 8], "axes": [1, 2]},
      "IFFTND": {"input": complex_2d, "fft_length": [4, 8], "axes": [1, 2]},
      "RFFTND": {"input": f(2, 4, 8), "fft_length": [4, 8], "axes": [1, 2]},
      "IRFFTND": {"input": tf.signal.rfft2d(f(2, 4, 8)),
                  "fft_length": [4, 8], "axes": [1, 2]},

      # Gradients whose forward pass is already checked.
      "Conv3DBackpropInput": {
          "input_sizes": [2, 4, 5, 6, 3], "filter": f(2, 3, 3, 3, 4),
          "out_backprop": f(2, 4, 5, 6, 4),
          "strides": [1, 1, 1, 1, 1], "padding": "SAME"},
      "Conv3DBackpropFilter": {
          "input": f(2, 4, 5, 6, 3), "filter": f(2, 3, 3, 3, 4),
          "out_backprop": f(2, 4, 5, 6, 4),
          "strides": [1, 1, 1, 1, 1], "padding": "SAME"},
      "StridedSliceGrad": {"shape": [6, 5], "begin": [0, 1], "end": [5, 4],
                           "strides": [2, 1], "dy": f(3, 3)},
      "Dilation2DBackpropInput": {
          "input": image, "filter": f(2, 2, 3),
          "out_backprop": tf.nn.dilation2d(
              image, f(2, 2, 3), [1, 1, 1, 1], "SAME", "NHWC", [1, 1, 1, 1]),
          "strides": [1, 1, 1, 1], "rates": [1, 1, 1, 1], "padding": "SAME"},
      "Dilation2DBackpropFilter": {
          "input": image, "filter": f(2, 2, 3),
          "out_backprop": tf.nn.dilation2d(
              image, f(2, 2, 3), [1, 1, 1, 1], "SAME", "NHWC", [1, 1, 1, 1]),
          "strides": [1, 1, 1, 1], "rates": [1, 1, 1, 1], "padding": "SAME"},

      # The recurrent block cells.

      # The collectives, over the one device this backend has.
  }
  return recipes


# Random ops cannot be compared against the CPU: the point of them is that the
# answer differs. They are checked for shape, dtype, finiteness and for not
# being constant, which is what a broken generator usually returns.
def nondeterministic():
  with tf.device("/CPU:0"):
    return _nondeterministic()


def _nondeterministic():
  f = lambda *s: tf.constant(RNG.standard_normal(s, dtype=np.float32))
  return {
      "RandomUniform": {"shape": [4, 5], "dtype": tf.float32},
      "RandomStandardNormal": {"shape": [4, 5], "dtype": tf.float32},
      "TruncatedNormal": {"shape": [4, 5], "dtype": tf.float32},
      "RandomUniformInt": {"shape": [4, 5], "minval": 0, "maxval": 10},
      "Multinomial": {"logits": f(2, 5), "num_samples": 6},
      "RandomGamma": {"shape": [4], "alpha": tf.constant([2.0, 3.0])},
      "ParameterizedTruncatedNormal": {
          "shape": [2, 5], "means": tf.constant([0.0, 0.0]),
          "stdevs": tf.constant([1.0, 1.0]),
          "minvals": tf.constant([-2.0, -2.0]),
          "maxvals": tf.constant([2.0, 2.0])},
      "StatelessMultinomial": {"logits": f(2, 5), "num_samples": 6,
                               "seed": tf.constant([1, 2], dtype=tf.int32)},
      "StatelessParameterizedTruncatedNormal": {
          "shape": [2, 5], "seed": tf.constant([1, 2], dtype=tf.int32),
          "means": 0.0, "stddevs": 1.0, "minvals": -2.0, "maxvals": 2.0},
      "StatelessRandomGammaV2": {
          "shape": [4], "seed": tf.constant([1, 2], dtype=tf.int32),
          "alpha": tf.constant([2.0, 3.0, 2.0, 3.0])},
      "StatelessRandomGammaV3": {
          "shape": [4], "key": tf.constant([1], dtype=tf.uint64),
          "counter": tf.constant([1, 2], dtype=tf.uint64), "alg": 3,
          "alpha": tf.constant([2.0, 3.0, 2.0, 3.0])},
  }


# Ops TensorFlow has no CPU kernel for, so there is nothing to compare
# against. Each is checked against a property that pins it down instead.
def no_cpu_reference():
  with tf.device("/CPU:0"):
    return _no_cpu_reference()


def _no_cpu_reference():
  f = lambda *s: tf.constant(RNG.standard_normal(s, dtype=np.float32))
  real = f(2, 4, 8)
  spectrum = tf.signal.rfft2d(real)
  complex_2d = tf.complex(f(2, 4, 8), f(2, 4, 8))
  return {
      "FFTND": ({"input": complex_2d, "fft_length": [4, 8], "axes": [1, 2]},
                "round trip through IFFTND"),
      "IFFTND": ({"input": complex_2d, "fft_length": [4, 8], "axes": [1, 2]},
                 "round trip through FFTND"),
      "RFFTND": ({"input": real, "fft_length": [4, 8], "axes": [1, 2]},
                 "matches RFFT2D"),
      "IRFFTND": ({"input": spectrum, "fft_length": [4, 8], "axes": [1, 2]},
                  "matches IRFFT2D"),
  }


def gradients_and_rest():
  with tf.device("/CPU:0"):
    return _gradients_and_rest()


def _gradients_and_rest():
  f = lambda *s: tf.constant(RNG.standard_normal(s, dtype=np.float32))
  u = lambda *s: tf.constant(RNG.random(s).astype(np.float32) * 0.9 + 0.05)
  image = f(2, 7, 9, 3)
  image5 = f(2, 4, 5, 6, 3)
  filt3 = f(2, 3, 3, 3, 4)
  pooled = tf.nn.max_pool2d(image, 2, 2, "VALID")
  argmax = tf.raw_ops.MaxPoolWithArgmax(input=image, ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1], padding="VALID")
  pool = dict(ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
  bn = dict(scale=f(3), offset=f(3), mean=u(3), variance=u(3))
  forward = tf.raw_ops.FusedBatchNormV3(x=image, **bn, is_training=False)
  boxes = tf.constant([[0.0, 0.0, 0.6, 0.6], [0.1, 0.1, 0.9, 0.9]],
                      dtype=tf.float32)
  box_index = tf.constant([0, 1], dtype=tf.int32)
  cropped = tf.raw_ops.CropAndResize(image=image, boxes=boxes,
                                     box_ind=box_index, crop_size=[4, 5])
  sparse_indices = tf.constant([[0, 0], [1, 2], [2, 1], [3, 3]],
                               dtype=tf.int64)
  sparse_shape = tf.constant([4, 4], dtype=tf.int64)

  return {
      # Normalisation gradients, in training mode, which is the only mode
      # where the reserve spaces mean anything.
      "FusedBatchNormGrad": {
          "y_backprop": image, "x": image, "scale": bn["scale"],
          "reserve_space_1": forward[3], "reserve_space_2": forward[4],
          "is_training": False},
      "FusedBatchNormGradV2": {
          "y_backprop": image, "x": image, "scale": bn["scale"],
          "reserve_space_1": forward[3], "reserve_space_2": forward[4],
          "is_training": False},
      "FusedBatchNormGradV3": {
          "y_backprop": image, "x": image, "scale": bn["scale"],
          "reserve_space_1": forward[3], "reserve_space_2": forward[4],
          "reserve_space_3": forward[5], "is_training": False},
      "BatchNormWithGlobalNormalizationGrad": {
          "t": image, "m": u(3), "v": u(3), "gamma": f(3), "backprop": image,
          "variance_epsilon": 1e-3, "scale_after_normalization": True},

      # Pooling gradients that carry indices.
      "MaxPoolGradWithArgmax": {
          "input": image, "grad": tf.ones_like(pooled),
          "argmax": argmax[1], **pool},
      "MaxPoolGradGradWithArgmax": {
          "input": image, "grad": tf.ones_like(image),
          "argmax": argmax[1], **pool},

      # Image gradients.
      "CropAndResizeGradImage": {
          "grads": tf.ones_like(cropped), "boxes": boxes,
          "box_ind": box_index, "image_size": [2, 7, 9, 3], "T": tf.float32},
      "CropAndResizeGradBoxes": {
          "grads": tf.ones_like(cropped), "image": image, "boxes": boxes,
          "box_ind": box_index},

      # The deprecated three-dimensional gradient names.
      "Conv3DBackpropInput": {
          "input": image5, "filter": filt3,
          "out_backprop": tf.nn.conv3d(image5, filt3, [1, 1, 1, 1, 1], "SAME"),
          "strides": [1, 1, 1, 1, 1], "padding": "SAME"},
      "Conv3DBackpropFilter": {
          "input": image5, "filter": filt3,
          "out_backprop": tf.nn.conv3d(image5, filt3, [1, 1, 1, 1, 1], "SAME"),
          "strides": [1, 1, 1, 1, 1], "padding": "SAME"},

      # Recurrent gradients.

      # Sparse and ragged.
      "SparseBincount": {
          "indices": sparse_indices,
          "values": tf.constant([0, 1, 1, 3], dtype=tf.int32),
          "dense_shape": sparse_shape, "size": 5,
          "weights": tf.constant([], dtype=tf.float32),
          "binary_output": False},
      "RaggedBincount": {
          "splits": tf.constant([0, 2, 4], dtype=tf.int64),
          "values": tf.constant([0, 1, 1, 3], dtype=tf.int32), "size": 5,
          "weights": tf.constant([], dtype=tf.float32),
          "binary_output": False},

      # Sequence and detection.
      "CTCLoss": {
          "inputs": f(6, 2, 5),
          "labels_indices": tf.constant([[0, 0], [0, 1], [1, 0]],
                                        dtype=tf.int64),
          "labels_values": tf.constant([1, 2, 1], dtype=tf.int32),
          "sequence_length": tf.constant([6, 6], dtype=tf.int32)},
      "CTCLossV2": {
          "inputs": f(6, 2, 5),
          "labels_indices": tf.constant([[0, 0], [0, 1], [1, 0]],
                                        dtype=tf.int64),
          "labels_values": tf.constant([1, 2, 1], dtype=tf.int32),
          "sequence_length": tf.constant([6, 6], dtype=tf.int32)},

      # Fused forms the grappler emits.
      "_FusedMatMul": {
          "a": u(6, 5), "b": u(5, 4), "args": [f(4)],
          "fused_ops": ["BiasAdd"], "num_args": 1},
      "_FusedConv2D": {
          "input": image, "filter": f(3, 3, 3, 4), "args": [f(4)],
          "strides": [1, 1, 1, 1], "padding": "SAME",
          "fused_ops": ["BiasAdd"], "num_args": 1},

      # The remaining collectives.

      "TopK": {"input": u(6, 5), "k": 3},
      "TileGrad": {"input": f(12, 5), "multiples": [2, 1]},
      "_TensorToHashBucketFast": {"input": tf.constant([1, 2, 3],
                                                       dtype=tf.int32),
                                  "num_buckets": 7},
      "DebugNumericSummaryV2": {"input": f(6, 5), "tensor_debug_mode": 2},
  }


def internal():
  """Ops with no Python wrapper, called straight through the eager executor.

  TensorFlow generates wrappers only for public ops, so the fused forms the
  grappler emits and the collectives' halves are absent from tf.raw_ops.
  Skipping them would leave a tenth of what the grappler actually produces
  unexercised, so each is given its inputs and its attributes by hand.
  """
  with tf.device("/CPU:0"):
    f = lambda *s: tf.constant(RNG.standard_normal(s, dtype=np.float32))
    u = lambda *s: tf.constant(RNG.random(s).astype(np.float32) * 0.9 + 0.05)
    a, b, bias = u(6, 5), u(5, 4), f(4)
    image, filt, cbias = f(2, 7, 9, 3), f(3, 3, 3, 4), f(4)
    scale, offset, mean, variance = f(3), f(3), u(3), u(3)
    integers = tf.constant([1, 2, 3, 4], dtype=tf.int32)
  ft = tf.float32.as_datatype_enum
  return {
      "_FusedMatMul": (
          [a, b, bias],
          ("T", ft, "transpose_a", False, "transpose_b", False,
           "num_args", 1, "fused_ops", ["BiasAdd"], "epsilon", 1e-4,
           "leakyrelu_alpha", 0.2), 1),
      "_FusedConv2D": (
          [image, filt, cbias],
          ("T", ft, "TArgs", [ft], "num_args", 1, "num_host_args", 0,
           "strides", [1, 1, 1, 1], "padding", b"SAME",
           "explicit_paddings", [], "data_format", b"NHWC",
           "dilations", [1, 1, 1, 1], "use_cudnn_on_gpu", True,
           "fused_ops", ["BiasAdd"], "epsilon", 1e-4,
           "leakyrelu_alpha", 0.2), 1),
      "_FusedBatchNormEx": (
          [image, scale, offset, mean, variance],
          ("T", ft, "U", ft, "epsilon", 1e-3, "exponential_avg_factor", 1.0,
           "num_side_inputs", 0, "activation_mode", b"Identity",
           "data_format", b"NHWC", "is_training", False), 6),
      "_FusedBatchNormGradEx": (
          [image, image, scale, mean, variance,
           tf.constant([], dtype=tf.float32), offset, image],
          ("T", ft, "U", ft, "epsilon", 1e-3, "num_side_inputs", 0,
           "activation_mode", b"Identity", "data_format", b"NHWC",
           "is_training", False), 5),
      "_TensorToHashBucketFast": (
          [integers], ("T", tf.int32.as_datatype_enum, "num_buckets", 7), 1),
  }


def last_few():
  """The remainder: ops TensorFlow cannot run on the CPU, and a few stragglers.

  TopK and TileGrad are here because TensorFlow's own CPU registrations for
  them do not match their nodes in this release, so there is no reference to
  compare against even though the ops are perfectly ordinary. They are checked
  against what they mean instead.
  """
  with tf.device("/CPU:0"):
    f = lambda *s: tf.constant(RNG.standard_normal(s, dtype=np.float32))
    u = lambda *s: tf.constant(RNG.random(s).astype(np.float32) * 0.9 + 0.05)
    values = u(6, 5)
    tiled = f(12, 5)
    image = f(2, 7, 9, 3)
    logits = f(6, 2, 5)
    lengths = tf.constant([6, 6], dtype=tf.int32)
    argmax = tf.raw_ops.MaxPoolWithArgmax(input=image, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1],
                                          padding="VALID")
    scores = tf.constant([[0.9, 0.75, 0.6, 0.5]], dtype=tf.float32)
    boxes = tf.constant([[[0.0, 0.0, 0.6, 0.6], [0.1, 0.1, 0.9, 0.9],
                          [0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.4, 0.4]]],
                        dtype=tf.float32)
    anchors = tf.constant([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5],
                           [0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.6, 0.6]],
                          dtype=tf.float32)
  return {
      "CTCLossV2": ({"inputs": logits,
                     "labels_indices": tf.constant([[0, 0], [0, 1], [1, 0]],
                                                   dtype=tf.int64),
                     "labels_values": tf.constant([1, 2, 1], dtype=tf.int32),
                     "sequence_length": lengths},
                    "agrees with CTCLoss"),
      "MaxPoolGradGradWithArgmax": (
          {"input": image, "grad": tf.ones_like(image), "argmax": argmax[1],
           "ksize": [1, 2, 2, 1], "strides": [1, 2, 2, 1], "padding": "VALID"},
          "finite, and shaped like the pooled output"),
      "ApproxTopK": ({"input": values, "k": 3},
                     "the three largest, in order"),
      "GenerateBoundingBoxProposals": (
          {"scores": tf.reshape(scores, [1, 1, 1, 4]),
           "bbox_deltas": tf.zeros([1, 1, 1, 16], dtype=tf.float32),
           "image_info": tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0]],
                                     dtype=tf.float32),
           "anchors": anchors, "nms_threshold": 0.7, "pre_nms_topn": 4,
           "min_size": 0.0},
          "finite, and no more boxes than were offered"),
  }
