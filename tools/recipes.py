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
      "AdjustContrastv2": {"images": image, "contrast_factor": 1.5},
      "AdjustHue": {"images": u(2, 7, 9, 3), "delta": 0.2},
      "AdjustSaturation": {"images": u(2, 7, 9, 3), "scale": 1.4},
      "RGBToHSV": {"images": u(2, 7, 9, 3)},
      "HSVToRGB": {"images": u(2, 7, 9, 3)},
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
      "FakeQuantWithMinMaxArgs": {"inputs": f(6, 5), "min": -1.0, "max": 1.0},
      "FakeQuantWithMinMaxArgsGradient": {"gradients": f(6, 5),
                                          "inputs": f(6, 5),
                                          "min": -1.0, "max": 1.0},
      "FakeQuantWithMinMaxVars": {"inputs": f(6, 5), "min": -1.0, "max": 1.0},
      "FakeQuantWithMinMaxVarsGradient": {"gradients": f(6, 5),
                                          "inputs": f(6, 5),
                                          "min": -1.0, "max": 1.0},
      "FakeQuantWithMinMaxVarsPerChannel": {
          "inputs": f(6, 5), "min": tf.constant([-1.0] * 5),
          "max": tf.constant([1.0] * 5)},
      "FakeQuantWithMinMaxVarsPerChannelGradient": {
          "gradients": f(6, 5), "inputs": f(6, 5),
          "min": tf.constant([-1.0] * 5), "max": tf.constant([1.0] * 5)},
      "QuantizeAndDequantizeV2": {"input": f(6, 5), "input_min": -1.0,
                                  "input_max": 1.0},
      "QuantizeAndDequantizeV3": {"input": f(6, 5), "input_min": -1.0,
                                  "input_max": 1.0, "num_bits": 8},
      "QuantizeAndDequantizeV4": {"input": f(6, 5), "input_min": -1.0,
                                  "input_max": 1.0},
      "QuantizeAndDequantizeV4Grad": {"gradients": f(6, 5), "input": f(6, 5),
                                      "input_min": -1.0, "input_max": 1.0},

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
      "SparseReorder": {"input_indices": tf.constant([[1, 0], [0, 1]],
                                                     dtype=tf.int64),
                        "input_values": tf.constant([1.0, 2.0]),
                        "input_shape": sparse_shape},
      "SparseReshape": {"input_indices": sparse_indices,
                        "input_shape": sparse_shape,
                        "new_shape": tf.constant([2, 8], dtype=tf.int64)},
      "SparseFillEmptyRows": {"indices": sparse_indices,
                              "values": sparse_values,
                              "dense_shape": sparse_shape,
                              "default_value": 0.0},
      "SparseSlice": {"indices": sparse_indices, "values": sparse_values,
                      "shape": sparse_shape,
                      "start": tf.constant([0, 0], dtype=tf.int64),
                      "size": tf.constant([2, 4], dtype=tf.int64)},
      "SparseSplit": {"split_dim": tf.constant(0, dtype=tf.int64),
                      "indices": sparse_indices, "values": sparse_values,
                      "shape": sparse_shape, "num_split": 2},
      "SparseConcat": {"indices": [sparse_indices, sparse_indices],
                       "values": [sparse_values, sparse_values],
                       "shapes": [sparse_shape, sparse_shape],
                       "concat_dim": 0},
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


# Ops TensorFlow itself will not run any more, whatever the device: their
# kernels raise "not available in GraphDef version ... It has been removed in
# version ...". They are registered here because the CUDA build registers
# them, and a registration is all either build can offer.
REMOVED_FROM_GRAPHDEF = {
    "BatchFFT", "BatchFFT2D", "BatchFFT3D", "BatchIFFT", "BatchIFFT2D",
    "BatchIFFT3D", "BatchMatrixBandPart", "BatchMatrixDiag",
    "BatchMatrixDiagPart", "BatchMatrixSetDiag", "BatchMatrixTriangularSolve",
    "QuantizeAndDequantize", "AdjustContrast",
}

# Ops that need entry points a released TensorFlow does not export, so the
# plugin deliberately leaves them to the host. In an in-tree build they are
# registered and work; out of tree there is nothing to test.
NEEDS_UNEXPORTED_C_API = {
    "Assign", "AssignAdd", "AssignSub",
    "ResourceApplyAdam", "ResourceApplyGradientDescent",
    "ResourceApplyKerasMomentum", "ResourceApplyMomentum",
    "ResourceApplyRMSProp",
    "ResourceGather", "ResourceGatherNd", "ResourceScatterUpdate",
    "ParallelConcat", "_ParallelConcatStart", "_ParallelConcatUpdate",
}


def more():
  with tf.device("/CPU:0"):
    return _more()


def _more():
  f = lambda *s: tf.constant(RNG.standard_normal(s, dtype=np.float32))
  u = lambda *s: tf.constant(RNG.random(s).astype(np.float32) * 0.9 + 0.05)
  small = u(6, 5)
  square = tf.constant(RNG.random((5, 5)).astype(np.float32) + np.eye(5, dtype=np.float32) * 4)
  symmetric = tf.constant(((square + tf.transpose(square)) / 2).numpy())
  complex_1d = tf.complex(f(2, 8), f(2, 8))
  complex_2d = tf.complex(f(2, 4, 8), f(2, 4, 8))
  complex_3d = tf.complex(f(2, 4, 4, 8), f(2, 4, 4, 8))
  boolean = tf.constant(RNG.random((6, 5)) > 0.5)
  image = f(2, 7, 9, 3)
  # One recurrent step's worth of state, shared by the block cells.
  batch, units, inputs, steps = 3, 4, 5, 6
  x_seq = f(steps, batch, inputs)
  h0 = f(batch, units)
  c0 = f(batch, units)
  w_lstm = f(inputs + units, 4 * units)
  b_lstm = f(4 * units)
  wci = f(units)

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
      "Lu": {"input": square},
      "SelfAdjointEigV2": {"input": symmetric, "compute_v": True},
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
      "LSTMBlockCell": {
          "x": f(batch, inputs), "cs_prev": c0, "h_prev": h0, "w": w_lstm,
          "wci": wci, "wcf": wci, "wco": wci, "b": b_lstm},
      "BlockLSTM": {
          "seq_len_max": tf.constant(steps, dtype=tf.int64), "x": x_seq,
          "cs_prev": c0, "h_prev": h0, "w": w_lstm, "wci": wci, "wcf": wci,
          "wco": wci, "b": b_lstm},
      "BlockLSTMV2": {
          "seq_len_max": tf.constant(steps, dtype=tf.int64), "x": x_seq,
          "cs_prev": c0, "h_prev": h0, "w": w_lstm, "wci": wci, "wcf": wci,
          "wco": wci, "b": b_lstm},
      "GRUBlockCell": {
          "x": f(batch, inputs), "h_prev": h0,
          "w_ru": f(inputs + units, 2 * units), "w_c": f(inputs + units, units),
          "b_ru": f(2 * units), "b_c": f(units)},

      # The collectives, over the one device this backend has.
      "NcclAllReduce": {"input": small, "reduction": "sum", "num_devices": 1,
                        "shared_name": "one"},
      "NcclBroadcast": {"input": small, "shape": [6, 5]},
      "NcclReduce": {"input": [small], "reduction": "sum"},
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
  small = tf.constant(RNG.random((6, 5)).astype(np.float32))
  return {
      "FFTND": ({"input": complex_2d, "fft_length": [4, 8], "axes": [1, 2]},
                "round trip through IFFTND"),
      "IFFTND": ({"input": complex_2d, "fft_length": [4, 8], "axes": [1, 2]},
                 "round trip through FFTND"),
      "RFFTND": ({"input": real, "fft_length": [4, 8], "axes": [1, 2]},
                 "matches RFFT2D"),
      "IRFFTND": ({"input": spectrum, "fft_length": [4, 8], "axes": [1, 2]},
                  "matches IRFFT2D"),
      "NcclAllReduce": ({"input": small, "reduction": "sum",
                         "num_devices": 1, "shared_name": "sweep"},
                        "reducing over one device is a copy"),
      "NcclBroadcast": ({"input": small, "shape": [6, 5]},
                        "broadcasting to one device is a copy"),
      "NcclReduce": ({"input": [small], "reduction": "sum"},
                     "reducing one input is a copy"),
  }
