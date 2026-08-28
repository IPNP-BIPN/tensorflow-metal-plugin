"""Checks the plugin against the TensorFlow it is loaded into.

Every numeric check compares a GPU result to the CPU kernel for the same op,
with soft placement off, so a missing GPU kernel raises instead of quietly
producing the right answer on the wrong device.
"""

import os

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.framework import load_library

PLUGIN = os.environ.get(
    "METAL_PLUGIN",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "build", "libmetal_plugin.dylib"))


@pytest.fixture(scope="session", autouse=True)
def plugin():
  load_library.load_pluggable_device_library(PLUGIN)
  tf.config.set_soft_device_placement(False)


def test_device_appears():
  names = [d.name for d in tf.config.list_physical_devices("GPU")]
  assert names == ["/physical_device:GPU:0"], names


def _rng():
  return np.random.default_rng(0)


@pytest.mark.parametrize("name", [
    "matmul", "softmax", "conv2d", "relu", "maxpool", "reduce_sum", "transpose",
])
def test_matches_cpu(name):
  rng = _rng()
  a = rng.standard_normal((64, 48), dtype=np.float32)
  x = rng.standard_normal((2, 16, 16, 3), dtype=np.float32)
  k = rng.standard_normal((3, 3, 3, 8), dtype=np.float32)

  def run(op):
    return {
        "matmul": lambda: tf.matmul(tf.constant(a), tf.constant(a.T)),
        "softmax": lambda: tf.nn.softmax(tf.constant(a)),
        "conv2d": lambda: tf.nn.conv2d(x, k, strides=1, padding="SAME"),
        "relu": lambda: tf.nn.relu(tf.constant(a)),
        "maxpool": lambda: tf.nn.max_pool2d(x, 2, 2, "VALID"),
        "reduce_sum": lambda: tf.reduce_sum(tf.constant(a), axis=1),
        "transpose": lambda: tf.transpose(tf.constant(a)),
    }[op]()

  with tf.device("/GPU:0"):
    got = run(name)
  assert got.device.endswith("GPU:0"), got.device
  with tf.device("/CPU:0"):
    want = run(name)
  np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-4, atol=2e-4)


def test_soft_placement_is_really_off():
  """Without this, every check above could be passing on the CPU."""
  with pytest.raises(Exception):
    with tf.device("/GPU:0"):
      tf.raw_ops.MatrixDeterminant(input=tf.eye(4))


def test_host_round_trip():
  rng = _rng()
  data = rng.standard_normal((257, 33), dtype=np.float32)
  with tf.device("/GPU:0"):
    on_device = tf.constant(data)
  np.testing.assert_array_equal(on_device.numpy(), data)
