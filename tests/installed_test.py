# Copyright 2026 The TensorFlow Metal Plugin Authors. All Rights Reserved.
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
"""What a user gets after `pip install`, and nothing else.

Every other test here loads the freshly built dylib by hand, which proves the
kernels and proves nothing about the package: a wheel that puts the shared
object in the wrong directory passes all of them, and fails the only step a
user actually performs.

So this file loads nothing. It imports TensorFlow and asks what devices exist.
If the answer includes a GPU, the install worked, because the only thing that
could have put it there is TensorFlow scanning `site-packages/tensorflow-plugins`
at import.

Run it against an interpreter the package is installed into:

    pip install .
    python tests/installed_test.py
"""

import sys

import numpy as np
import tensorflow as tf

FAILURES = []


def check(name, condition, detail=""):
  print(f"  {name:46s} {'ok' if condition else 'FAILED'}"
        f"{'  ' + detail if detail else ''}")
  if not condition:
    FAILURES.append(name)


def main():
  rng = np.random.default_rng(0)
  print(f"\ninstalled package, TensorFlow {tf.__version__}:")

  devices = tf.config.list_physical_devices()
  names = [d.name for d in devices]
  check("a GPU device appears with no manual load",
        any(d.device_type == "GPU" for d in devices), str(names))
  if FAILURES:
    print("\nno GPU device, so nothing below would mean anything")
    return 1

  # Numerics, because a device that registers and answers wrongly is worse
  # than no device.
  a = rng.standard_normal((256, 384), dtype=np.float32)
  b = rng.standard_normal((384, 128), dtype=np.float32)
  with tf.device("/GPU:0"):
    on_gpu = tf.matmul(tf.constant(a), tf.constant(b)).numpy()
  with tf.device("/CPU:0"):
    on_cpu = tf.matmul(tf.constant(a), tf.constant(b)).numpy()
  worst = float(np.max(np.abs(on_gpu - on_cpu)))
  check("MatMul agrees with the CPU", worst < 1e-3, f"max diff {worst:.2e}")

  # An ordinary Keras script, unmodified: no device scope, no plugin-specific
  # anything. This is the definition of done for the package.
  print("\na standard Keras training script:")
  images = rng.standard_normal((512, 16, 16, 3)).astype(np.float32)
  labels = rng.integers(0, 10, size=512)
  model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(16, 16, 3)),
      tf.keras.layers.Conv2D(16, 3, activation="relu"),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation="relu"),
      tf.keras.layers.Dense(10),
  ])
  model.compile(
      optimizer="adam",
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
  history = model.fit(images, labels, epochs=3, batch_size=64, verbose=0)
  losses = history.history["loss"]
  check("it trains", bool(np.all(np.isfinite(losses))),
        " ".join(f"{v:.4f}" for v in losses))
  # Three epochs on random labels will not converge, but a training step that
  # is wired up at all reduces the loss on data it has seen this many times.
  check("the loss falls", losses[-1] < losses[0],
        f"{losses[0]:.4f} to {losses[-1]:.4f}")

  # The fallback, under the placement a user actually has, which is soft.
  # `run_tests.py` checks the opposite direction with soft placement off.
  with tf.device("/GPU:0"):
    inverted = tf.linalg.inv(tf.eye(4) * 2.0)
  check("an op with no Metal kernel runs on the host",
        inverted.device.endswith("CPU:0"), inverted.device)
  check("and answers correctly",
        float(np.max(np.abs(inverted.numpy() - np.eye(4) * 0.5))) < 1e-6)

  if FAILURES:
    print(f"\n{len(FAILURES)} failed: {', '.join(FAILURES)}")
    return 1
  print("\nthe installed package works")
  return 0


if __name__ == "__main__":
  sys.exit(main())
