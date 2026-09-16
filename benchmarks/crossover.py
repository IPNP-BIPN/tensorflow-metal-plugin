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
"""Where each op family starts beating the CPU, by size.

`benchmark.py` measures whole cases at one size, which answers "is this
backend worth it" and cannot answer "should this op be registered". A case
that loses at 4096x4096 may win at 512x512 or the other way round, and the
brief's rule about unregistering an op slower than the CPU needs the shape of
that curve rather than one point on it.

So this walks one family across sizes and reports the crossover: the smallest
size at which the GPU wins and keeps winning. Same paired timing as
benchmark.py, both devices alternated inside one process, because the failure
mode it was written to avoid applies here too.

Read the crossover, not the ratios. A family whose crossover is larger than
anything a real model contains is a candidate for the host; one that wins from
some modest size upward is doing its job, and the losing sizes below it are
kernel launch overhead rather than a reason to unregister anything.
"""

import argparse
import statistics
import sys
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import load_library

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from benchmark import paired  # pylint: disable=g-import-not-at-top,wrong-import-position

# Square sides, so the element counts run from a thousand to sixteen million.
SIDES = [32, 64, 128, 256, 512, 1024, 2048, 4096]


def families(side):
  """One callable per family at this size, and the bytes each one moves."""
  rng = np.random.default_rng(0)
  a = tf.constant(rng.standard_normal((side, side), dtype=np.float32))
  b = tf.constant(rng.standard_normal((side, side), dtype=np.float32))

  return {
      # The three cases issue #10 is about.
      "Mul": lambda: tf.raw_ops.Mul(x=a, y=b),
      "elementwise chain": lambda: tf.raw_ops.Mul(
          x=tf.raw_ops.AddV2(x=a, y=b), y=a),
      "ReduceSum": lambda: tf.raw_ops.Sum(input=a, axis=[0, 1]),
      # Controls: one the GPU is known to win, one that reads the same bytes
      # as Mul and does far more arithmetic with them.
      "Relu": lambda: tf.raw_ops.Relu(features=a),
      "MatMul": lambda: tf.raw_ops.MatMul(a=a, b=b),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--plugin", default=None)
  parser.add_argument("--runs", type=int, default=15,
                      help="paired repetitions per size")
  parser.add_argument("--repeats", type=int, default=3,
                      help="how many times to walk the whole grid")
  args = parser.parse_args()

  if args.plugin and not tf.config.list_physical_devices("GPU"):
    load_library.load_pluggable_device_library(args.plugin)
  if not tf.config.list_physical_devices("GPU"):
    print("no GPU device, nothing to compare")
    return 1
  tf.config.set_soft_device_placement(True)
  print(f"tensorflow {tf.__version__}, {args.repeats} walks of "
        f"{len(SIDES)} sizes, {args.runs} paired repetitions each\n")

  # ratios[name][side] is one median ratio per walk.
  ratios = {name: {side: [] for side in SIDES} for name in families(32)}
  for _ in range(args.repeats):
    for side in SIDES:
      for name, fn in families(side).items():
        _, _, per_pair = paired(fn, runs=args.runs)
        ratios[name][side].append(statistics.median(per_pair))

  header = "| family | " + " | ".join(f"{s}^2" for s in SIDES) + " | crossover |"
  print(header)
  print("| --- | " + " ---: |" * len(SIDES) + " --- |")
  for name in ratios:
    cells, crossover = [], None
    # The crossover is the smallest size that wins and is not followed by a
    # loss: a single size winning between two losses is noise, not a curve.
    for index, side in enumerate(SIDES):
      worst = min(ratios[name][side])
      cells.append(f"{statistics.median(ratios[name][side]):.2f}x")
      if crossover is None and worst > 1.0:
        if all(min(ratios[name][s]) > 1.0 for s in SIDES[index:]):
          crossover = side
    print(f"| {name} | " + " | ".join(cells) + " | "
          + (f"{crossover}^2" if crossover else "never") + " |")

  print("\nEach cell is the median across walks of the median paired ratio, "
        "CPU time over GPU time, so above 1.0 the GPU won. The crossover "
        "column is the smallest size whose slowest walk won and which is not "
        "followed by any size that lost.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
