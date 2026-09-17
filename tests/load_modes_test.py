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
"""Checks the four ways this plugin can be asked to load, or not to.

A subprocess per case, for two reasons. The decision is read once and cached
for the life of the process, so a second case in the same interpreter would
see the first one's answer. And the failure being guarded against is the
process dying: a check that runs in-process cannot tell "declined" from
"aborted", because both leave nothing behind to assert on.

The mismatch case needs a library built against a different TensorFlow, which
is produced here by rebuilding with TF_SUPPORTED_VERSION temporarily changed.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PIN = ROOT / "TF_SUPPORTED_VERSION"

# Loads the plugin and prints what devices resulted, or that it raised.
PROBE = """
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.python.framework import load_library
try:
  load_library.load_pluggable_device_library(sys.argv[1])
except Exception as error:
  print("RAISED", type(error).__name__)
  raise SystemExit(0)
print("DEVICES", len(tf.config.list_physical_devices("GPU")))
"""

_failures = 0


def check(name, condition, detail=""):
  global _failures
  print(f"  {name:44s} {'ok' if condition else 'FAILED'}"
        f"{('  ' + detail) if detail else ''}")
  if not condition:
    _failures += 1


def load(library, **environment):
  """Loads `library` in a fresh interpreter, returns (devices, output)."""
  env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
  # Whatever the caller running these tests happens to have set would decide
  # the answer instead of the case under test, so both switches start clear.
  for switch in ("TF_DISABLE_METAL", "TF_METAL_SKIP_VERSION_CHECK"):
    env.pop(switch, None)
  env.update(environment)
  result = subprocess.run([sys.executable, "-c", PROBE, str(library)],
                          capture_output=True, text=True, env=env)
  output = result.stdout + result.stderr
  if result.returncode != 0:
    return None, output
  for line in result.stdout.splitlines():
    if line.startswith("DEVICES"):
      return int(line.split()[1]), output
  return None, output


def build_with_pin(version, destination, python):
  """Builds the plugin with TF_SUPPORTED_VERSION set to `version`."""
  original = PIN.read_text()
  try:
    PIN.write_text(version + "\n")
    subprocess.run(["make", f"PYTHON={python}", "-j8"], cwd=ROOT, check=True,
                   capture_output=True)
    shutil.copy(ROOT / "build" / "libmetal_plugin.dylib", destination)
  finally:
    PIN.write_text(original)
    # Leave the tree holding a library that matches the pin again, so a
    # failure here does not quietly poison whatever runs next.
    subprocess.run(["make", f"PYTHON={python}", "-j8"], cwd=ROOT, check=True,
                   capture_output=True)


def main():
  python = sys.executable
  plugin = os.environ.get(
      "METAL_PLUGIN", str(ROOT / "build" / "libmetal_plugin.dylib"))
  pinned = PIN.read_text().strip()
  print(f"pinned to {pinned}")

  print("\nthe matching build:")
  devices, output = load(plugin)
  check("it loads", devices is not None, "" if devices is not None else output)
  check("it offers one GPU device", devices == 1, f"got {devices}")

  print("\nTF_DISABLE_METAL:")
  devices, output = load(plugin, TF_DISABLE_METAL="1")
  # The process surviving is the whole point. Returning a non-OK status from
  # SE_InitPlugin aborts it, which is what this used to do.
  check("the process survives", devices is not None,
        "" if devices is not None else output)
  check("it offers no device", devices == 0, f"got {devices}")
  check("it says why", "TF_DISABLE_METAL" in output)

  print("\na build against a different TensorFlow:")
  wrong = ".".join(["2", str(int(pinned.split(".")[1]) - 1), "0"])
  with tempfile.TemporaryDirectory() as scratch:
    mismatched = pathlib.Path(scratch) / "mismatched.dylib"
    print(f"  (rebuilding with the pin set to {wrong}, this takes a while)")
    build_with_pin(wrong, mismatched, python)

    devices, output = load(mismatched)
    check("the process survives", devices is not None,
          "" if devices is not None else output)
    check("it offers no device", devices == 0, f"got {devices}")
    check("it names both versions",
          wrong in output and pinned in output)

    devices, _ = load(mismatched, TF_METAL_SKIP_VERSION_CHECK="1")
    check("TF_METAL_SKIP_VERSION_CHECK forces it through", devices == 1,
          f"got {devices}")

  print()
  if _failures:
    print(f"{_failures} failed")
    return 1
  print("all checks passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
