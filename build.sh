#!/usr/bin/env bash
# Compile every project C++ program (deliverables/parser, tests, driver) against the llama.cpp build from setup.sh.
# Each <dir>/<name>.cpp becomes the executable <dir>/<name>.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L="$ROOT/llama.cpp"

shopt -s nullglob
for src in "$ROOT"/deliverables/{parser,tests,driver}/*.cpp; do
  g++ -std=c++17 -O2 "$src" -o "${src%.cpp}" \
      -I "$L/include" -I "$L/ggml/include" -I "$L/common" -I "$L/vendor" \
      -L "$L/build/bin" -lllama-common -lllama -lggml-base -Wl,-rpath,"$L/build/bin"
  echo "built ${src#"$ROOT"/}" | sed 's/\.cpp$//'
done
