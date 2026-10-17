#!/usr/bin/env bash
# One-shot setup, up to and including the model:
#   0. model download starts in the background
#   1. conda env "LDLS"   2. LaTeX sources (+ flatten)   3. llama.cpp CPU build   4. wait for model + smoke test
# Re-runnable: every step skips work that is already done.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_NAME=LDLS
LLAMA_DIR="$ROOT/llama.cpp"
MODEL_DIR="$ROOT/models"
MODEL_FILE=Llama-3.2-1B-Instruct-Q4_K_M.gguf
MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/$MODEL_FILE"
JOBS="$(nproc)"

step() { printf '\n==> %s\n' "$*"; }

# ---------------------------------------------------------------- 0. model download (background)
# This step is the slowest one, so we start it now in the background while steps 1-3 run; step 4 waits for it.
mkdir -p "$MODEL_DIR"
MODEL_LOG="$MODEL_DIR/download.log"
DL_PID=""
if [[ -s "$MODEL_DIR/$MODEL_FILE" ]]; then
  echo "[model] already present"
elif pgrep -f "wget .*$MODEL_FILE" > /dev/null; then
  echo "[model] another download of $MODEL_FILE is already running; not starting a second one" >&2
  echo "        re-run setup.sh once it has finished" >&2
  exit 1
else
  echo "[model] downloading in background (log: $MODEL_LOG)"
  # -c resumes a partial .part file; rename only on success so a .gguf is always complete
  ( wget -c -q --show-progress --progress=dot:giga -O "$MODEL_DIR/$MODEL_FILE.part" "$MODEL_URL" && mv "$MODEL_DIR/$MODEL_FILE.part" "$MODEL_DIR/$MODEL_FILE" ) > "$MODEL_LOG" 2>&1 &
  DL_PID=$!
  # don't leave the download orphaned if a later step fails or you press Ctrl-C.
  # $DL_PID is the subshell; killing only it would leave wget running, so kill its children first.
  trap 'if [[ -n "$DL_PID" ]]; then pkill -P "$DL_PID" 2>/dev/null; kill "$DL_PID" 2>/dev/null; fi; true' EXIT
fi

# ---------------------------------------------------------------- 1. conda env
step "1/4 conda env '$ENV_NAME'"
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f environment.yml --prune
else
  conda env create -y -f environment.yml
fi
set +u; conda activate "$ENV_NAME"; set -u   # conda's activate hooks trip on `set -u`

# Just in case: verify the env really has its tools (e.g. after an interrupted create); install what's missing
if [[ "$(command -v cmake || true)" != "$CONDA_PREFIX"/* ]]; then
  echo "[env] cmake missing from $ENV_NAME; installing"
  conda install -y -n "$ENV_NAME" -c conda-forge cmake
  hash -r
fi
if ! python -c "import pylatexenc" 2>/dev/null; then
  echo "[env] pylatexenc missing from $ENV_NAME; installing"
  python -m pip install -r requirements.txt
fi
echo "[env] $(command -v cmake): $(cmake --version | head -1)"
echo "[env] $(command -v python): pylatexenc $(python -c 'import pylatexenc; print(pylatexenc.__version__)')"

# ---------------------------------------------------------------- 2. LaTeX sources
step "2/4 LaTeX sources"
PYTHON="$(command -v python)" bash "$ROOT/download_latex.sh"

# ---------------------------------------------------------------- 3. llama.cpp (CPU)
step "3/4 llama.cpp CPU build"
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=ON \
      -DLLAMA_OPENSSL=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_SERVER=OFF \
      -DGGML_CCACHE=OFF
cmake --build "$LLAMA_DIR/build" -j "$JOBS" \
      --target llama-batched llama-simple # llama-tokenize

# ---------------------------------------------------------------- 4. model
step "4/4 model ($MODEL_FILE)"
if [[ -n "$DL_PID" ]]; then
  echo "[model] waiting for background download..."
  if ! wait "$DL_PID"; then
    DL_PID=""
    echo "[model] download failed; see $MODEL_LOG. Re-run setup.sh to resume." >&2
    exit 1
  fi
  DL_PID=""
fi
[[ "$(head -c4 "$MODEL_DIR/$MODEL_FILE")" == GGUF ]] || { echo "[model] not a GGUF file" >&2; exit 1; }
ls -lh "$MODEL_DIR/$MODEL_FILE"

step "smoke test"
"$LLAMA_DIR/build/bin/llama-simple" -m "$MODEL_DIR/$MODEL_FILE" -n 32 \
  "A theorem in mathematics is"

step "done. Next: ./build.sh"
