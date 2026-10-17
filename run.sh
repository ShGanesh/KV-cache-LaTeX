#!/usr/bin/env bash
# Everything, one step after the other:
#   1. setup.sh       conda env, LaTeX sources (+ flatten), llama.cpp CPU build, model
#   2. build.sh       deliverables/{parser/tokenize, tests/two_seq, driver/ldls_driver}
#   3. parser         data/flat/*.tex -> data/json/*.json
#   4. two_seq        two prompts in two sequences vs. separate runs
#   5. run_tests.py   tests (a) and (b) on every paper, RUNS runs each (median timings)
#   6. make_report.py deliverables/report/report.md
# Usage: RUNS=5 ./run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUNS="${RUNS:-5}"
MODEL="$ROOT/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

step() { printf '\n==> %s\n' "$*"; }

step "1/6 setup"
bash "$ROOT/setup.sh"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate LDLS; set -u   # conda's activate hooks trip on `set -u`

step "2/6 build"
bash "$ROOT/build.sh"

step "3/6 parser"
python -I deliverables/parser/latex_parser.py data/flat/*.tex -o data/json

step "4/6 two prompts in two sequences vs. separate runs"
mkdir -p deliverables/report
deliverables/tests/two_seq "$MODEL" 32 1 on | tee deliverables/report/two_seq.txt

step "5/6 tests (a) and (b), $RUNS runs per paper"
status=0
python -I deliverables/tests/run_tests.py --runs "$RUNS" || status=$?

step "6/6 report"
python -I deliverables/report/make_report.py
exit "$status"
