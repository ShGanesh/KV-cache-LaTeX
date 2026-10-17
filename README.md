# Structure-aware KV cache on llama.cpp with respect to Research papers in LaTeX

## The problem

When an LLM reads a document, it stores a key and a value vector for every token (the **KV cache**), and every later token attends to all of them. The document becomes one long stream:

- **Edits are expensive.** Changing one section invalidates the cache for everything after it.
- **Memory can't be cut cleanly.** You can't drop one section without affecting the rest.
- **Attention isn't focused.** A question about one proof also attends to every unrelated section.

llama.cpp lets every token carry **sequence IDs**, and its attention mask only lets a token see cached tokens of its own sequence. If each part of a document's tree (section, theorem + proof, figure) gets its own sequence, and the title and abstract are shared by all of them, then:

- a question about one part sees only that part plus the shared header;
- an unrelated part can be **changed** without changing the answer;
- an unrelated part can be **evicted**, freeing exactly its memory.

This repository implements that for **LaTeX research papers** and tests it on six arXiv papers, CPU only.

## Assignment Instructions

1. Build the CPU version of llama.cpp and run Llama-3.2-1B Q4_K_M. Read `llama.h`, trace the `batched` example and how the attention mask is built from sequence IDs.
2. Run two prompts in two sequences and match their outputs against separate runs.
3. Define a JSON schema with the fields `tokens`, `positions`, `seq_ids`, `node_type`, `parent`, plus merge rules for small subtrees (llama.cpp allows only a limited number of sequences).
4. Write a parser that turns sample documents of the chosen domain into this schema, each under 2,000 tokens. Domain here: LaTeX papers (sections; theorem, proof and figure environments), parsed with pylatexenc.
5. Build the shared C++ driver from the `batched` template: load the JSON, prefill with structured sequence IDs, answer a question, evict a subtree by position range.
6. Tests: **(a)** change subtree B, ask about subtree A, and check that the output is unchanged; **(b)** remove an unrelated subtree, check that the answer holds and KV memory drops.
7. Report: domain, tree statistics (depth, subtree sizes, share of tokens that are pure markup or syntax), the test results, and CPU run times.

## How to Run

Requirements: Linux, conda, `g++` (C++17), `git`, `curl`, `wget`, `file`, about 2 GB of disk and 8 GB of RAM.

Everything, in order (setup, build, parse, all tests with 5 runs per paper for median timings, report):
```bash
RUNS=5 ./run.sh > run.log 2>&1
```
`run.sh` runs these steps; each can also be run on its own:

| Step | Command | Purpose |
|---|---|---|
| 1 | `./setup.sh` | conda env `LDLS` (Python 3.11, cmake, pylatexenc); arXiv LaTeX sources into `data/raw/`, flattened into `data/flat/` (`download_latex.sh`, `flatten_tex.py`); clones and builds llama.cpp for CPU; downloads the model (in the background) and runs a smoke test. Re-running skips finished work. |
| 2 | `./build.sh` | compiles the C++ programs in `deliverables/` against the llama.cpp build |
| 3 | `python -I deliverables/parser/latex_parser.py data/flat/*.tex -o data/json` | parses the papers into the schema, one JSON per paper, each < 2,000 tokens |
| 4 | `deliverables/tests/two_seq MODEL 32 1 on` | two prompts in two sequences vs. separate runs |
| 5 | `python -I deliverables/tests/run_tests.py --runs 5` | tests (a) and (b) on every paper, 5 runs each |
| 6 | `python -I deliverables/report/make_report.py` | writes `deliverables/report/report.md` |

`MODEL` is `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`. Python steps run inside the conda env (`conda activate LDLS`).

One driver call, for exploring:
```bash
deliverables/driver/ldls_driver -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --doc data/json/1904.09237.json --seq 1 -p "What is the caption of Algorithm 1?" \
    --replace 18 "new text for node 18" --evict 3
```

## Adding other arXiv papers

1. **Pick a paper with LaTeX source.** On arXiv, the paper's "TeX Source" link must exist (some papers only offer a PDF). Theorem-like environments are found through the paper's own `\newtheorem` declarations, so any naming works.
2. **Add its ID to the list** in `download_latex.sh`, inside the parentheses of `DEFAULT_IDS`, one ID per line:
   ```bash
   DEFAULT_IDS=(
     1406.2661   # GAN
     ...
     2006.11239  # DDPM
     2106.09685  # the new paper
   )
   ```
   (For a one-off download without editing the list: `./download_latex.sh 2106.09685`.)
3. **Download, flatten and parse:** `./setup.sh` fetches only the papers that are missing and writes `data/flat/<id>.tex`. The parser then takes every file in `data/flat/`:
   ```bash
   python -I deliverables/parser/latex_parser.py data/flat/*.tex -o data/json
   ```
   The report includes every paper in `data/json/` automatically.
4. **Tests (a) and (b) run on it automatically.** `run_tests.py` tests every paper in `data/json/`. It picks group A, a leaf in group B to rewrite, and group C to evict by a fixed rule (A, B, C share no cells; C is cited by no group). Papers with fewer than 3 such groups are skipped. Without a case, the question is `default_question` from `deliverables/tests/cases.json`. To ask about a specific fact, add a case:
   ```json
   {"doc": "2106.09685", "question": "…", "anchor": "a snippet of the paper's LaTeX that contains the answer", "expected": "…"}
   ```
   - `anchor` selects group A: the first group whose text contains it. If no group contains it, the paper falls back to group 1 and the default question.
   - `expected` is optional. It's reported (was the answer right?), not tested.

## Deliverables (`deliverables/`)

| Path | Description |
|---|---|
| `schema/schema.json` | the JSON format: one node per tree node with `node_type`, `parent`, `tokens`, `positions` (per sequence), `seq_ids` (home sequence first, then sequences that share it) |
| `schema/MERGE_RULES.md` | how a paper becomes that format: leaf types, trimming to < 2,000 tokens, glue / minimum size / budget merge passes, shared cited leaves, positions, and the driver's decode order |
| `parser/latex_parser.py` | the LaTeX parser (pylatexenc): tree, tokenization, markup count, trimming, merge rules, sequence IDs and positions |
| `parser/tokenize.cpp` | the parser's tokenizer: loads only the model's vocabulary, once, and also returns each token's byte length |
| `driver/ldls_driver.cpp` | the driver, a copy of llama.cpp's `examples/batched/batched.cpp` with the changes for structured sequence IDs: JSON prefill, question answering, `--replace` (change a subtree, recompute only its group), `--evict` (remove a group by position range) |
| `driver/median_runs.py` | runs the driver several times and reports median timings |
| `tests/two_seq.cpp` | two prompts in two sequences vs. separate runs (tokens and full logits compared) |
| `tests/cases.json`, `tests/run_tests.py` | tests (a) and (b) on every parsed paper: the runner picks the question group A, a leaf to change in group B and a group C to evict by a fixed rule; `cases.json` holds optional questions (with an anchor text locating A and the expected answer) |
| `report/report.md` | the report: setup, how the mask is built from sequence IDs (traced in the llama.cpp source), tree statistics and markup share, all test results, median CPU timings, findings and limitations |
| `report/make_report.py`, `report/results/`, `report/two_seq.txt` | the report generator and the raw results it reads |

Results in short: 12 / 12 tests pass on all six papers in all runs, eviction frees exactly the evicted group's 32 KiB per token, and about half of all tokens (51%) are LaTeX markup. Details are in `deliverables/report/report.md`.

## Repository layout

```
setup.sh, run.sh, build.sh          setup, everything, compile
download_latex.sh, flatten_tex.py   arXiv sources -> one flattened .tex per paper
environment.yml, requirements.txt   conda env (Python 3.11, cmake, pylatexenc)
deliverables/                       schema, parser, driver, tests, report
```
Not committed, created when you run it: `models/`, `llama.cpp/`, `data/raw/` and `data/flat/` by `setup.sh` (the papers are downloaded from arXiv on every new setup), and `data/json/` by the parser (step 3).
