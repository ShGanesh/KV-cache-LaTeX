# Structure-aware KV cache on llama.cpp: LaTeX papers

## Domain and setup

- **Domain:** LaTeX research papers. Structure: sections and environments (theorem, proof, figure). Questions usually target one section or one proof, so each part of the paper gets its own sequence.
- **Papers (6, arXiv sources):** `1406.2661`, `1701.07875`, `1706.03762`, `1806.07572`, `1904.09237`, `2006.11239`. Flattened with `flatten_tex.py`, parsed with pylatexenc (`deliverables/parser/latex_parser.py`), each trimmed to < 2,000 tokens.
- **Model:** Llama-3.2-1B-Instruct, Q4_K_M GGUF (762.8 MiB). KV cache: 32 KiB per token (16 layers × 8 KV heads × 64 × 2 bytes × K+V).
- **Hardware:** AMD Ryzen 7 7445HS w/ Radeon 740M Graphics, CPU only, llama.cpp built with `GGML_NATIVE=ON`.
- **Driver settings:** `kv_unified = true`, flash attention on, greedy decoding, prefill/question batches on all threads, single-token decodes on 1 thread, the question's group prefilled first. Merge parameters: MIN_TOKENS = 64, MAX_GROUPS = 6, MAX_REFS = 2 (`deliverables/schema/MERGE_RULES.md`).
- **Runs:** every test case was run 5 times; timings are medians.

## How the attention mask is built from sequence IDs

Traced in the llama.cpp checkout used for this run (commit `23f7aab`); line numbers refer to it.

**1. The batch.** Every token in a `llama_batch` carries a position and a *list* of sequence IDs. `examples/batched` adds the shared prompt once, tagged with every sequence, which is the pattern the driver uses for the paper's root:

```
examples/batched/batched.cpp:124   seq_ids[i] = i;
examples/batched/batched.cpp:129   batch.add(tokens_list[i], i, seq_ids, false);
```

**2. The cells.** Each KV cell stores one token's K/V, its position, and a bitset of the sequences that own it:

```
src/llama-kv-cells.h:38    using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;
```

**3. The mask.** `set_input_kq_mask_impl` fills one row per query token and one column per cell. A cell is visible (0) only if it is in use, belongs to the query's sequence, and is not in the query's future; everything else gets −∞, which softmax turns into a weight of exactly 0:

```
src/llama-kv-cache.cpp:1587  const T mask_drop = llama_cast<T>(-INFINITY);
src/llama-kv-cache.cpp:1599  for (uint32_t s = 0; s < n_stream; ++s) {
src/llama-kv-cache.cpp:1607  const llama_seq_id seq_id = ubatch->seq_id[i][0];
src/llama-kv-cache.cpp:1612  const llama_pos p1 = ubatch->pos[i];
src/llama-kv-cache.cpp:1661  if (cells.is_empty(j)) {
src/llama-kv-cache.cpp:1666  if (!cells.seq_has(j, seq_id)) {
src/llama-kv-cache.cpp:1670  p0 = cells.pos_get(j);
src/llama-kv-cache.cpp:1683  if (p0 > p1) {
src/llama-kv-cache.cpp:1711  data[idst + j] = mask_keep;
src/llama-kv-cache.cpp:1716  data[idst + j] = mask_drop;
```

Rule: a query at (sequence *s*, position *p*) sees cell *j* iff *j* is in use, *s* ∈ seqs(*j*), and pos(*j*) ≤ *p*. Note `seq_id[i][0]`: a *query* uses only its first sequence ID, while a *cell* may belong to many. So the root, stored once with all IDs, is visible to every group, and a group's own cells are visible only to that group.

**4. Eviction.** `llama_memory_seq_rm` clears one sequence's bit on the cells in a position range; a cell is freed only when no sequence owns it any more:

```
src/llama-kv-cells.h:244   bool seq_rm(uint32_t i, llama_seq_id seq_id) {
src/llama-kv-cells.h:250   seq[i].reset(seq_id);
src/llama-kv-cells.h:253   if (seq[i].none()) {
src/llama-kv-cells.h:254   pos[i] = -1;
```

So evicting a group frees its own cells, while the shared root and any cells shared into other groups stay.

**5. Rules llama.cpp enforces on a batch** (found by running it): within one decode, each sequence's positions must be gap-free and continue at `seq_pos_max + 1` (so positions are counted per sequence: every group restarts right after the root); a token tagged with several sequences needs `kv_unified = true`; `llama_memory_seq_cp` may create gaps, which is how a cited theorem is shared into the citing group.

## Tree statistics

| paper | tokens | nodes | leaves | depth | root | heading | paragraph | theorem | proof | float | groups | group tokens min / median / max | section tokens min / median / max | shared leaves | refs unresolved / forward | markup share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1406.2661 | 1997 | 19 | 13 | 3 | 243 | 5 | 0 | 3 | 1 | 3 | 5 | 67 / 157 / 1069 | 83 / 602 / 1069 | 1 | 0 / 0 | 47.1% |
| 1701.07875 | 1996 | 21 | 17 | 2 | 10 | 3 | 3 | 5 | 5 | 0 | 5 | 274 / 300 / 681 | 14 / 37 / 1935 | 0 | 8 / 0 | 56.1% |
| 1706.03762 | 1998 | 25 | 17 | 4 | 244 | 7 | 2 | 0 | 0 | 7 | 5 | 152 / 280 / 704 | 280 / 361.5 / 704 | 0 | 0 / 0 | 51.8% |
| 1806.07572 | 1976 | 17 | 12 | 3 | 376 | 4 | 1 | 6 | 0 | 0 | 6 | 79 / 254.5 / 444 | 523 / 800 / 1077 | 1 | 1 / 1 | 39.0% |
| 1904.09237 | 1999 | 23 | 17 | 2 | 221 | 5 | 3 | 5 | 0 | 3 | 6 | 75 / 239 / 776 | 15 / 266 / 988 | 1 | 0 / 0 | 55.0% |
| 2006.11239 | 1998 | 26 | 18 | 3 | 186 | 7 | 4 | 0 | 0 | 6 | 5 | 96 / 238 / 765 | 96 / 563 / 1003 | 0 | 1 / 0 | 58.5% |

Depth counts edges from the document node (document → section → subsection → leaf). The root column is the root leaf's token count (title + abstract). Groups are the merged subtrees, one sequence each. A token is *markup* if every non-space byte of it is LaTeX syntax: `\begin{...}`/`\end{...}`, macro names, the key arguments of `\label`/`\ref`/`\cite`/`\includegraphics`, `{ } $ & ^ _ ~`; whitespace-only tokens count as formatting markup.

Markup share by leaf type, all papers together:

| leaf type | tokens | markup tokens | share |
|---|---|---|---|
| root | 1280 | 197 | 15.4% |
| heading | 323 | 211 | 65.3% |
| paragraph | 290 | 114 | 39.3% |
| theorem | 4458 | 2309 | 51.8% |
| proof | 367 | 212 | 57.8% |
| float | 5246 | 3093 | 59.0% |
| **all** | 11964 | 6136 | **51.3%** |

## Test 0: two prompts in two sequences vs. separate runs

`deliverables/tests/two_seq` (flash attention on, `kv_unified = true`, 32 greedy tokens per prompt):

```
n_gen = 32, kv_unified = true, flash_attn = on

prompt 0 (seq 0 in the joint run)
  joint   : 
\begin{align*}
\int_{0}^{1} f_{\theta}(x) dx &= \frac{1}{2} \
  separate: 
\begin{align*}
\int_{0}^{1} f_{\theta}(x) dx &= \frac{1}{2} \
  tokens identical: YES   max |logit diff| over 32 steps: 0

prompt 1 (seq 1 in the joint run)
  joint   :  the critic learned to predict the mean of the 10th percentile of the data. The mean of the 10th percentile is 0.5. The
  separate:  the critic learned to predict the mean of the 10th percentile of the data. The mean of the 10th percentile is 0.5. The
  tokens identical: YES   max |logit diff| over 32 steps: 0

time: joint 2533 ms | separate 1834 + 1791 = 3625 ms
RESULT: PASS
```

## Tests (a) and (b)

(a) change subtree B (one leaf rewritten, only its group recomputed), ask about subtree A: the answer must be token-identical. (b) evict an unrelated subtree C with `llama_memory_seq_rm(seq, p0, p1)`: the answer must be token-identical and the KV memory in use (`llama_state_get_size`) must drop by at least |C| × 32 KiB. A, B and C are picked by a fixed rule (`deliverables/tests/run_tests.py`): A is the group containing the question's anchor text, B the last group sharing no cells with A (one of its leaves is rewritten), C the first other group sharing no cells with A and cited by no group. A test passes only if it held in all 5 runs.

| paper | A | question | answer | (a) | B (recomputed tokens) | (b) | C (evicted tokens) | KV before → after (KiB) | dropped / expected (KiB) |
|---|---|---|---|---|---|---|---|---|---|
| 1406.2661 | 1 | In Algorithm 1, what value of k was used in the experiments? | `1` | PASS | seq 5 (26) | PASS | seq 2 (378) | 62109.8 → 50009.3 | 12100.4 / 12096.0 |
| 1701.07875 | 1 | What is the name of the example labelled exa::lines? | `Wasserstein GAN` | PASS | seq 5 (111) | PASS | seq 2 (274) | 57845.8 → 49074.6 | 8771.2 / 8768.0 |
| 1706.03762 | 4 | What BLEU score does the Transformer (big) achieve on English-to-German? | `\textbf{41.8` | PASS | seq 5 (308) | PASS | seq 1 (152) | 60699.4 → 55833.6 | 4865.8 / 4864.0 |
| 1806.07572 | 1 | What is the input distribution assumed to be? | `The input distribution is assumed to be the empirical distribution on a finite dataset $x_1, ..., x_N$` | PASS | seq 6 (25) | PASS | seq 2 (444) | 56350.8 → 42137.6 | 14213.2 / 14208.0 |
| 1904.09237 | 1 | What is the caption of Algorithm 1? | `$\boxed{Generic Adaptive Method Setup}$` | PASS | seq 6 (261) | PASS | seq 2 (163) | 63198.2 → 57980.3 | 5217.9 / 5216.0 |
| 2006.11239 | 3 | What is the caption of Algorithm 1? | `$\text{dog}` | PASS | seq 5 (415) | PASS | seq 1 (150) | 52759.4 → 47957.6 | 4801.8 / 4800.0 |

**12 / 12 tests pass.**

Answer quality is separate from the tests (the tests check that answers do not change). Whether the 1B model's answer contains the expected fact:

| paper | expected | answer | correct |
|---|---|---|---|
| 1406.2661 | 1 | `1` | yes |
| 1701.07875 | Learning parallel lines | `Wasserstein GAN` | no |
| 1706.03762 | 28.4 | `\textbf{41.8` | no |
| 1806.07572 | empirical | `The input distribution is assumed to be the empirical distribution on a finite dataset $x_1, ..., x_N$` | yes |
| 1904.09237 | Generic Adaptive Method Setup | `$\boxed{Generic Adaptive Method Setup}$` | yes |
| 2006.11239 | Training | `$\text{dog}` | no |

## CPU run times (medians)

| paper | tokens | prefill (ms) | prefill (tok/s) | answer (ms) | recompute one group (ms) | full prefill / recompute |
|---|---|---|---|---|---|---|
| 1406.2661 | 1997 | 16400 | 122 | 547 | 307 | 53.4× |
| 1701.07875 | 1996 | 17812 | 112 | 583 | 947 | 18.8× |
| 1706.03762 | 1998 | 17393 | 115 | 975 | 2942 | 5.9× |
| 1806.07572 | 1976 | 15147 | 130 | 2004 | 283 | 53.5× |
| 1904.09237 | 1999 | 17152 | 117 | 780 | 1919 | 8.9× |
| 2006.11239 | 1998 | 21154 | 94 | 1266 | 4811 | 4.4× |

Prefill is the whole paper (root + every group). "Recompute one group" is test (a)'s cost: `seq_rm` of group B plus its re-decode. Without sequence IDs, editing B would mean re-prefilling everything after it.

## Findings and limitations

- **Isolation is exact at the mask level.** A query sees a cell only if the cell carries the query's sequence ID; others get −∞, i.e. weight exactly 0.
- **But llama.cpp's CPU arithmetic depends on the cache layout.** When freed cells are reused, the question and answer land in other slots, and logits move by ~0.5. Greedy answers then flip wherever the top two tokens are nearly tied (e.g. 28.4 vs 41.8 with a margin of 0.02). Refilling the freed slots made the logits bit-identical again, which isolates the cause. The driver avoids it by prefilling the question's group first (every hole then lies after it) and by decoding single tokens on one thread. This fix was verified empirically (logit differences 0); the explanation, that multi-row attention depends only on the order of the attended cells while multi-threaded single-token attention also depends on their slots, is our reading of the measurements. With flash attention off, results also depend on batch shape: `deliverables/tests/two_seq ... 1 off` diverges after 9–31 greedy tokens.
- **Trimming** keeps theorem/proof/float leaves first, so references to trimmed parts become unresolved (see the table).
- **References:** only backward `\ref`-family citations are modelled (the cited leaf's cells are shared with `seq_cp`, at most MAX_REFS per group). Forward references, plain-text references ("see Theorem 1") and `\cite` are not.
- **Answers** come from a 1B model with a plain `Question: … Answer:` prompt; several are wrong, which does not affect the isolation tests but limits the usefulness of the answers themselves.
