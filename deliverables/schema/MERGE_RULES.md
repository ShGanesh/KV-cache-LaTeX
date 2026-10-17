# Merge rules

Input: the parsed tree of one paper. Output: `seq_ids` and `positions` for every leaf (`schema.json`).
Parameters: `MIN_TOKENS = 64`, `MAX_GROUPS = 6`, `MAX_REFS = 2`.

## Leaves

- **Containers:** the document, and one node per `\section`, `\subsection`, `\subsubsection` (starred too). They hold no text.
- **Leaves**, in document order:
  - `root`: `\title{...}` + the `abstract` environment. Exactly one; it is never merged.
  - `heading`: the sectioning command line itself.
  - `theorem`: an environment declared in the paper's own preamble with `\newtheorem{env}...`.
  - `proof`: the `proof` environment.
  - `float`: `figure`, `table`, `algorithm` (and starred forms).
  - `paragraph`: all other text, split at blank lines.
- The preamble, `\maketitle`, and the bibliography are not tokenized.
- A leaf's text is its LaTeX source, verbatim. It is tokenized on its own, so leaf boundaries are token boundaries.

## Trimming to under 2,000 tokens

Before merging: keep the root; then go through the theorem, proof and float leaves in document order, then the paragraph leaves in document order. Keep a leaf if the paper stays under 2,000 tokens with it and with any of its section headings not yet kept. Kept leaves stay in document order; containers left empty are removed.

## Cut strength

Between two neighbouring leaves (or groups): `2` if they are in different sections, `1` if they are in the same section but a different innermost container, `0` if they share the innermost container.

## Passes

Every pass only removes cuts, so a group is always a contiguous run of leaves.

1. **Glue** (in this order):
   1. A `proof` joins the previous atom if that atom is in the same innermost container.
   2. A `float` joins the previous non-heading atom in the same innermost container; otherwise the next atom in that container; otherwise it stays alone.
   3. A `heading` joins the next atom. Headings in a row cascade (`\section` + `\subsection` + text). A heading with no next atom joins the previous one.
2. **Minimum size:** while some atom has fewer than `MIN_TOKENS` tokens, take the first such atom and merge it with the first of these that exists: the previous atom in the same section, the next atom in the same section, the previous atom, the next atom.
3. **Budget:** while there are more than `MAX_GROUPS` groups, erase the cut with the smallest key (strength, merged size in tokens, index).
4. **References:** for each `\ref`, `\autoref`, `\eqref`, `\cref`, `\Cref` in group *g* whose `\label` is in leaf *L* of an earlier group *h*, append *g* to `L.seq_ids`. At most `MAX_REFS` distinct leaves per group, in order of first citation. References inside the same group need nothing. Forward references and missing labels are ignored and counted. A label defined twice resolves to its first definition.

## Sequence IDs and positions

- Groups are numbered `1..G` in document order. The root gets `seq_ids = [0, 1, ..., G]`; seq 0 holds only the root.
- The root takes positions `0..R-1`.
- In group *g*, the leaves continue at `p0 = max(R, last position of any leaf shared into g + 1)`, consecutively in document order. A shared leaf keeps its original positions.

## Decode order (driver)

1. Decode the root, tagged with all of its `seq_ids`.
2. Decode the groups one by one: first the question's group (preceded by the groups it cites), then the others in document order. For each group *g*: `llama_memory_seq_cp(home, g, L.p0, L.p1)` for each leaf *L* shared into *g*, then decode *g*'s own leaves tagged `[g]`. A cited group is always decoded before the group citing it.
3. Context: `n_seq_max = G + 1` (seqs 0..G), `kv_unified = true`, `flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED`, single-token decodes on 1 thread (`n_threads = 1`, batches on all threads).

Why steps 2 and 3 fix the order and the threads: the mask isolates sequences exactly, but llama.cpp's CPU arithmetic depends on which cache slots the question and answer occupy. With the question's group first, every freed or rewritten cell lies after it; with one thread for single-token decodes, the remaining slot dependence disappeared in our measurements (our reading: multi-threaded single-token attention splits the cache between threads by slot). Together they kept answers bit-identical when another group was replaced or evicted.

Evicting group *g* is `llama_memory_seq_rm(mem, g, p0, p1)` over *g*'s own positions. The root and any leaves *g* received are kept, because other seqs still hold them.
