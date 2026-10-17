#!/usr/bin/env python3
"""Write deliverables/report/report.md from data/json/*.json, two_seq.txt and results.json (both in deliverables/report/).

Usage:
    python -I deliverables/report/make_report.py
"""
import json
import platform
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repository root
REPORT = Path(__file__).resolve().parent
LEAF_TYPES = ["root", "heading", "paragraph", "theorem", "proof", "float"]


def cpu_name():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def excerpt(rel, start, patterns):
    """Source lines of llama.cpp/<rel> matching `patterns` in order, after the first line containing `start`."""
    lines = (ROOT / "llama.cpp" / rel).read_text().splitlines()
    k = next((i for i, l in enumerate(lines) if start in l), 0)
    out = []
    for pat in patterns:
        hit = next((i for i in range(k, len(lines)) if pat in lines[i]), None)
        if hit is None:                      # a newer llama.cpp moved or changed this line
            out.append(f"{rel}: (not found in this llama.cpp version) {pat}")
            continue
        out.append(f"{rel}:{hit + 1:<5} {lines[hit].strip()}")
        k = hit + 1
    return "\n".join(out)


def llama_commit():
    head = (ROOT / "llama.cpp" / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref:"):
        head = (ROOT / "llama.cpp" / ".git" / head.split()[1]).read_text().strip()
    return head[:7]


def tree_stats(doc):
    nodes = doc["nodes"]
    depth = {}
    for n in nodes:                                      # pre-order: parents come first
        depth[n["id"]] = 0 if n["parent"] is None else depth[n["parent"]] + 1
    leaves = [n for n in nodes if n["tokens"]]
    n_tok = sum(len(n["tokens"]) for n in leaves)
    n_markup = sum(n.get("n_markup", 0) for n in leaves)
    groups = {}
    for n in leaves:
        if n["node_type"] != "root":
            groups[n["seq_ids"][0]] = groups.get(n["seq_ids"][0], 0) + len(n["tokens"])
    sections = {}                                        # tokens per top-level section subtree
    for n in leaves:
        p, top = n["id"], None
        while nodes[p]["parent"] is not None:
            if nodes[p]["node_type"] == "section":
                top = p
            p = nodes[p]["parent"]
        if top is not None:
            sections[top] = sections.get(top, 0) + len(n["tokens"])
    by_type = {t: [n for n in leaves if n["node_type"] == t] for t in LEAF_TYPES}
    return {
        "tokens": n_tok, "nodes": len(nodes), "leaves": len(leaves), "depth": max(depth.values()),
        "count": {t: len(v) for t, v in by_type.items()},
        "markup_share": n_markup / n_tok,
        "markup_by_type": {t: (sum(x.get("n_markup", 0) for x in v), sum(len(x["tokens"]) for x in v)) for t, v in by_type.items()},
        "groups": sorted(groups.values()), "sections": sorted(sections.values()),
        "root": len(next(n for n in leaves if n["node_type"] == "root")["tokens"]),
        "shared": sum(len(n["seq_ids"]) - 1 for n in leaves if n["node_type"] != "root"),
        "refs": doc.get("ref_stats", {}),
    }


def mmm(xs):
    return f"{min(xs)} / {statistics.median(xs):g} / {max(xs)}" if xs else "–"


def main():
    docs = [json.loads(p.read_text()) for p in sorted((ROOT / "data" / "json").glob("*.json"))]
    stats = {d["doc_id"]: tree_stats(d) for d in docs}
    data = json.loads((REPORT / "results.json").read_text())
    results, skipped = data["results"], data["skipped"]
    two_seq = (REPORT / "two_seq.txt").read_text()
    params = docs[0]["params"]
    runs = results[0]["runs"] if results else 0

    out = []
    w = out.append
    w("# Structure-aware KV cache on llama.cpp: LaTeX papers\n")
    w("## Domain and setup\n")
    w("- **Domain:** LaTeX research papers. Structure: sections and environments (theorem, proof, figure). "
      "Questions usually target one section or one proof, so each part of the paper gets its own sequence.")
    w(f"- **Papers ({len(docs)}, arXiv sources):** " + ", ".join(f"`{d['doc_id']}`" for d in docs) +
      ". Flattened with `flatten_tex.py`, parsed with pylatexenc (`deliverables/parser/latex_parser.py`), each trimmed to < 2,000 tokens.")
    w("- **Model:** Llama-3.2-1B-Instruct, Q4_K_M GGUF (762.8 MiB). KV cache: 32 KiB per token (16 layers × 8 KV heads × 64 × 2 bytes × K+V).")
    w(f"- **Hardware:** {cpu_name()}, CPU only, llama.cpp built with `GGML_NATIVE=ON`.")
    w(f"- **Driver settings:** `kv_unified = true`, flash attention on, greedy decoding, prefill/question batches on all threads, "
      f"single-token decodes on 1 thread, the question's group prefilled first. "
      f"Merge parameters: MIN_TOKENS = {params['MIN_TOKENS']}, MAX_GROUPS = {params['MAX_GROUPS']}, MAX_REFS = {params['MAX_REFS']} "
      f"(`deliverables/schema/MERGE_RULES.md`).")
    w(f"- **Runs:** every test case was run {runs} times; timings are medians.\n")

    w("## How the attention mask is built from sequence IDs\n")
    w(f"Traced in the llama.cpp checkout used for this run (commit `{llama_commit()}`); line numbers refer to it.\n")
    w("**1. The batch.** Every token in a `llama_batch` carries a position and a *list* of sequence IDs. "
      "`examples/batched` adds the shared prompt once, tagged with every sequence, which is the pattern the driver uses for the paper's root:\n")
    w("```\n" + excerpt("examples/batched/batched.cpp", "std::vector<llama_seq_id> seq_ids(n_parallel, 0);",
                         ["seq_ids[i] = i;", "batch.add(tokens_list[i], i, seq_ids, false);"]) + "\n```\n")
    w("**2. The cells.** Each KV cell stores one token's K/V, its position, and a bitset of the sequences that own it:\n")
    w("```\n" + excerpt("src/llama-kv-cells.h", "using seq_set_t", ["using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;"]) + "\n```\n")
    w("**3. The mask.** `set_input_kq_mask_impl` fills one row per query token and one column per cell. A cell is visible (0) only if it is "
      "in use, belongs to the query's sequence, and is not in the query's future; everything else gets −∞, which softmax turns into a weight of exactly 0:\n")
    w("```\n" + excerpt("src/llama-kv-cache.cpp", "static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {",
                         ["const T mask_drop = llama_cast<T>(-INFINITY);", "for (uint32_t s = 0; s < n_stream; ++s) {",
                          "const llama_seq_id seq_id = ubatch->seq_id[i][0];", "const llama_pos p1 = ubatch->pos[i];", "if (cells.is_empty(j)) {", "if (!cells.seq_has(j, seq_id)) {",
                          "p0 = cells.pos_get(j);", "if (p0 > p1) {", "data[idst + j] = mask_keep;", "data[idst + j] = mask_drop;"]) + "\n```\n")
    w("Rule: a query at (sequence *s*, position *p*) sees cell *j* iff *j* is in use, *s* ∈ seqs(*j*), and pos(*j*) ≤ *p*. "
      "Note `seq_id[i][0]`: a *query* uses only its first sequence ID, while a *cell* may belong to many. "
      "So the root, stored once with all IDs, is visible to every group, and a group's own cells are visible only to that group.\n")
    w("**4. Eviction.** `llama_memory_seq_rm` clears one sequence's bit on the cells in a position range; a cell is freed only when no sequence owns it any more:\n")
    w("```\n" + excerpt("src/llama-kv-cells.h", "bool seq_rm(uint32_t i, llama_seq_id seq_id) {",
                         ["bool seq_rm(uint32_t i, llama_seq_id seq_id) {", "seq[i].reset(seq_id);", "if (seq[i].none()) {", "pos[i] = -1;"]) + "\n```\n")
    w("So evicting a group frees its own cells, while the shared root and any cells shared into other groups stay.\n")
    w("**5. Rules llama.cpp enforces on a batch** (found by running it): within one decode, each sequence's positions must be gap-free and continue at "
      "`seq_pos_max + 1` (so positions are counted per sequence: every group restarts right after the root); a token tagged with several sequences "
      "needs `kv_unified = true`; `llama_memory_seq_cp` may create gaps, which is how a cited theorem is shared into the citing group.\n")
    w("## Tree statistics\n")
    w("| paper | tokens | nodes | leaves | depth | root | heading | paragraph | theorem | proof | float | groups | group tokens min / median / max | section tokens min / median / max | shared leaves | refs unresolved / forward | markup share |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in docs:
        s = stats[d["doc_id"]]
        c = s["count"]
        w(f"| {d['doc_id']} | {s['tokens']} | {s['nodes']} | {s['leaves']} | {s['depth']} | {s['root']} | {c['heading']} | {c['paragraph']} | "
          f"{c['theorem']} | {c['proof']} | {c['float']} | {len(s['groups'])} | {mmm(s['groups'])} | {mmm(s['sections'])} | {s['shared']} | "
          f"{s['refs'].get('unresolved', 0)} / {s['refs'].get('forward', 0)} | {s['markup_share']:.1%} |")
    w("")
    w("Depth counts edges from the document node (document → section → subsection → leaf). The root column is the root leaf's token count (title + abstract). "
      "Groups are the merged subtrees, one sequence each. "
      "A token is *markup* if every non-space byte of it is LaTeX syntax: `\\begin{...}`/`\\end{...}`, macro names, the key arguments of `\\label`/`\\ref`/`\\cite`/`\\includegraphics`, "
      "`{ } $ & ^ _ ~`; whitespace-only tokens count as formatting markup.\n")
    tot = {t: [0, 0] for t in LEAF_TYPES}
    for s in stats.values():
        for t, (m, n) in s["markup_by_type"].items():
            tot[t][0] += m
            tot[t][1] += n
    w("Markup share by leaf type, all papers together:\n")
    w("| leaf type | tokens | markup tokens | share |")
    w("|---|---|---|---|")
    for t in LEAF_TYPES:
        m, n = tot[t]
        if n:
            w(f"| {t} | {n} | {m} | {m / n:.1%} |")
    all_m, all_n = sum(v[0] for v in tot.values()), sum(v[1] for v in tot.values())
    w(f"| **all** | {all_n} | {all_m} | **{all_m / all_n:.1%}** |\n")

    w("## Test 0: two prompts in two sequences vs. separate runs\n")
    w("`deliverables/tests/two_seq` (flash attention on, `kv_unified = true`, 32 greedy tokens per prompt):\n")
    w("```\n" + two_seq.strip() + "\n```\n")

    w("## Tests (a) and (b)\n")
    w("(a) change subtree B (one leaf rewritten, only its group recomputed), ask about subtree A: the answer must be token-identical. "
      "(b) evict an unrelated subtree C with `llama_memory_seq_rm(seq, p0, p1)`: the answer must be token-identical and the KV memory in use "
      "(`llama_state_get_size`) must drop by at least |C| × 32 KiB. "
      "A, B and C are picked by a fixed rule (`deliverables/tests/run_tests.py`): A is the group containing the question's anchor text, "
      "B the last group sharing no cells with A (one of its leaves is rewritten), C the first other group sharing no cells with A and cited by no group. "
      f"A test passes only if it held in all {runs} runs.\n")
    w("| paper | A | question | answer | (a) | B (recomputed tokens) | (b) | C (evicted tokens) | KV before → after (KiB) | dropped / expected (KiB) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        a, b = r["test_a"], r["test_b"]
        w(f"| {r['doc']} | {r['seq_A']} | {r['question']} | `{r['answer'].strip()}` | {'PASS' if a['pass'] else 'FAIL'} | "
          f"seq {a['seq_B']} ({a['recomputed_tokens']}) | {'PASS' if b['pass'] else 'FAIL'} | seq {b['seq_C']} ({b['evicted_tokens']}) | "
          f"{b['kv_kib_before']} → {b['kv_kib_after']} | {b['kv_kib_dropped']} / {b['kv_kib_expected']} |")
    n_pass = sum(r["test_a"]["pass"] for r in results) + sum(r["test_b"]["pass"] for r in results)
    w(f"\n**{n_pass} / {2 * len(results)} tests pass.**" + (f" Skipped (fewer than 3 independent groups): {', '.join(skipped)}." if skipped else "") + "\n")
    w("Answer quality is separate from the tests (the tests check that answers do not change). "
      "Whether the 1B model's answer contains the expected fact:\n")
    w("| paper | expected | answer | correct |")
    w("|---|---|---|---|")
    for r in results:
        if r["expected"] is not None:
            w(f"| {r['doc']} | {r['expected']} | `{r['answer'].strip()}` | {'yes' if r['answer_correct'] else 'no'} |")
    w("")

    w("## CPU run times (medians)\n")
    w("| paper | tokens | prefill (ms) | prefill (tok/s) | answer (ms) | recompute one group (ms) | full prefill / recompute |")
    w("|---|---|---|---|---|---|---|")
    for r in results:
        t = r["timings_ms_median"]
        w(f"| {r['doc']} | {r['n_tokens']} | {t['prefill']} | {r['n_tokens'] / t['prefill'] * 1000:.0f} | {t['answer']} | {t['replace']} | "
          f"{t['prefill'] / t['replace']:.1f}× |")
    w("")
    w("Prefill is the whole paper (root + every group). \"Recompute one group\" is test (a)'s cost: `seq_rm` of group B plus its re-decode. "
      "Without sequence IDs, editing B would mean re-prefilling everything after it.\n")

    w("## Findings and limitations\n")
    w("- **Isolation is exact at the mask level.** A query sees a cell only if the cell carries the query's sequence ID; others get −∞, i.e. weight exactly 0.")
    w("- **But llama.cpp's CPU arithmetic depends on the cache layout.** When freed cells are reused, the question and answer land in other slots, and logits move by ~0.5. "
      "Greedy answers then flip wherever the top two tokens are nearly tied (e.g. 28.4 vs 41.8 with a margin of 0.02). "
      "Refilling the freed slots made the logits bit-identical again, which isolates the cause. The driver avoids it by prefilling the question's group first "
      "(every hole then lies after it) and by decoding single tokens on one thread. This fix was verified empirically (logit differences 0); the explanation, "
      "that multi-row attention depends only on the order of the attended cells while multi-threaded single-token attention also depends on their slots, is our reading of the measurements. "
      "With flash attention off, results also depend on batch shape: `deliverables/tests/two_seq ... 1 off` diverges after 9–31 greedy tokens.")
    w("- **Trimming** keeps theorem/proof/float leaves first, so references to trimmed parts become unresolved (see the table).")
    w("- **References:** only backward `\\ref`-family citations are modelled (the cited leaf's cells are shared with `seq_cp`, at most MAX_REFS per group). "
      "Forward references, plain-text references (\"see Theorem 1\") and `\\cite` are not.")
    w("- **Answers** come from a 1B model with a plain `Question: … Answer:` prompt; several are wrong, which does not affect the isolation tests but limits the usefulness of the answers themselves.")

    (REPORT / "report.md").write_text("\n".join(out) + "\n")
    print(f"wrote {REPORT / 'report.md'}")


if __name__ == "__main__":
    main()
