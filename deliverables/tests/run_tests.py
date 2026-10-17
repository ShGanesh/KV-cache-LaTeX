#!/usr/bin/env python3
"""The assignment's two tests on every parsed paper in data/json/.

(a) change subtree B (--replace), ask about subtree A: the answer must be token-identical.
(b) remove an unrelated subtree C (--evict): the answer must hold and KV memory must drop.

A, B and C are chosen per paper, so any parsed LaTeX paper can be tested:
- A: the first group containing the case's `anchor` text (deliverables/tests/cases.json), else group 1.
  Papers without a case get `default_question`.
- B: the last group that shares no cells with A and has an unshared non-heading leaf; that leaf is rewritten.
- C: the first remaining group that shares no cells with A and none of whose leaves is cited elsewhere.
Papers with fewer than 3 groups are skipped.

Each paper is run --runs times through driver/median_runs.py (median timings).
Writes deliverables/report/results/<doc>.json per paper and deliverables/report/results.json; exit code 1 if any test fails.

Usage:
    python -I deliverables/tests/run_tests.py [--runs 5]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repository root
DELIV = ROOT / "deliverables"
MODEL = ROOT / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
KV_BYTES_PER_TOKEN = 32 * 1024


def pick_groups(doc, anchor):
    """Return (A, B node id, C, note) for a parsed paper, or None if it has too few groups."""
    leaves = [n for n in doc["nodes"] if n["tokens"] and n["node_type"] != "root"]
    G = doc["n_seq"] - 1
    if G < 3:
        return None
    own = {g: [n for n in leaves if n["seq_ids"][0] == g] for g in range(1, G + 1)}

    def related(g):
        """Groups that share cells with g, in either direction."""
        out = set()
        for n in leaves:
            home, rest = n["seq_ids"][0], n["seq_ids"][1:]
            if home == g:
                out.update(rest)
            elif g in rest:
                out.add(home)
        return out

    a, note = 1, "no anchor: group 1"
    if anchor:
        hit = next((n["seq_ids"][0] for n in leaves if anchor in n["text"]), None)
        a, note = (hit, "anchor found") if hit else (1, "anchor not found: group 1, default question")
    free = [g for g in range(1, G + 1) if g != a and g not in related(a)]
    b_opts = [g for g in free if any(len(n["seq_ids"]) == 1 and n["node_type"] != "heading" for n in own[g])]
    if not b_opts:
        return None
    b = b_opts[-1]
    b_node = next(n["id"] for n in own[b] if len(n["seq_ids"]) == 1 and n["node_type"] != "heading")
    c = next((g for g in free if g != b and all(len(n["seq_ids"]) == 1 for n in own[g])), None)
    if c is None:
        return None
    return a, b_node, c, note, sum(len(n["tokens"]) for n in own[c])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    spec = json.loads((DELIV / "tests" / "cases.json").read_text())
    cases = {c["doc"]: c for c in spec["cases"]}
    out_dir = DELIV / "report" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results, skipped = [], []
    for doc_path in sorted((ROOT / "data" / "json").glob("*.json")):
        doc = json.loads(doc_path.read_text())
        case = cases.get(doc["doc_id"], {})
        picked = pick_groups(doc, case.get("anchor"))
        if picked is None:
            skipped.append(doc["doc_id"])
            print(f"== {doc['doc_id']}: skipped (needs 3 groups that share no cells)", file=sys.stderr)
            continue
        a, b_node, c, note, c_tokens = picked
        found = note == "anchor found"
        question = case["question"] if found else spec["default_question"]
        expected = case.get("expected") if found else None

        out = out_dir / f"{doc['doc_id']}.json"
        print(f"== {doc['doc_id']}: A = {a} ({note}), B node = {b_node}, C = {c}; {args.runs} runs", file=sys.stderr)
        subprocess.run([sys.executable, "-I", str(DELIV / "driver" / "median_runs.py"), "--runs", str(args.runs), "--out", str(out), "--",
                        "-m", str(MODEL), "--doc", str(doc_path), "--seq", str(a), "-p", question,
                        "--replace", str(b_node), spec["replace_text"], "--evict", str(c)],
                       check=True, stdout=subprocess.DEVNULL)
        r = json.loads(out.read_text())

        e = r["evict"]
        dropped_kib = e["kv_kib_before"] - e["kv_kib_after"]
        expected_kib = c_tokens * KV_BYTES_PER_TOKEN / 1024
        results.append({
            "doc": doc["doc_id"], "question": question, "seq_A": a, "choice": note, "answer": r["answer"],
            "expected": expected, "answer_correct": None if expected is None else expected.lower() in r["answer"].lower(),
            "answer_same_every_run": r["answer_same_every_run"],
            "test_a": {"seq_B": r["replace"]["seq"], "node": b_node,
                       "recomputed_tokens": r["replace"]["recomputed_tokens"],
                       "pass": r["replace"]["identical_every_run"]},
            "test_b": {"seq_C": e["seq"], "evicted_tokens": c_tokens,
                       "kv_kib_before": round(e["kv_kib_before"], 1), "kv_kib_after": round(e["kv_kib_after"], 1),
                       "kv_kib_dropped": round(dropped_kib, 1), "kv_kib_expected": round(expected_kib, 1),
                       "pass": e["identical_every_run"] and dropped_kib >= expected_kib},
            "timings_ms_median": {"prefill": round(r["prefill_ms_median"]), "answer": round(r["answer_ms_median"]),
                                  "replace": round(r["replace"]["ms_median"])},
            "n_tokens": r["n_tokens"], "runs": r["runs"],
        })

    (DELIV / "report" / "results.json").write_text(json.dumps({"results": results, "skipped": skipped}, indent=2) + "\n")

    print(f"\n{'doc':<11} {'A':>2} {'answer':<42} {'(a)':<5} {'(b)':<5} {'KV dropped KiB':>14} {'prefill ms':>10}")
    for x in results:
        print(f"{x['doc']:<11} {x['seq_A']:>2} {x['answer'].strip()[:42]:<42} "
              f"{'PASS' if x['test_a']['pass'] else 'FAIL':<5} {'PASS' if x['test_b']['pass'] else 'FAIL':<5} "
              f"{x['test_b']['kv_kib_dropped']:>14} {x['timings_ms_median']['prefill']:>10}")
    if skipped:
        print(f"skipped: {', '.join(skipped)}")
    ok = bool(results) and all(x["test_a"]["pass"] and x["test_b"]["pass"] for x in results)
    print(f"\nALL TESTS: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
