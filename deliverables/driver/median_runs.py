#!/usr/bin/env python3
"""Run deliverables/driver/ldls_driver several times and report the median timings.

Everything after "--" is passed to the driver unchanged.

Usage:
    python -I deliverables/driver/median_runs.py --runs 5 [--out result.json] -- -m MODEL --doc DOC.json --seq 1 -p "QUESTION" [--replace NODE TEXT] [--evict SEQ]
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

DRIVER = Path(__file__).resolve().parent / "ldls_driver"


def run_once(driver_args):
    out = subprocess.run([str(DRIVER), *driver_args], capture_output=True, text=True, check=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit(__doc__)
    split = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", type=Path, help="write the summary as JSON")
    args = ap.parse_args(argv[:split])
    driver_args = argv[split + 1:]

    runs = []
    for k in range(args.runs):
        r = run_once(driver_args)
        runs.append(r)
        print(f"run {k + 1}/{args.runs}: prefill {r['prefill_ms']:.0f} ms, answer {r['answer_ms']:.0f} ms", file=sys.stderr)

    first = runs[0]
    summary = {
        "doc_id": first["doc_id"], "seq": first["seq"], "question": first["question"], "runs": args.runs,
        "n_tokens": first["n_tokens"],
        "answer": first["answer"],
        "answer_same_every_run": all(r["answer_ids"] == first["answer_ids"] for r in runs),
        "prefill_ms_median": statistics.median(r["prefill_ms"] for r in runs),
        "prefill_ms_all": [round(r["prefill_ms"]) for r in runs],
        "answer_ms_median": statistics.median(r["answer_ms"] for r in runs),
    }
    summary["prefill_tok_per_s_median"] = first["n_tokens"] / summary["prefill_ms_median"] * 1000
    if "replace" in first:
        summary["replace"] = {k: first["replace"][k] for k in ("node", "seq", "recomputed_tokens")}
        summary["replace"]["ms_median"] = statistics.median(r["replace"]["ms"] for r in runs)
        summary["replace"]["identical_every_run"] = all(r["replace"]["identical"] for r in runs)
    if "evict" in first:
        e = first["evict"]
        summary["evict"] = {"seq": e["seq"], "p0": e["p0"], "p1": e["p1"],
                            "kv_kib_before": e["kv_bytes_before"] / 1024, "kv_kib_after": e["kv_bytes_after"] / 1024,
                            "identical_every_run": all(r["evict"]["identical"] for r in runs)}

    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
