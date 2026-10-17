#!/usr/bin/env python3
"""Turn a flattened LaTeX paper into the schema in deliverables/schema/schema.json, following deliverables/schema/MERGE_RULES.md.

Usage:
    python -I deliverables/parser/latex_parser.py data/flat/*.tex -o data/json
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from pylatexenc.latexwalker import (LatexEnvironmentNode, LatexGroupNode, LatexMacroNode, LatexWalker)

ROOT = Path(__file__).resolve().parents[2]          # repository root
TOKENIZER = Path(__file__).resolve().parent / "tokenize"
MODEL = ROOT / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

PARAMS = {"MIN_TOKENS": 64, "MAX_GROUPS": 6, "MAX_REFS": 2}
MAX_DOC_TOKENS = 2000

LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3}
FLOATS = {"figure", "table", "algorithm"}
DROP_ENVS = {"thebibliography"}
DROP_MACROS = {"bibliography", "bibliographystyle", "maketitle"}

NEWTHEOREM_RE = re.compile(r"\\newtheorem\*?\s*\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\s*\{([^}]*)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|autoref|cref|Cref)\s*\{([^}]*)\}")
# bytes that are markup, not prose (a token is markup if all its non-space bytes are)
MARKUP_RES = [re.compile(p) for p in (
    rb"\\(?:begin|end)\s*\{[^}]*\}",
    rb"\\(?:label|ref|eqref|autoref|cref|Cref|cite[a-zA-Z]*|pageref|url|includegraphics)\s*(?:\[[^\]]*\])?\s*\{[^}]*\}",
    rb"\\[A-Za-z@]+\*?",
    rb"\\.",
    rb"[{}$&^_~]",
)]


# ---------------------------------------------------------------- 1. LaTeX -> leaves
def find_anchors(nodes, theorem_envs, out):
    """Collect the nodes that start or are leaves (sectioning, title, special envs), in document order."""
    for n in nodes or []:
        if isinstance(n, LatexMacroNode):
            name = n.macroname.rstrip("*")
            if name in LEVELS:
                out.append(("heading", n, LEVELS[name]))
            elif name == "title":
                out.append(("title", n, None))
            elif name in DROP_MACROS:
                out.append(("drop", n, None))
        elif isinstance(n, LatexEnvironmentNode):
            env = n.environmentname.rstrip("*")
            if env == "abstract":
                out.append(("abstract", n, None))
            elif n.environmentname in theorem_envs:
                out.append(("theorem", n, None))
            elif env == "proof":
                out.append(("proof", n, None))
            elif env in FLOATS:
                out.append(("float", n, None))
            elif env in DROP_ENVS:
                out.append(("drop", n, None))
            else:
                find_anchors(n.nodelist, theorem_envs, out)   # e.g. \begin{appendices} holds sections
        elif isinstance(n, LatexGroupNode):
            find_anchors(n.nodelist, theorem_envs, out)


def extend_to_eol(text, end):
    """Extend a span over trailing spaces/tabs and one newline."""
    while end < len(text) and text[end] in " \t":
        end += 1
    if end < len(text) and text[end] == "\n":
        end += 1
    return end


def split_paragraphs(chunk):
    return [p.strip("\n") + "\n" for p in re.split(r"\n[ \t]*\n", chunk) if p.strip()]


def parse_leaves(src):
    """Return (root_text, leaves). Each leaf: dict(kind, text, level) in document order."""
    theorem_envs = set(NEWTHEOREM_RE.findall(src))
    nodes, _, _ = LatexWalker(src, tolerant_parsing=True).get_latex_nodes()

    doc = next((n for n in nodes if isinstance(n, LatexEnvironmentNode) and n.environmentname == "document"), None)
    if doc is None:
        raise ValueError("no \\begin{document}")
    pre_anchors, anchors = [], []
    find_anchors([n for n in nodes if n is not doc], theorem_envs, pre_anchors)
    find_anchors(doc.nodelist, theorem_envs, anchors)

    title, abstract, leaves = "", "", []
    for kind, n, _ in pre_anchors:
        if kind == "title" and not title:
            title = src[n.pos:n.pos + n.len] + "\n"

    body_start = doc.nodelist[0].pos if doc.nodelist else doc.pos
    body_end = doc.nodelist[-1].pos + doc.nodelist[-1].len if doc.nodelist else doc.pos
    cur = body_start
    for kind, n, level in anchors:
        if n.pos < cur:          # nested inside an anchor already taken
            continue
        leaves += [{"kind": "paragraph", "text": p} for p in split_paragraphs(src[cur:n.pos])]
        if kind in ("heading", "drop") and isinstance(n, LatexMacroNode):   # the whole line: args may be unparsed
            end = src.find("\n", n.pos) + 1 or len(src)
        else:
            end = extend_to_eol(src, n.pos + n.len)
        text = src[n.pos:end]
        if kind == "title":
            title = title or text
        elif kind == "abstract":
            abstract = abstract or text
        elif kind == "heading":
            leaves.append({"kind": "heading", "text": text, "level": level})
        elif kind != "drop":
            leaves.append({"kind": kind, "text": text})
        cur = end
    leaves += [{"kind": "paragraph", "text": p} for p in split_paragraphs(src[cur:body_end])]
    return title + abstract, leaves


# ---------------------------------------------------------------- 2. tokens + markup
def tokenize(texts_bos):
    inp = "".join(json.dumps({"text": t, "bos": b}) + "\n" for t, b in texts_bos)
    out = subprocess.run([str(TOKENIZER), str(MODEL)], input=inp, capture_output=True, text=True, check=True)
    return [json.loads(line) for line in out.stdout.splitlines()]


def count_markup(text, lens):
    raw = text.encode()
    if sum(lens) != len(raw):
        raise ValueError("token byte lengths do not cover the text")
    mark = bytearray(len(raw))
    for rx in MARKUP_RES:
        for m in rx.finditer(raw):
            mark[m.start():m.end()] = b"\1" * (m.end() - m.start())
    n, pos = 0, 0
    for ln in lens:
        span = range(pos, pos + ln)
        content = [i for i in span if raw[i] not in b" \t\n"]
        if ln and all(mark[i] for i in content):   # whitespace-only tokens count as formatting markup
            n += 1
        pos += ln
    return n


# ---------------------------------------------------------------- 3. tree
def build_tree(root_text, leaves):
    """Nodes in pre-order. Returns (nodes, leaf_ids), leaf_ids excluding the root."""
    nodes = [{"id": 0, "node_type": "document", "parent": None, "text": ""}]
    nodes.append({"id": 1, "node_type": "root", "parent": 0, "text": root_text})
    stack = [(0, 0)]                      # (level, node id) of open containers
    leaf_ids = []
    for lf in leaves:
        if lf["kind"] == "heading":
            while stack[-1][0] >= lf["level"]:
                stack.pop()
            cid = len(nodes)
            nodes.append({"id": cid, "node_type": {1: "section", 2: "subsection", 3: "subsubsection"}[lf["level"]],
                          "parent": stack[-1][1], "text": ""})
            stack.append((lf["level"], cid))
        nid = len(nodes)
        nodes.append({"id": nid, "node_type": lf["kind"], "parent": stack[-1][1], "text": lf["text"]})
        leaf_ids.append(nid)
    return nodes, leaf_ids


def section_of(nodes, nid):
    """Top-level section id containing node nid (None if before the first section)."""
    p = nodes[nid]["parent"]
    while p is not None and nodes[p]["node_type"] not in ("section", "document"):
        p = nodes[p]["parent"]
    return p if p is not None and nodes[p]["node_type"] == "section" else None


def trim(nodes, leaf_ids, budget):
    """Keep the paper under `budget` tokens: the root, then theorem/proof/float leaves, then paragraphs,
    each in document order, a leaf being kept only if it fits together with its not-yet-kept section headings."""
    heading_of = {nodes[i]["parent"]: i for i in leaf_ids if nodes[i]["node_type"] == "heading"}
    cost = lambda i: len(nodes[i]["tokens"])
    used, kept = cost(1), set()
    first = [i for i in leaf_ids if nodes[i]["node_type"] in ("theorem", "proof", "float")]
    rest = [i for i in leaf_ids if nodes[i]["node_type"] == "paragraph"]
    for i in first + rest:
        need, p = {i}, nodes[i]["parent"]
        while p is not None:
            if p in heading_of:
                need.add(heading_of[p])
            p = nodes[p]["parent"]
        need -= kept
        extra = sum(cost(j) for j in need)
        if used + extra < budget:
            kept |= need
            used += extra
    return [i for i in leaf_ids if i in kept]


# ---------------------------------------------------------------- 4. merge rules
def merge(nodes, leaves, P):
    size = lambda atom: sum(len(nodes[i]["tokens"]) for i in atom)
    parent = lambda i: nodes[i]["parent"]
    sec = lambda i: section_of(nodes, i)
    kind = lambda i: nodes[i]["node_type"]

    def strength(a, b):
        if sec(a) != sec(b):
            return 2
        return 1 if parent(a) != parent(b) else 0

    atoms = [[i] for i in leaves]

    # 1.1 proof joins the previous atom in the same innermost container
    out = []
    for a in atoms:
        if kind(a[0]) == "proof" and out and parent(out[-1][-1]) == parent(a[0]):
            out[-1] += a
        else:
            out.append(a)
    atoms = out

    # 1.2 float joins the previous non-heading atom in the same container, else the next atom in it
    k = 0
    while k < len(atoms):
        a = atoms[k]
        if len(a) == 1 and kind(a[0]) == "float":
            prev = atoms[k - 1] if k > 0 else None
            nxt = atoms[k + 1] if k + 1 < len(atoms) else None
            if prev and kind(prev[-1]) != "heading" and parent(prev[-1]) == parent(a[0]):
                prev += a
                del atoms[k]
                continue
            if nxt and parent(nxt[0]) == parent(a[0]):
                atoms[k + 1] = a + nxt
                del atoms[k]
                continue
        k += 1

    # 1.3 heading joins the next atom (right to left, so heading runs cascade)
    for k in range(len(atoms) - 1, -1, -1):
        if len(atoms) > 1 and len(atoms[k]) == 1 and kind(atoms[k][0]) == "heading":
            if k + 1 < len(atoms):
                atoms[k + 1] = atoms[k] + atoms[k + 1]
            else:
                atoms[k - 1] += atoms[k]
            del atoms[k]

    # 2. minimum size
    while len(atoms) > 1:
        small = next((k for k, a in enumerate(atoms) if size(a) < P["MIN_TOKENS"]), None)
        if small is None:
            break
        a = atoms[small]
        opts = []
        if small > 0 and sec(atoms[small - 1][-1]) == sec(a[0]):
            opts.append(small - 1)
        if small + 1 < len(atoms) and sec(atoms[small + 1][0]) == sec(a[-1]):
            opts.append(small + 1)
        opts += [k for k in (small - 1, small + 1) if 0 <= k < len(atoms)]
        t = opts[0]
        lo, hi = min(t, small), max(t, small)
        atoms[lo:hi + 1] = [atoms[lo] + atoms[hi]]

    # 3. budget
    while len(atoms) > P["MAX_GROUPS"]:
        k = min(range(len(atoms) - 1),
                key=lambda k: (strength(atoms[k][-1], atoms[k + 1][0]), size(atoms[k]) + size(atoms[k + 1]), k))
        atoms[k:k + 2] = [atoms[k] + atoms[k + 1]]
    return atoms


def share_refs(nodes, groups, P, stats):
    """Pass 4: for each group, the earlier leaves it cites (at most MAX_REFS), in order of first citation."""
    group_of = {i: g for g, grp in enumerate(groups, start=1) for i in grp}
    group_of[1] = 0                                        # the root is in every seq already
    label_at = {}
    for nid in [1] + [i for grp in groups for i in grp]:
        for lab in LABEL_RE.findall(nodes[nid]["text"]):
            if lab in label_at:
                stats["duplicate_labels"] += 1
            else:
                label_at[lab] = nid
    shares = {g: [] for g in range(1, len(groups) + 1)}
    for g, grp in enumerate(groups, start=1):
        for nid in grp:
            for arg in REF_RE.findall(nodes[nid]["text"]):
                for lab in (x.strip() for x in arg.split(",")):
                    target = label_at.get(lab)
                    if target is None:
                        stats["unresolved"] += 1
                    elif group_of[target] > g:
                        stats["forward"] += 1
                    elif group_of[target] in (0, g):
                        stats["same_group"] += 1
                    elif target not in shares[g]:
                        if len(shares[g]) < P["MAX_REFS"]:
                            shares[g].append(target)
                            stats["shared"] += 1
                        else:
                            stats["over_max_refs"] += 1
    return shares


# ---------------------------------------------------------------- 5. seq_ids + positions
def assign(nodes, groups, shares):
    G = len(groups)
    root = nodes[1]
    R = len(root["tokens"])
    root["positions"] = list(range(R))
    root["seq_ids"] = list(range(G + 1))
    for g, grp in enumerate(groups, start=1):
        p = max([R] + [nodes[t]["positions"][-1] + 1 for t in shares[g]])
        for nid in grp:
            n = nodes[nid]
            n["positions"] = list(range(p, p + len(n["tokens"])))
            n["seq_ids"] = [g]
            p += len(n["tokens"])
        for t in shares[g]:
            nodes[t]["seq_ids"].append(g)
    return G + 1


# ---------------------------------------------------------------- main
def parse(path, budget):
    src = path.read_text()
    root_text, leaves = parse_leaves(src)
    nodes, leaf_ids = build_tree(root_text, leaves)

    toks = tokenize([(root_text, True)] + [(nodes[i]["text"], False) for i in leaf_ids])
    for nid, t in zip([1] + leaf_ids, toks):
        nodes[nid]["tokens"] = t["ids"]
        nodes[nid]["n_markup"] = count_markup(nodes[nid]["text"], t["lens"])

    kept = trim(nodes, leaf_ids, budget)
    nodes = renumber(nodes, set(kept) | {0, 1})
    kept = [n["id"] for n in nodes if n["node_type"] not in ("document", "root") and "tokens" in n]

    groups = merge(nodes, kept, PARAMS)
    stats = dict.fromkeys(["shared", "same_group", "forward", "unresolved", "over_max_refs", "duplicate_labels"], 0)
    shares = share_refs(nodes, groups, PARAMS, stats)
    n_seq = assign(nodes, groups, shares)
    for n in nodes:
        n.setdefault("tokens", [])
        n.setdefault("positions", [])
        n.setdefault("seq_ids", [])
    keys = ["id", "node_type", "parent", "text", "tokens", "positions", "seq_ids", "n_markup"]
    doc = {"doc_id": path.stem, "params": PARAMS, "n_seq": n_seq, "ref_stats": stats,
           "nodes": [{k: n[k] for k in keys if k in n} for n in nodes]}
    return doc, stats


def renumber(nodes, keep_leaves):
    """Drop the leaves that were trimmed and the containers left empty; re-index ids and parents."""
    has_leaf = set()
    for nid in keep_leaves:
        p = nodes[nid]["parent"]
        while p is not None:
            has_leaf.add(p)
            p = nodes[p]["parent"]
    keep = [n for n in nodes if n["id"] in keep_leaves or n["id"] in has_leaf]
    new_id = {n["id"]: k for k, n in enumerate(keep)}
    for n in keep:
        n["id"] = new_id[n["id"]]
        n["parent"] = None if n["parent"] is None else new_id[n["parent"]]
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "data" / "json")
    ap.add_argument("--max-tokens", type=int, default=MAX_DOC_TOKENS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.files:
        doc, stats = parse(f, args.max_tokens)
        n_tok = sum(len(n["tokens"]) for n in doc["nodes"])
        assert n_tok < args.max_tokens, f"{f}: {n_tok} tokens"
        (args.out / f"{doc['doc_id']}.json").write_text(json.dumps(doc))
        print(f"[done] {doc['doc_id']}: {n_tok} tokens, {len(doc['nodes'])} nodes, "
              f"{doc['n_seq'] - 1} groups, refs {stats}", file=sys.stderr)


if __name__ == "__main__":
    main()
