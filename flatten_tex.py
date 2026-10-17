#!/usr/bin/env python3
"""Flatten a multi-file LaTeX project into a single .tex file.

Finds the main file (the one with \\documentclass), recursively inlines
\\input / \\include / \\subfile, and drops comments (so commented-out
includes like `%\\input{draft}` are never followed).

Usage:
    python flatten_tex.py data/raw/1706.03762 -o data/flat
    python flatten_tex.py data/raw/*/ -o data/flat        # many papers at once
Each <dir> is written to <out>/<dir name>.tex.
"""
import argparse
import re
import sys
from pathlib import Path

INCLUDE_RE = re.compile(r"\\(input|include|subfile)\s*\{([^}]+)\}|\\input\s+([^\s{}\\%]+)")
COMMENT_ENV_RE = re.compile(r"\\begin\{comment\}.*?\\end\{comment\}", re.DOTALL)
DOCCLASS_RE = re.compile(r"^\s*\\documentclass", re.MULTILINE)


def strip_comments(text: str) -> str:
    """Remove %-comments (unescaped %) and comment environments."""
    out = []
    for line in text.splitlines():
        i = 0
        while True:
            i = line.find("%", i)
            if i == -1:
                break
            # % is escaped iff preceded by an odd number of backslashes
            n_bs = len(line[:i]) - len(line[:i].rstrip("\\"))
            if n_bs % 2 == 0:
                break
            i += 1
        if i == -1:
            out.append(line.rstrip())  # trailing whitespace is wasted tokens
        elif line[:i].strip():
            out.append(line[:i].rstrip())
        # else: whole-line comment -> drop the line entirely
    return COMMENT_ENV_RE.sub("", "\n".join(out) + "\n")


def find_main(root: Path) -> Path:
    candidates = []
    for p in sorted(root.rglob("*.tex")):
        text = strip_comments(p.read_text(errors="replace"))
        if DOCCLASS_RE.search(text):
            has_body = "\\begin{document}" in text
            candidates.append((has_body, len(text), p))
    if not candidates:
        raise FileNotFoundError(f"no file with \\documentclass under {root}")
    if len(candidates) > 1:
        names = ", ".join(str(c[2].relative_to(root)) for c in candidates)
        print(f"[warn] {root.name}: several main candidates ({names}); picking the largest", file=sys.stderr)
    # prefer files with \begin{document}, then the largest
    return max(candidates)[2]


def resolve(name: str, root: Path, cur_dir: Path) -> Path | None:
    name = name.strip()
    for base in (cur_dir, root):
        for cand in (base / name, base / f"{name}.tex"):
            cand = cand.resolve()
            # stay inside the project directory (sources are untrusted)
            if cand.is_file() and cand.is_relative_to(root):
                return cand
    return None


def inline(path: Path, root: Path, stack: tuple[Path, ...]) -> str:
    if path in stack:
        print(f"[warn] include cycle at {path.relative_to(root)}; skipped", file=sys.stderr)
        return ""
    text = strip_comments(path.read_text(errors="replace"))

    def repl(m: re.Match) -> str:
        name = m.group(2) or m.group(3)
        target = resolve(name, root, path.parent)
        if target is None:
            print(f"[warn] {path.relative_to(root)}: cannot find '{name}'; left as is", file=sys.stderr)
            return m.group(0)
        body = inline(target, root, stack + (path,))
        if m.group(1) == "subfile":  # subfiles carry their own preamble; keep only the body
            mb = re.search(r"\\begin\{document\}(.*)\\end\{document\}", body, re.DOTALL)
            body = mb.group(1) if mb else body
        return body.rstrip("\n")

    return INCLUDE_RE.sub(repl, text)


def flatten(src: Path) -> str:
    root = src.resolve()
    main = find_main(root)
    text = inline(main, root, ())
    return re.sub(r"\n{3,}", "\n\n", text)  # collapse blank runs left by removed comments


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="paper source directories")
    ap.add_argument("-o", "--out", type=Path, default=Path("data/flat"), help="output directory")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for d in args.dirs:
        try:
            flat = flatten(d)
        except FileNotFoundError as e:
            print(f"[skip] {e}", file=sys.stderr)
            continue
        if not re.search(r"\\section\*?\{", flat):
            print(f"[warn] {d}: no \\section found (PDF-only wrapper or unusual source?)", file=sys.stderr)
        dst = args.out / f"{d.resolve().name}.tex"
        dst.write_text(flat)
        print(f"[done] {d} -> {dst} ({len(flat.splitlines())} lines)")


if __name__ == "__main__":
    main()
