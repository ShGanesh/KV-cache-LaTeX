#!/usr/bin/env bash
# Download arXiv LaTeX sources into data/raw/<arxiv_id>/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$ROOT/data/raw"

DEFAULT_IDS=(
  1406.2661   # GAN
  1904.09237  # AMSGrad ("On the Convergence of Adam and Beyond")
  1701.07875  # Wasserstein GAN
  1806.07572  # Neural Tangent Kernel
  1706.03762  # Transformer
  2006.11239  # DDPM
) 
# Append here for more papers. 
IDS=("${@:-${DEFAULT_IDS[@]}}")

mkdir -p "$RAW"

for id in "${IDS[@]}"; do
  dir="$RAW/$id"
  if [[ -d "$dir" ]] && [[ -n "$(find "$dir" -name '*.tex' -print -quit)" ]]; then
    echo "[skip] $id (already downloaded)"
    continue
  fi
  mkdir -p "$dir"
  archive="$dir/source.download"

  echo "[get ] $id"
  curl -fsSL -A "LDLS-coursework/1.0" \
       -o "$archive" "https://arxiv.org/e-print/$id"

  # arXiv can return a tar.gz (multi-file), a plain gzipped .tex (single-file), or a PDF (no source).
  case "$(file -b --mime-type "$archive")" in
    application/gzip|application/x-gzip)
      if tar -tzf "$archive" > /dev/null 2>&1; then
        tar -xzf "$archive" -C "$dir" --no-same-owner --no-same-permissions
      else
        gunzip -c "$archive" > "$dir/main.tex"
      fi
      ;;
    application/x-tar)
      tar -xf "$archive" -C "$dir" --no-same-owner --no-same-permissions
      ;;
    text/x-tex|text/plain)
      mv "$archive" "$dir/main.tex"
      ;;
    application/pdf)
      echo "[warn] $id has no LaTeX source (PDF only); skipping" >&2
      ;;
    *)
      echo "[warn] $id: unrecognised format '$(file -b "$archive")'" >&2
      ;;
  esac
  rm -f "$archive"

  n_tex=$(find "$dir" -name '*.tex' | wc -l)
  echo "[done] $id -> $n_tex .tex file(s)"

  sleep 3   # arXiv asks for <= 1 request every 3 seconds
done

echo "Sources in $RAW"

# Flatten every downloaded paper (main file + \input/\include) into data/flat/<id>.tex
FLAT="$ROOT/data/flat"
dirs=()
for id in "${IDS[@]}"; do
  [[ -d "$RAW/$id" && -n "$(find "$RAW/$id" -name '*.tex' -print -quit)" ]] && dirs+=("$RAW/$id")
done
if (( ${#dirs[@]} )); then
  "${PYTHON:-python3}" -I "$ROOT/flatten_tex.py" "${dirs[@]}" -o "$FLAT"
  echo "Flattened sources in $FLAT"
fi
