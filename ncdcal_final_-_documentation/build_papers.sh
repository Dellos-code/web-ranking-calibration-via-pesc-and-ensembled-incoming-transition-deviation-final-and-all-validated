#!/bin/bash
# Rebuilds the numbers/tables of both manuscripts and compiles them with XeLaTeX (two passes each).
set -e
cd "$(dirname "$0")"
python3 make_paper_assets.py
python3 check_build.py
cd paper
for f in paper_en paper_gr; do
  xelatex -interaction=nonstopmode "$f.tex" > "build_$f.1.log" 2>&1
  xelatex -interaction=nonstopmode "$f.tex" > "build_$f.2.log" 2>&1
  if grep -Ein '^!|Overfull|multiply defined|undefined (references|citations?)' "$f.log" \
      "build_$f.2.log"; then
    echo "paper build checks failed for $f" >&2
    exit 1
  fi
  echo "$f.pdf: $(grep -c 'Overfull' "$f.log" || true) overfull boxes, $(pdfinfo "$f.pdf" | grep Pages)"
done
