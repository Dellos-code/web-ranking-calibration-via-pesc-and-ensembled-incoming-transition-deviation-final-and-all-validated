#!/bin/bash
# Builds DOCUMENTATION_EN.pdf and DOCUMENTATION_GR.pdf from the Markdown sources: pandoc writes a
# standalone LaTeX file, XeLaTeX compiles it (three passes for the table of contents), and the log is
# checked for errors and overfull boxes.
set -e
export LC_ALL=C.UTF-8 LANG=C.UTF-8
cd "$(dirname "$0")"
BUILD=$(mktemp -d)
cat > "$BUILD/header.tex" <<'TEX'
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}}
\fvset{breaklines=true}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
\setlength{\emergencystretch}{3em}
\AtBeginDocument{\sloppy}
TEX
COMMON=(--standalone --toc -V fontfamily=fontspec -V mainfont=FreeSerif -V mainfontoptions="Language=Default"
        -V mathfont="Latin Modern Math" -V monofont="DejaVu Sans Mono" -V monofontoptions="Scale=0.82"
        -V geometry:margin=2.2cm -V fontsize=10pt -V colorlinks=true -V linkcolor=blue -V urlcolor=blue
        -H "$BUILD/header.tex")
for L in EN GR; do
  pandoc "DOCUMENTATION_$L.md" -t latex -o "$BUILD/doc_$L.tex" "${COMMON[@]}"
  for pass in 1 2 3; do
    (cd "$BUILD" && xelatex -interaction=nonstopmode "doc_$L.tex" > "doc_$L.pass$pass.log" 2>&1) || true
  done
  if grep -Ein '^!|Overfull|multiply defined|undefined (references|citations?)' \
      "$BUILD/doc_$L.log" "$BUILD/doc_$L.pass3.log"; then
    echo "documentation build checks failed for $L" >&2
    rm -rf "$BUILD"
    exit 1
  fi
  errors=$(grep -c '^!' "$BUILD/doc_$L.log" || true)
  overfull=$(grep -c 'Overfull' "$BUILD/doc_$L.log" || true)
  cp "$BUILD/doc_$L.pdf" "DOCUMENTATION_$L.pdf"
  cp "$BUILD/doc_$L.log" "logs/build_DOCUMENTATION_$L.log"
  echo "DOCUMENTATION_$L.pdf: $errors errors, $overfull overfull boxes, $(pdfinfo "DOCUMENTATION_$L.pdf" | grep Pages)"
done
rm -rf "$BUILD"
