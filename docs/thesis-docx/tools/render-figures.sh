#!/usr/bin/env bash
# Render the two tikzpicture figures from the canonical thesis source to PNG.
# Source: ../thesis-latex/BaoCaoDATN.tex
set -euo pipefail
cd "$(dirname "$0")/.."
LATEX=../thesis-latex
TMP=build/tikz
mkdir -p "$TMP"

cat > "$TMP/header.tex" <<'EOF'
\documentclass[12pt,border=3mm]{standalone}
\usepackage{fontspec}
\usepackage{polyglossia}
\setmainlanguage{vietnamese}
\setotherlanguage{english}
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{Liberation Serif}}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds, calc}
\begin{document}
EOF
printf '\\end{document}\n' > "$TMP/footer.tex"

python3 - <<'PY'
import re
hdr = open('build/tikz/header.tex').read()
ftr = open('build/tikz/footer.tex').read()
src = open('../thesis-latex/BaoCaoDATN.tex').read()
src = src.split(r'\begin{document}', 1)[1].split(r'\end{document}', 1)[0]
figures = re.findall(
    r'\\begin\{tikzpicture\}\s*\[.*?\\end\{tikzpicture\}', src, re.S)
if len(figures) != 2:
    raise RuntimeError(f'Expected 2 tikz figures, found {len(figures)}')
for body, name in zip(figures, ('fig-kien-truc', 'fig-phu-luc')):
    open(f'build/tikz/{name}.tex', 'w').write(
        hdr + body + '\n' + ftr)
    print(name)
PY

for f in fig-kien-truc fig-phu-luc; do
  (cd "$TMP" && xelatex -interaction=nonstopmode "$f.tex" >/dev/null)
  pdftoppm -png -r 200 "$TMP/$f.pdf" "$TMP/$f"
  cp "$TMP/${f}-1.png" "build/${f}.png"
done
echo "figures rendered to build/*.png"
