#!/usr/bin/env bash
# Render the two tikzpicture figures to PNG for docx embedding.
# Sources: ../thesis-latex/chapters/03-phuong-phap.tex, 07-phu-luc.tex
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
for src_f, name in [('../thesis-latex/chapters/03-phuong-phap.tex', 'fig-kien-truc'),
                    ('../thesis-latex/chapters/07-phu-luc.tex', 'fig-phu-luc')]:
    src = open(src_f).read()
    m = re.search(r'\\begin\{tikzpicture\}\s*\[.*?\\end\{tikzpicture\}', src, re.S)
    open(f'build/tikz/{name}.tex', 'w').write(hdr + m.group(0) + '\n' + ftr)
    print(name)
PY

for f in fig-kien-truc fig-phu-luc; do
  (cd "$TMP" && xelatex -interaction=nonstopmode "$f.tex" >/dev/null)
  pdftoppm -png -r 200 "$TMP/$f.pdf" "$TMP/$f"
  cp "$TMP/${f}-1.png" "build/${f}.png"
done
echo "figures rendered to build/*.png"
