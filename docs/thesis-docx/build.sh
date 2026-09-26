#!/usr/bin/env bash
# Rebuild the thesis docx from the LaTeX sources.
# Usage: docs/thesis-docx/build.sh
# Requires: xelatex + pdftoppm (for TikZ figures), uv (python-docx), and
# pandoc via pypandoc-binary cache (first run: `uvx --from pypandoc-binary
# python -c "import pypandoc"` to populate the cache).
set -euo pipefail
cd "$(dirname "$0")"

PANDOC="${PANDOC:-$(find "$HOME/.cache/uv" -name pandoc -type f 2>/dev/null | head -1)}"
if [ -z "$PANDOC" ]; then
  echo "pandoc binary not found; run:" >&2
  echo '  uvx --from pypandoc-binary python -c "import pypandoc"' >&2
  exit 1
fi
echo "pandoc: $($PANDOC --version | head -1)"

# 1. render TikZ figures -> PNG (skip if PNGs exist)
for name in fig-kien-truc fig-phu-luc; do
  if [ ! -f "build/${name}.png" ]; then
    echo "rendering $name"
    # sources extracted by preprocess step 0; see tools/render-figures.sh
    bash tools/render-figures.sh
    break
  fi
done

# 2. flatten + preprocess
python3 tools/preprocess.py

# 3. label map for the lua filter
python3 - <<'PY'
import re
aux = open('../thesis-latex/BaoCaoDATN_full.aux').read()
labels = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux))
with open('build/labelmap.lua', 'w') as f:
    f.write("return {\n")
    for k, v in labels.items():
        f.write(f'  ["{k}"] = "{v}",\n')
    f.write("}\n")
print(len(labels), "labels")
PY

# 4. reference doc (style template)
if [ ! -f build/reference-doc.docx ]; then
  $PANDOC -o /tmp/ref-default.docx --print-default-data-file reference.docx
  uv run --with python-docx tools/make_reference.py /tmp/ref-default.docx build/reference-doc.docx
fi

# 5. convert
cd build
"$PANDOC" flattened.tex -f latex -t docx \
  --resource-path=. \
  --lua-filter=../tools/fixups.lua \
  --citeproc --bibliography=../../thesis-latex/references.bib \
  --csl=../tools/ieee.csl \
  -M link-citations=true \
  -M figPrefix="Hình" -M tblPrefix="Bảng" -M eqnPrefix="phương trình" \
  --reference-doc=reference-doc.docx \
  -o thesis-raw.docx
cd ..

# 6. post-process -> final docx
uv run --with python-docx tools/postprocess.py build/thesis-raw.docx \
   "BaoCaoDATN_25410139_NguyenPhuocTho.docx" \
   ../thesis-latex/BaoCaoDATN_full.aux
