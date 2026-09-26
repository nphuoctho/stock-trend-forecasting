r"""Flatten thesis-latex into a single pandoc-friendly .tex for docx conversion.

Reads the real source tree (no copy of chapter content), resolves ``\\input``,
conditionals, custom commands, ``\\eqref``, math labels, tikz figures and
layout-only commands. Output: ``build/flattened.tex``
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "thesis-latex"
BUILD = Path(__file__).resolve().parents[1] / "build"

# ---------------------------------------------------------------- aux labels
aux = (ROOT / "BaoCaoDATN_full.aux").read_text(encoding="utf-8")
LABELS = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux))

# ---------------------------------------------------------------- input order
ORDER = [
    "frontmatter/02-hoi-dong.tex",
    "frontmatter/03-loi-cam-on.tex",
    "frontmatter/04-danh-muc-viet-tat.tex",
    "frontmatter/05-tom-tat.tex",
    "frontmatter/05b-abstract.tex",
    "chapters/01-mo-dau.tex",
    "chapters/02-tong-quan.tex",
    "chapters/03-phuong-phap.tex",
    "chapters/04-ket-qua.tex",
    "chapters/05-ket-luan.tex",
    "chapters/06-huong-phat-trien.tex",
]

def read(f):
    return (ROOT / f).read_text(encoding="utf-8")

parts = [read(f) for f in ORDER]
# bibliography placeholder + appendix
bib = r"\printbibliography" + "\n"
parts.append(bib)
parts.append("\\appendix\n" + read("chapters/07-phu-luc.tex"))
src = "\n\n".join(parts)

# ---------------------------------------------------------------- macros
MACROS = {
    "thesistitle": "Xây dựng hệ thống dự báo xu hướng biến động giá cổ phiếu dựa trên phân tích cảm xúc tin tức tài chính bằng mô hình Transformer và chuỗi thời gian",
    "thesistitleEN": "Building a stock price movement forecasting system based on financial news sentiment analysis using Transformer and time-series models",
    "authorname": "Nguyễn Phước Thọ",
    "studentid": "25410139",
    "advisor": "TS. Đặng Văn Thìn",
    "major": "Trí tuệ nhân tạo",
    "thesisyear": "2026",
}
for name, val in MACROS.items():
    src = re.sub(r"\\" + name + r"(?![a-zA-Z])\{\}?", val, src)
    src = re.sub(r"\\" + name + r"(?![a-zA-Z])", val, src)

# ---------------------------------------------------------------- conditionals
# \ifShowDiagram is true; \ifProgressReport is false
def resolve_if(s, name, keep_true):
    pat = re.compile(
        r"\\if" + name + r"\b(.*?)(?:\\else(.*?))?\\fi", re.S)
    def repl(m):
        return m.group(1) if keep_true else (m.group(2) or "")
    prev = None
    while prev != s:
        prev = s
        s = pat.sub(repl, s)
    return s

src = resolve_if(src, "ShowDiagram", True)
src = resolve_if(src, "ProgressReport", False)

# ---------------------------------------------------------------- tikz -> png
TIKZ = {
    0: r"\includegraphics[width=0.9\textwidth]{fig-kien-truc.png}",
    1: r"\includegraphics[width=0.9\textwidth]{fig-phu-luc.png}",
}
idx = [0]
def tikz_repl(m):
    out = TIKZ[idx[0]]
    idx[0] += 1
    return out
src = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", tikz_repl, src,
             flags=re.S)

# ---------------------------------------------------------------- eq numbering

def add_eq_num(m):
    env, body = m.group(1), m.group(2)
    lm = re.search(r"\\label\{([^}]*)\}", body)
    if not lm:
        return m.group(0)
    num = LABELS.get(lm.group(1))
    if not num:
        return m.group(0)
    # keep \label so the lua filter can anchor the equation
    body = body.rstrip()
    return (r"\begin{" + env + "}" + body +
            r"\qquad(\mathrm{" + num + "})" + r"\end{" + env + "}")
src = re.sub(
    r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", add_eq_num, src, flags=re.S)

# flatten \substack{a\\b} -> single-line \text; texmath cannot parse it.
# braces are nested (\text{...}), so consume balanced braces manually.
def _consume_braced(s, start):
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:i], i + 1
    return s[start + 1:], len(s)

def flatten_substack(s):
    out = []
    i = 0
    while True:
        j = s.find(r"\substack{", i)
        if j < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        inner, end = _consume_braced(s, j + len(r"\substack") + 0)
        # inner starts after '{'
        inner = inner.replace("\\\\", " ")
        inner = re.sub(r"\\text\{([^{}]*)\}", r"\1", inner)
        inner = re.sub(r"\s+", " ", inner).strip()
        out.append(r"\text{" + inner + "}")
        i = end
    return "".join(out)

src = flatten_substack(src)

# texmath (pandoc) cannot parse \notag in align; drop it
src = re.sub(r"\\notag\b", "", src)

# cases env inside equation: leave as is



# ---------------------------------------------------------------- eqref -> ref
src = re.sub(r"\\eqref\{([^}]*)\}", r"(\\ref{\1})", src)

# ---------------------------------------------------------------- tcolorbox
def tcolor(m):
    body = m.group(2)
    title = re.search(r"title=([^,\]]+)", m.group(1))
    head = ""
    if title:
        t = title.group(1).replace("\\textbf", "").strip("{}")
        head = "\n\n\\textbf{" + t + "}\n\n"
    return head + r"\begin{tcolorboxdoc}" + body + r"\end{tcolorboxdoc}"
src = re.sub(r"\\begin\{tcolorbox\}(\[[^\]]*\])?(.*?)\\end\{tcolorbox\}",
             tcolor, src, flags=re.S)
# pandoc treats unknown envs as divs with the env name as class;
# the lua filter maps 'tcolorboxdoc' to a bordered paragraph.

# ---------------------------------------------------------------- strip/layout
REMOVE_CMDS = [
    r"\\centering\b",
    r"\\onehalfspacing\b",
    r"\\clearpage\b",
    r"\\newpage\b",
    r"\\noindent\b",
    r"\\raggedright\b",
    r"\\raggedleft\b",
    r"\\vfill\b",
    r"\\selectfont\b",
    r"\\makeatletter\b",
    r"\\makeatother\b",
]
for c in REMOVE_CMDS:
    src = re.sub(c, "", src)

# \vspace{n} -> newline-ish spacing; pandoc drops it anyway, keep blank line
src = re.sub(r"\\vspace\*?\{[^}]*\}", "\n\n", src)
src = re.sub(r"\\hspace\{[^}]*\}", " ", src)
src = re.sub(r"\\hspace\*?\{[^}]*\}", " ", src)

# {\fontsize{a}{b}\selectfont ...} -> keep inner content
src = re.sub(r"\{\\fontsize\{[^}]*\}\{[^}]*\}\\selectfont\b", "{", src)
src = re.sub(r"\\fontsize\{[^}]*\}\{[^}]*\}\\selectfont\b", "", src)
src = re.sub(r"\\fontsize\{[^}]*\}\{[^}]*\}", "", src)

# \MakeUppercase{x} -> uppercase literal
src = re.sub(r"\\MakeUppercase\{([^}]*)\}",
             lambda m: m.group(1).upper(), src)

# \textbf/… kept. \par -> paragraph break
src = re.sub(r"\\par\b", "\n\n", src)
# \titleformat etc already gone (preamble not input)
# \addcontentsline -> drop (handled via TOC field)
src = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", src)

# \printbibliography -> refs div marker for citeproc
src = src.replace(r"\printbibliography",
                  "\n\n::: {#refs}\n:::\n")

# table centering is fine; ensure [H] doesn't confuse pandoc -> strip float args
src = re.sub(r"\\begin\{figure\}\[[^\]]*\]", r"\\begin{figure}", src)
src = re.sub(r"\\begin\{table\}\[[^\]]*\]", r"\\begin{table}", src)

# \textwidth in includegraphics width already fine
# \% escapes ok

# collapse 3+ blank lines
src = re.sub(r"\n{4,}", "\n\n\n", src)

out = BUILD / "flattened.tex"
out.write_text(src, encoding="utf-8")
print("flattened ->", out, len(src), "chars")

# ---------------------------------------------------------------- labels report
missing = [k for k in LABELS if ("{" + k + "}") not in src]
if missing:
    print("labels in .aux not present in flattened src:", missing)
