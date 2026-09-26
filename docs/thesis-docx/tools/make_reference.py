# /// script
# dependencies = ["python-docx"]
# ///
"""Build reference-doc.docx for pandoc: Times New Roman 13pt, 1.5 line
spacing, A4 with thesis margins (top 3 / bottom 3.5 / left 3.5 / right 2 cm),
black bold headings, small bold caption labels, centered page number footer.
"""
import sys
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ref-default.docx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "reference-doc.docx"

doc = Document(SRC)

def set_font(style, name="Times New Roman", size=None, bold=None,
             color=None, italic=None):
    f = style.font
    f.name = name
    if size is not None:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color is not None:
        f.color.rgb = RGBColor(*color)
    # also set complex-script/eastasia font for Vietnamese
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)

BLACK = (0, 0, 0)
styles = doc.styles

# ---- Normal ----
n = styles["Normal"]
set_font(n, size=13)
pf = n.paragraph_format
pf.line_spacing = 1.5
pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
pf.space_before = Pt(0)
pf.space_after = Pt(0)
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
pf.first_line_indent = Cm(1.0)

# Body Text / First Paragraph inherit Normal spacing
for sname in ("Body Text", "First Paragraph", "Compact", "Block Text"):
    try:
        s = styles[sname]
        set_font(s, size=13)
        s.paragraph_format.line_spacing = 1.5
        s.paragraph_format.space_before = Pt(0)
        s.paragraph_format.space_after = Pt(0)
        s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    except KeyError:
        pass

# ---- headings: black, bold, Vietnamese regulation sizes ----
heading_sizes = {1: 14, 2: 13, 3: 13, 4: 13, 5: 13, 6: 13}
for lvl, sz in heading_sizes.items():
    try:
        h = styles[f"Heading {lvl}"]
    except KeyError:
        continue
    set_font(h, size=sz, bold=True, color=BLACK)
    h.paragraph_format.line_spacing = 1.5
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    h.paragraph_format.first_line_indent = Cm(0)
    # remove "keep with next" weirdness is fine; ensure color theme off
    rpr = h.element.get_or_add_rPr()
    col = rpr.find(qn("w:color"))
    if col is not None:
        col.set(qn("w:val"), "000000")

# ---- captions ----
for sname in ("Table Caption", "Image Caption", "Caption"):
    try:
        s = styles[sname]
    except KeyError:
        continue
    set_font(s, size=11, bold=False)
    s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s.paragraph_format.line_spacing = 1.5
    s.paragraph_format.first_line_indent = Cm(0)
    s.paragraph_format.space_before = Pt(6)
    s.paragraph_format.space_after = Pt(6)

# ---- TOC heading / toc levels: keep black ----
for i in range(1, 5):
    try:
        s = styles[f"TOC Heading" if i == 1 else f"toc {i}"]
        set_font(s, size=13, color=BLACK)
    except KeyError:
        pass
try:
    set_font(styles["TOC Heading"], size=14, bold=True, color=BLACK)
except KeyError:
    pass

# ---- bibliography ----
for sname in ("Bibliography",):
    try:
        s = styles[sname]
        set_font(s, size=13)
        s.paragraph_format.line_spacing = 1.5
        s.paragraph_format.first_line_indent = Cm(0)
        s.paragraph_format.left_indent = Cm(0.75)
        s.paragraph_format.first_line_indent = Cm(-0.75)
    except KeyError:
        pass

# ---- page setup: A4 + thesis margins ----
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(3.0)
sec.bottom_margin = Cm(3.5)
sec.left_margin = Cm(3.5)
sec.right_margin = Cm(2.0)

# ---- footer: centered page number ----
footer = sec.footer
footer.is_linked_to_previous = False
p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.first_line_indent = Cm(0)
run = p.add_run()
fld_begin = OxmlElement("w:fldChar"); fld_begin.set(qn("w:fldCharType"), "begin")
instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
instr.text = " PAGE "
fld_end = OxmlElement("w:fldChar"); fld_end.set(qn("w:fldCharType"), "end")
run._r.append(fld_begin); run._r.append(instr); run._r.append(fld_end)

doc.save(OUT)
print("reference-doc ->", OUT)
