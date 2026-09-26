# /// script
# dependencies = ["python-docx"]
# ///
"""Post-process pandoc-generated thesis docx.

Input : build/thesis-raw.docx  (pandoc output; refs land at document end)
Output: BaoCaoDATN_25410139_NguyenPhuocTho.docx

Steps:
 1. Insert 2 cover pages (with double page border) before committee page.
 2. Section breaks: covers / front matter (no page numbers) / body
    (page numbering restarts at 1, centered PAGE footer).
 3. Number headings: "Chương N." / "N.M." / appendix "PHỤ LỤC A", "A.1".
 4. Bold "Hình x.y:" / "Bảng x.y:" caption prefixes; build LOF/LOT.
 5. Move bibliography before appendix, add TÀI LIỆU THAM KHẢO heading.
 6. framed-note -> bordered paragraph; tables -> thin borders;
    Heading1 -> page-break-before.
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import Part
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

IN, OUT = sys.argv[1], sys.argv[2]
AUX = sys.argv[3] if len(sys.argv) > 3 else "../thesis-latex/BaoCaoDATN_full.aux"
LABELS = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}",
                         Path(AUX).read_text(encoding="utf-8")))

doc = Document(IN)
body = doc.element.body

def el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(f"w:{k}"), str(v))
    return e

def para_text(p):
    return "".join(t.text or "" for t in p.iter(qn("w:t")))

def style_of(p):
    ppr = p.find(qn("w:pPr"))
    if ppr is None:
        return None
    st = ppr.find(qn("w:pStyle"))
    return st.get(qn("w:val")) if st is not None else None

def new_p(text="", size=13, bold=False, italic=False, align="center",
          before=0, after=0):
    p = OxmlElement("w:p")
    ppr = el("w:pPr")
    ppr.append(el("w:jc", val=align))
    ppr.append(el("w:spacing", before=str(before), after=str(after),
                  line="360", lineRule="auto"))
    ind = el("w:ind", firstLine="0"); ppr.append(ind)
    p.append(ppr)
    if text:
        r = OxmlElement("w:r")
        rpr = el("w:rPr")
        rpr.append(el("w:rFonts", ascii="Times New Roman",
                      hAnsi="Times New Roman", cs="Times New Roman"))
        rpr.append(el("w:sz", val=str(size * 2)))
        if bold:
            rpr.append(el("w:b"))
        if italic:
            rpr.append(el("w:i"))
        r.append(rpr)
        t = el("w:t"); t.text = text
        t.set(qn("xml:space"), "preserve")
        r.append(t)
        p.append(r)
    return p

def sect_pr():
    sp = el("w:sectPr")
    sp.append(el("w:pgSz", w="11906", h="16838"))
    sp.append(el("w:pgMar", top="1701", right="1134", bottom="1984",
                 left="1984", header="851", footer="851", gutter="0"))
    return sp

def insert_after_first(sect, node):
    """insert footerReference/pgBorders honoring sectPr schema order:
    headerReference*, footerReference*, footnotePr, endnotePr, type,
    pgSz, pgMar, paperSrc, pgBorders, ..."""
    # simplest: header/footer refs go first, pgBorders right after pgMar
    pass  # handled inline below

def make_footer_part(name, field=None):
    xml = (b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           b'<w:ftr xmlns:w="http://schemas.openxmlformats.org/'
           b'wordprocessingml/2006/main">')
    if field:
        xml += (b'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r>'
                b'<w:fldChar w:fldCharType="begin"/></w:r><w:r>'
                b'<w:instrText xml:space="preserve"> PAGE </w:instrText>'
                b'</w:r><w:r><w:fldChar w:fldCharType="end"/></w:r>'
                b'</w:p>')
    else:
        xml += b'<w:p><w:pPr><w:jc w:val="center"/></w:pPr></w:p>'
    xml += b'</w:ftr>'
    return Part(PackURI(f"/word/{name}.xml"),
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.footer+xml",
                xml, doc.part.package)

def add_footer_ref(sect, rid):
    """footerReference must precede pgSz in sectPr."""
    fr = OxmlElement("w:footerReference")
    fr.set(qn("w:type"), "default")
    fr.set(qn("r:id"), rid)
    sect.insert(0, fr)

def sect_break_par(sect):
    p = OxmlElement("w:p")
    ppr = el("w:pPr")
    ppr.append(sect)
    p.append(ppr)
    return p

# ------------------------------------------------------------- 1. covers
TITLE_VN = ("Xây dựng hệ thống dự báo xu hướng biến động giá cổ phiếu "
            "dựa trên phân tích cảm xúc tin tức tài chính bằng mô hình "
            "Transformer và chuỗi thời gian")
TITLE_EN = ("Building a stock price movement forecasting system based on "
            "financial news sentiment analysis using Transformer and "
            "time-series models")
def cover(with_advisor):
    ps = [new_p("", after=400)]
    ps.append(new_p("ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH", 13, bold=True,
                    after=120))
    ps.append(new_p("TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN", 13, bold=True,
                    after=120))
    ps.append(new_p("KHOA KHOA HỌC MÁY TÍNH", 13, after=700))
    name = "NGUYỄN PHƯỚC THỌ" + (" – 25410139" if with_advisor else "")
    ps.append(new_p(name, 14, after=1100))
    ps.append(new_p("ĐỒ ÁN TỐT NGHIỆP", 16, bold=True, after=600))
    ps.append(new_p(TITLE_VN, 15, bold=True, after=300))
    ps.append(new_p(TITLE_EN, 13, italic=True, after=1100))
    ps.append(new_p("CỬ NHÂN NGÀNH TRÍ TUỆ NHÂN TẠO", 14, after=700))
    if with_advisor:
        ps.append(new_p("GIẢNG VIÊN HƯỚNG DẪN", 13, after=120))
        ps.append(new_p("TS. Đặng Văn Thìn", 13, after=700))
    else:
        ps.append(new_p("", after=900))
    ps.append(new_p("TP. HỒ CHÍ MINH, 2026", 13, before=500))
    return ps

# footers
rid_blank = doc.part.relate_to(make_footer_part("footerBlank"), RT.FOOTER)
rid_page = doc.part.relate_to(make_footer_part("footerPage", field=True),
                              RT.FOOTER)


def idx_of_text(prefix, start=0):
    for i, c in enumerate(children):
        if i < start or c.tag != qn("w:p"):
            continue
        if para_text(c).strip().startswith(prefix):
            return i
    return -1

# section 1 (covers): double page border + blank footer
sec1 = sect_pr()
pgb = el("w:pgBorders", offsetFrom="page")
for side in ("top", "left", "bottom", "right"):
    pgb.append(el(f"w:{side}", val="double", sz="12", space="24",
                  color="000000"))
sec1.append(pgb)            # pgBorders right after pgMar
add_footer_ref(sec1, rid_blank)  # footerRef hoisted to front by helper

first_p = body.find(qn("w:p"))
for p in cover(False) + cover(True):
    first_p.addprevious(p)

children = list(body)

# reorder: move committee + acknowledgements right after the covers
# (the lua filter prepended TOC/LOF/LOT at doc start, before them)
i_mucluc = idx_of_text("MỤC LỤC")
i_committee = idx_of_text("THÔNG TIN HỘI ĐỒNG")
i_abbrev = idx_of_text("DANH MỤC TỪ VIẾT TẮT")
moved = children[i_committee:i_abbrev]   # hội đồng + lời cảm ơn
for c in moved:
    body.remove(c)
anchor = children[i_mucluc]              # insert before MỤC LỤC
for c in moved:
    anchor.addprevious(c)
children = list(body)

# acknowledgements sign-off -> right align
for c in children:
    if c.tag == qn("w:p") and "Sinh viên thực hiện" in para_text(c):
        ppr = c.find(qn("w:pPr"))
        if ppr is None:
            ppr = el("w:pPr"); c.insert(0, ppr)
        ppr.append(el("w:jc", val="right"))

# section 1 = covers only: break before MỤC LỤC (border + no footer)
# section 1 = covers only: break before committee page (border + no footer)
i_committee = idx_of_text("THÔNG TIN HỘI ĐỒNG")
children[i_committee].addprevious(sect_break_par(sec1))
children = list(body)

# section 2 (front matter): blank footer, no border
sec2 = sect_pr()
add_footer_ref(sec2, rid_blank)
i_abstract = idx_of_text("TÓM TẮT ĐỒ ÁN")
children[i_abstract].addprevious(sect_break_par(sec2))
children = list(body)

# main section: page numbering restart at 1 + PAGE footer
main_sect = body.find(qn("w:sectPr"))
add_footer_ref(main_sect, rid_page)
# pgNumType after pgMar/pgBorders order-wise; appending is accepted
main_sect.append(el("w:pgNumType", start="1"))

# ------------------------------------------------------------- 2. numbering
CH_TITLES = [
    "MỞ ĐẦU", "TỔNG QUAN", "CƠ SỞ LÝ THUYẾT VÀ PHƯƠNG PHÁP NGHIÊN CỨU",
    "TRÌNH BÀY, ĐÁNH GIÁ VÀ BÀN LUẬN KẾT QUẢ", "KẾT LUẬN",
    "HƯỚNG PHÁT TRIỂN",
]

def prepend_text(p, prefix, bold=False):
    r = OxmlElement("w:r")
    if bold:
        rpr = el("w:rPr"); rpr.append(el("w:b")); r.append(rpr)
    t = el("w:t"); t.text = prefix
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    first_r = p.find(qn("w:r"))
    if first_r is not None:
        first_r.addprevious(r)
    else:
        p.append(r)

def append_text(p, suffix):
    runs = p.findall(qn("w:r"))
    if runs:
        t = runs[-1].find(qn("w:t"))
        if t is not None:
            t.text = (t.text or "") + suffix
            return
    prepend_text(p, suffix)

chapter = 0
sub = [0, 0, 0]
in_appendix = False

for c in children:
    if c.tag != qn("w:p"):
        continue
    st = style_of(c)
    if not st or not st.startswith("Heading"):
        continue
    lvl = int(st[-1])
    txt = para_text(c).strip()
    if lvl == 1:
        if txt == "PHỤ LỤC":
            in_appendix = True
            chapter = "A"
            sub = [0, 0, 0]
            append_text(c, " A")
        elif txt in CH_TITLES:
            chapter = CH_TITLES.index(txt) + 1
            sub = [0, 0, 0]
            prepend_text(c, f"Chương {chapter}. ")
        continue
    if chapter == 0:
        continue
    depth = lvl - 1  # Heading2 -> sub[0]
    if depth > 3:
        continue
    sub[depth - 1] += 1
    for k in range(depth, 3):
        sub[k] = 0
    nums = [str(chapter)] + [str(x) for x in sub[:depth]]
    prepend_text(c, ".".join(nums) + ". ")

# ------------------------------------------------------------- 3. captions
chapter = 0
cf = tf = 0
fig_list, tab_list = [], []
fig_ph = tab_ph = None
for c in children:
    if c.tag != qn("w:p"):
        continue
    st = style_of(c)
    txt = para_text(c).strip()
    if st == "Heading1":
        if txt.startswith("Chương "):
            chapter = txt.split(".")[0].split()[-1]
            cf = tf = 0
        elif txt == "PHỤ LỤC A":
            chapter = "A"
            cf = tf = 0
    elif st == "TableCaption":
        tf += 1
        num = f"{chapter}.{tf}"
        prepend_text(c, f"Bảng {num}: ", bold=True)
        tab_list.append((num, txt))
    elif st == "ImageCaption":
        cf += 1
        num = f"{chapter}.{cf}"
        prepend_text(c, f"Hình {num}: ", bold=True)
        fig_list.append((num, txt))
    elif txt == "DANH_MUC_HINH_PLACEHOLDER":
        fig_ph = c
    elif txt == "DANH_MUC_BANG_PLACEHOLDER":
        tab_ph = c

# map display number -> bookmark name for PAGEREF (labels from .aux order)
def list_p(kind, num, cap, bookmark):
    p = OxmlElement("w:p")
    ppr = el("w:pPr")
    ppr.append(el("w:spacing", after="60", line="360", lineRule="auto"))
    ppr.append(el("w:tabs"))
    tabs = ppr.find(qn("w:tabs"))
    tabs.append(el("w:tab", val="right", leader="dot", pos="9126"))
    p.append(ppr)
    r1 = OxmlElement("w:r"); r1.append(el("w:rPr"))
    r1.find(qn("w:rPr")).append(el("w:b"))
    t1 = el("w:t"); t1.text = f"{kind} {num}: "
    t1.set(qn("xml:space"), "preserve"); r1.append(t1); p.append(r1)
    r2 = OxmlElement("w:r")
    t2 = el("w:t"); t2.text = cap; t2.set(qn("xml:space"), "preserve")
    r2.append(t2); p.append(r2)
    # tab + PAGEREF field
    rt = OxmlElement("w:r"); rt.append(OxmlElement("w:tab")); p.append(rt)
    rb = OxmlElement("w:r"); rb.append(el("w:fldChar", fldCharType="begin"))
    p.append(rb)
    ri = OxmlElement("w:r"); it = el("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f' PAGEREF {bookmark} \\h '
    ri.append(it); p.append(ri)
    rs = OxmlElement("w:r"); rs.append(el("w:fldChar", fldCharType="separate"))
    p.append(rs)
    rd = OxmlElement("w:r"); td = el("w:t"); td.text = "?"
    rd.append(td); p.append(rd)
    re_ = OxmlElement("w:r"); re_.append(el("w:fldChar", fldCharType="end"))
    p.append(re_)
    return p

# label -> displayed number, in caption-document order
CAP_BOOK = {}
for key in LABELS:
    if key.startswith("fig:"):
        CAP_BOOK[LABELS[key]] = ("Hình", key)
    elif key.startswith("tab:"):
        CAP_BOOK[LABELS[key]] = ("Bảng", key)

def cap_bookmark(kind, num):
    kind_map = {"Hình": "fig", "Bảng": "tab"}
    for lbl, v in LABELS.items():
        if v == num and lbl.startswith(kind_map[kind] + ":"):
            return lbl
    return None

if fig_ph is not None:
    for num, cap in fig_list:
        bm = cap_bookmark("Hình", num) or ""
        fig_ph.addnext(list_p("Hình", num, cap, bm))
        fig_ph = fig_ph.getnext()
for c in list(body.iter(qn("w:p"))):
    if para_text(c).strip() in ("DANH_MUC_HINH_PLACEHOLDER",
                                "DANH_MUC_BANG_PLACEHOLDER"):
        body.remove(c)
if tab_ph is not None:
    children = list(body)
    i = idx_of_text("DANH MỤC BẢNG BIỂU")
    anchor = children[i]
    for num, cap in tab_list:
        bm = cap_bookmark("Bảng", num) or ""
        anchor.addnext(list_p("Bảng", num, cap, bm))
        anchor = anchor.getnext()

# ------------------------------------------------------------- 4. refs move
children = list(body)
refs_paras = []
appendix_el = None
in_refs = False
for c in children:
    if c.tag != qn("w:p"):
        continue
    st = style_of(c)
    txt = para_text(c).strip()
    if st == "Heading1" and txt == "PHỤ LỤC A":
        appendix_el = c
    if st == "Bibliography":
        in_refs = True
        refs_paras.append(c)
    elif in_refs:
        in_refs = False

# drop literal ::: {#refs} marker paragraphs
for c in list(body.iter(qn("w:p"))):
    if para_text(c).strip() in (":::", "#refs"):
        body.remove(c)

if appendix_el is not None and refs_paras:
    h = OxmlElement("w:p")
    ppr = el("w:pPr"); ppr.append(el("w:pStyle", val="Heading1"))
    h.append(ppr)
    r = OxmlElement("w:r"); t = el("w:t"); t.text = "TÀI LIỆU THAM KHẢO"
    r.append(t); h.append(r)
    appendix_el.addprevious(h)
    for rp in refs_paras:
        appendix_el.addprevious(rp)

# ------------------------------------------------------------- 5. misc
for c in body.iter(qn("w:p")):
    ppr = c.find(qn("w:pPr"))
    if ppr is None:
        continue
    pst = ppr.find(qn("w:pStyle"))
    if pst is not None and pst.get(qn("w:val")) == "framed-note":
        pbdr = el("w:pBdr")
        for side in ("top", "left", "bottom", "right"):
            pbdr.append(el(f"w:{side}", val="single", sz="6",
                           space="4", color="993333"))
        ppr.append(pbdr)

for tbl in body.iter(qn("w:tbl")):
    tblpr = tbl.find(qn("w:tblPr"))
    if tblpr is None or tblpr.find(qn("w:tblBorders")) is not None:
        continue
    borders = el("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(el(f"w:{side}", val="single", sz="4", space="0",
                          color="000000"))
    tblpr.append(borders)

for st_el in doc.styles.element.iter(qn("w:style")):
    if st_el.get(qn("w:styleId")) == "Heading1":
        ppr = st_el.find(qn("w:pPr"))
        if ppr is None:
            ppr = el("w:pPr"); st_el.append(ppr)
        if ppr.find(qn("w:pageBreakBefore")) is None:
            ppr.append(el("w:pageBreakBefore"))

doc.save(OUT)
print("saved", OUT)
print("figures:", fig_list)
print("tables :", len(tab_list))
print("ref paras moved:", len(refs_paras))
