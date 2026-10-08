# /// script
# dependencies = ["python-docx"]
# ///
"""Post-process pandoc-generated thesis docx.

Input : build/thesis-raw.docx  (pandoc output; refs land at document end)
Output: BaoCaoDATN_25410139_NguyenPhuocTho.docx

Steps:
 1. Insert 2 cover pages using the distinct official BieuMau.docx borders.
 2. Section breaks: covers / front matter (no page numbers) / body
    (page numbering restarts at 1, centered PAGE footer).
 3. Number headings: "Chương N." / "N.M." / appendix "PHỤ LỤC A", "A.1".
 4. Bold "Hình x.y:" / "Bảng x.y:" caption prefixes; build LOF/LOT.
 5. Move bibliography before appendix, add TÀI LIỆU THAM KHẢO heading.
 6. framed-note -> bordered paragraph; tables -> thin borders;
    all heading styles -> Times New Roman; Heading1 -> page-break-before.
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
AUX = sys.argv[3] if len(sys.argv) > 3 else "../thesis-latex/build/BaoCaoDATN.aux"
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
                      hAnsi="Times New Roman", eastAsia="Times New Roman",
                      cs="Times New Roman"))
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
    """A4 body layout from Appendix 2 and the body sections of BieuMau.docx."""
    sp = el("w:sectPr")
    sp.append(el("w:pgSz", w="11906", h="16838"))
    sp.append(el("w:pgMar", top="1701", right="1134", bottom="1985",
                 left="1985", header="708", footer="708", gutter="0"))
    return sp


def cover_sect_pr(main_cover):
    """Cover margins and page-border styles copied from BieuMau.docx."""
    sp = el("w:sectPr")
    sp.append(el("w:pgSz", w="11906", h="16838", orient="portrait"))
    sp.append(el("w:pgMar", top="1135", right="1133", bottom="851",
                 left="1440", header="708", footer="708", gutter="0"))
    attrs = {"offsetFrom": "page"}
    if main_cover:
        attrs["display"] = "firstPage"
    borders = el("w:pgBorders", **attrs)
    for side in ("top", "left", "bottom", "right"):
        if main_cover:
            value = "thinThickSmallGap" if side in ("top", "left") \
                else "thickThinSmallGap"
            size = "24"
        else:
            value, size = "double", "4"
        borders.append(el(f"w:{side}", val=value, color="auto", sz=size,
                          space="24"))
    sp.append(borders)
    return sp

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


def make_blank_header_part(name):
    xml = (b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           b'<w:hdr xmlns:w="http://schemas.openxmlformats.org/'
           b'wordprocessingml/2006/main"><w:p/></w:hdr>')
    return Part(PackURI(f"/word/{name}.xml"),
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.header+xml",
                xml, doc.part.package)


def add_header_ref(sect, rid):
    """headerReference must precede footerReference and pgSz in sectPr."""
    hr = OxmlElement("w:headerReference")
    hr.set(qn("w:type"), "default")
    hr.set(qn("r:id"), rid)
    sect.insert(0, hr)

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
def blanks(count):
    return [new_p("", after=120) for _ in range(count)]


def cover(with_advisor):
    first_size = 14 if with_advisor else 15
    ps = [
        new_p("ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH", first_size, bold=True,
              after=120),
        new_p("TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN", 16, bold=True,
              after=120),
        new_p("KHOA KHOA HỌC MÁY TÍNH", 16, bold=True, after=120),
    ]
    ps.extend(blanks(2))
    name = "NGUYỄN PHƯỚC THỌ" + (" – 25410139" if with_advisor else "")
    ps.append(new_p(name, 14, bold=True, after=120))
    ps.extend(blanks(2))
    ps.extend([
        new_p("ĐỒ ÁN TỐT NGHIỆP", 16, bold=True, after=120),
        new_p(TITLE_VN, 18, bold=True, after=120),
        new_p(TITLE_EN, 16, bold=True, after=120),
    ])
    ps.extend(blanks(1))
    ps.append(new_p("CỬ NHÂN NGÀNH TRÍ TUỆ NHÂN TẠO", 14, bold=True,
                    after=120))
    if with_advisor:
        ps.extend(blanks(1))
        ps.append(new_p("GIẢNG VIÊN HƯỚNG DẪN", 14, bold=True, after=120))
        ps.append(new_p("TS. Đặng Văn Thìn", 14, bold=True, after=120))
        ps.extend(blanks(1))
    else:
        ps.extend(blanks(4))
    ps.append(new_p("TP. HỒ CHÍ MINH, 2026", 13, bold=True, after=120))
    return ps

# footers
rid_blank = doc.part.relate_to(make_footer_part("footerBlank"), RT.FOOTER)
rid_page = doc.part.relate_to(make_footer_part("footerPage", field=True),
                              RT.FOOTER)
rid_header_blank = doc.part.relate_to(
    make_blank_header_part("headerBlank"), RT.HEADER)


def idx_of_text(prefix, start=0):
    for i, c in enumerate(children):
        if i < start or c.tag != qn("w:p"):
            continue
        if para_text(c).strip().startswith(prefix):
            return i
    return -1

# Two one-page cover sections reproduce the distinct borders in BieuMau.docx.
sec_main_cover = cover_sect_pr(main_cover=True)
add_footer_ref(sec_main_cover, rid_blank)
add_header_ref(sec_main_cover, rid_header_blank)
main_cover = cover(False)
main_cover[-1].find(qn("w:pPr")).append(sec_main_cover)

sec_secondary_cover = cover_sect_pr(main_cover=False)
add_footer_ref(sec_secondary_cover, rid_blank)
add_header_ref(sec_secondary_cover, rid_header_blank)
secondary_cover = cover(True)
secondary_cover[-1].find(qn("w:pPr")).append(sec_secondary_cover)

first_p = body.find(qn("w:p"))
for p in main_cover + secondary_cover:
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

# Front matter: Appendix 2 body margins, no page number.
sec_front = sect_pr()
add_footer_ref(sec_front, rid_blank)
add_header_ref(sec_front, rid_header_blank)
i_summary = idx_of_text("TÓM TẮT ĐỒ ÁN")
children[i_summary].addprevious(sect_break_par(sec_front))
children = list(body)

# Main section: same body layout; page numbering restarts at the Vietnamese summary.
main_sect = body.find(qn("w:sectPr"))
main_pg_sz = main_sect.find(qn("w:pgSz"))
if main_pg_sz is None:
    main_pg_sz = el("w:pgSz")
    main_sect.append(main_pg_sz)
main_pg_sz.set(qn("w:w"), "11906")
main_pg_sz.set(qn("w:h"), "16838")
main_pg_mar = main_sect.find(qn("w:pgMar"))
if main_pg_mar is None:
    main_pg_mar = el("w:pgMar")
    main_sect.append(main_pg_mar)
for attr, value in {
    "top": "1701", "right": "1134", "bottom": "1985", "left": "1985",
    "header": "708", "footer": "708", "gutter": "0",
}.items():
    main_pg_mar.set(qn(f"w:{attr}"), value)
add_footer_ref(main_sect, rid_page)
add_header_ref(main_sect, rid_header_blank)
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

# drop literal reference-container marker paragraphs emitted by preprocessing
for c in list(body.iter(qn("w:p"))):
    marker = para_text(c).strip()
    if marker in (":::", "#refs") or marker.startswith("::: #refs"):
        c.getparent().remove(c)

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
    if tblpr is not None and tblpr.find(qn("w:tblBorders")) is None:
        borders = el("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            borders.append(el(f"w:{side}", val="single", sz="4", space="0",
                              color="000000"))
        tblpr.append(borders)

    for index, row in enumerate(tbl.findall(qn("w:tr"))):
        trpr = row.find(qn("w:trPr"))
        if trpr is None:
            trpr = el("w:trPr")
            row.insert(0, trpr)
        if index == 0 and trpr.find(qn("w:tblHeader")) is None:
            trpr.append(el("w:tblHeader", val="true"))
        if trpr.find(qn("w:cantSplit")) is None:
            trpr.append(el("w:cantSplit"))

def set_run_font(parent, name):
    rpr = parent.find(qn("w:rPr"))
    if rpr is None:
        rpr = el("w:rPr")
        parent.insert(0, rpr)
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = el("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{attr}"), name)


for st_el in doc.styles.element.iter(qn("w:style")):
    style_id = st_el.get(qn("w:styleId"), "")
    if not re.fullmatch(r"(?:Heading[1-9](?:Char)?|TOCHeading)", style_id):
        continue
    set_run_font(st_el, "Times New Roman")
    if style_id == "Heading1":
        ppr = st_el.find(qn("w:pPr"))
        if ppr is None:
            ppr = el("w:pPr"); st_el.append(ppr)
        if ppr.find(qn("w:pageBreakBefore")) is None:
            ppr.append(el("w:pageBreakBefore"))

for p in body.iter(qn("w:p")):
    if not re.fullmatch(r"(?:Heading[1-9]|TOCHeading)", style_of(p) or ""):
        continue
    for run in p.iter(qn("w:r")):
        set_run_font(run, "Times New Roman")

doc.save(OUT)
print("saved", OUT)
print("figures:", fig_list)
print("tables :", len(tab_list))
print("ref paras moved:", len(refs_paras))
