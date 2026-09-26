# Mẫu LaTeX Đồ án tốt nghiệp

Template LaTeX cho đồ án, tuân thủ Phụ lục 2 (hình thức trình bày) của trường:
Times New Roman 13pt, giãn dòng 1.5, lề T3/D3.5/L3.5/P2 cm, đánh số chương/mục,
tài liệu tham khảo IEEE.

## Cấu trúc

```
BaoCaoDATN_full.tex             bản đầy đủ, có sơ đồ
BaoCaoDATN_progress.tex         bản báo cáo tiến độ
preamble.tex                    cấu hình font/lề/spacing theo quy định trường
frontmatter/                    bìa chính, bìa phụ, hội đồng, lời cảm ơn, danh mục, tóm tắt
chapters/                       các chương của báo cáo
references.bib                  tài liệu tham khảo (IEEE, biber)
build/                          file trung gian (chỉ dùng khi build bằng latexmk; xelatex thủ công ghi cạnh .tex)
BaoCaoDATN_*.pdf                PDF đầu ra, nằm cạnh các file .tex
```

## Biên dịch

Cần XeLaTeX (font Unicode + tiếng Việt) và biber. Trên máy này không có `latexmk`;
biên dịch thủ công theo trình tự `xelatex` -> `biber` -> `xelatex` -> `xelatex`
trên `BaoCaoDATN_full.tex`, chạy từ thư mục `docs/thesis-latex/`:

```bash
xelatex BaoCaoDATN_full.tex
biber BaoCaoDATN_full
xelatex BaoCaoDATN_full.tex
xelatex BaoCaoDATN_full.tex
```

Bản báo cáo tiến độ build tương tự:

```bash
xelatex BaoCaoDATN_progress.tex
biber BaoCaoDATN_progress
xelatex BaoCaoDATN_progress.tex
xelatex BaoCaoDATN_progress.tex
```

Các file trung gian (`.aux`, `.bcf`, `.bbl`, `.toc`, `.lof`, `.lot`, `.out`,
`.run.xml`, `.blg`) nằm ngay cạnh file `.tex` trong `docs/thesis-latex/`; PDF đầu
ra cũng nằm cạnh file nguồn.

Nếu ở một máy khác có `latexmk`, `latexmkrc` vẫn được giữ để các file trung gian
nằm trong `build/` còn PDF cuối cùng nằm cạnh file `.tex`:

```bash
latexmk -xelatex BaoCaoDATN_full.tex
```

Với `latexmk`, dọn toàn bộ output bằng:

```bash
latexmk -C BaoCaoDATN_full.tex
```

Overleaf: upload cả thư mục, chọn compiler XeLaTeX.


## Lưu ý

- Font: nếu máy không có "Times New Roman", preamble tự fallback sang
  "Liberation Serif" (metric-compatible). Trên Overleaf có sẵn TNR.
- Sơ đồ kiến trúc chương 3 đã được vẽ trực tiếp bằng TikZ trong
  `chapters/03-phuong-phap.tex` (điều kiện `\ifShowDiagram`); file
  `figures/architecture.png` chỉ là bản xuất tham khảo, không được chèn vào báo cáo.
- Các bảng kết quả chương 4 (`chapters/04-ket-qua.tex`) đã được điền số liệu thực
  nghiệm; mọi con số cần truy về artifact trong `outputs/` trước khi sửa.

