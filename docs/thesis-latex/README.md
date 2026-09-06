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
build/                          các file trung gian khi biên dịch
BaoCaoDATN_*.pdf                PDF đầu ra, nằm cạnh các file .tex
```

## Biên dịch

Cần XeLaTeX (font Unicode + tiếng Việt). `latexmkrc` đã cấu hình để các file
trung gian nằm trong `build/`, còn PDF cuối cùng nằm cạnh file `.tex`.

Chạy từ thư mục `docs/thesis-latex/`:

```bash
latexmk -xelatex BaoCaoDATN_full.tex
```

Chỉ có bản báo cáo tiến độ ngoài bản đầy đủ:

```bash
latexmk -xelatex BaoCaoDATN_progress.tex
```

PDF sẽ nằm cạnh file nguồn, còn các file phụ sẽ nằm trong `build/`. Dọn toàn
bộ output khi cần:

```bash
latexmk -C BaoCaoDATN_full.tex
```

Cách thủ công (không dùng latexmk):

```bash
mkdir -p build
xelatex -output-directory=build BaoCaoDATN_full.tex
biber --output-directory=build build/BaoCaoDATN_full
xelatex -output-directory=build BaoCaoDATN_full.tex
xelatex -output-directory=build BaoCaoDATN_full.tex
mv build/BaoCaoDATN_full.pdf .
```

Cách 3 - tectonic (tự tải package):

```bash
tectonic --outdir build BaoCaoDATN_full.tex
mv build/BaoCaoDATN_full.pdf .
```

Cách 4 - Overleaf: upload cả thư mục, chọn compiler XeLaTeX.


## Lưu ý

- Font: nếu máy không có "Times New Roman", preamble tự fallback sang
  "Liberation Serif" (metric-compatible). Trên Overleaf có sẵn TNR.
- Chèn sơ đồ kiến trúc thật: đặt PNG vào figures/ và sửa \includegraphics
  trong chapters/03-phuong-phap.tex (đang để khung placeholder).
- Các bảng kết quả (chương 4) đang là khung mẫu, điền số thực tế sau khi chạy thực nghiệm.
