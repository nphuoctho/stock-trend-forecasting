# Mẫu LaTeX Đồ án tốt nghiệp

Template LaTeX cho đồ án, tuân thủ Phụ lục 2 (hình thức trình bày) của trường:
Times New Roman 13pt, giãn dòng 1.5, lề T3/D3.5/L3.5/P2 cm, đánh số chương/mục,
tài liệu tham khảo IEEE.

## Cấu trúc

```
main.tex              file chính (thứ tự bắt buộc: bìa -> ... -> phụ lục)
preamble.tex          cấu hình font/lề/spacing theo quy định trường
frontmatter/          bìa chính, bìa phụ, hội đồng, lời cảm ơn, danh mục, tóm tắt
chapters/             6 chương: mở đầu, tổng quan, phương pháp, kết quả,
                      kết luận, hướng phát triển + phụ lục
references.bib        tài liệu tham khảo (IEEE, biber)
```

## Biên dịch

Cần XeLaTeX (font Unicode + tiếng Việt). Cách 1 - latexmk:

```bash
latexmk -xelatex main.tex
```

Cách 2 - thủ công:

```bash
xelatex main && biber main && xelatex main && xelatex main
```

Cách 3 - tectonic (tự tải package):

```bash
tectonic -X compile main.tex   # hoặc: tectonic main.tex
```

Cách 4 - Overleaf: upload cả thư mục, chọn compiler XeLaTeX.

## Lưu ý

- Font: nếu máy không có "Times New Roman", preamble tự fallback sang
  "Liberation Serif" (metric-compatible). Trên Overleaf có sẵn TNR.
- Chèn sơ đồ kiến trúc thật: đặt PNG vào figures/ và sửa \includegraphics
  trong chapters/03-phuong-phap.tex (đang để khung placeholder).
- Các bảng kết quả (chương 4) đang là khung mẫu, điền số thực tế sau khi chạy thực nghiệm.
