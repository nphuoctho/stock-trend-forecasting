# Báo cáo đồ án tốt nghiệp bằng LaTeX

Nguồn báo cáo tuân thủ Phụ lục 2 về hình thức trình bày: Times New Roman 13 pt,
giãn dòng 1,5; lề trên 3 cm, dưới 3,5 cm, trái 3,5 cm, phải 2 cm; tài liệu tham
khảo theo chuẩn IEEE.

## Cấu trúc

```text
BaoCaoDATN.tex       nguồn LaTeX hoàn chỉnh và duy nhất
references.bib       cơ sở dữ liệu tài liệu tham khảo
latexmkrc            cấu hình đưa toàn bộ kết quả biên dịch vào build/
build/                PDF, log và các tệp trung gian
```

`BaoCaoDATN.tex` chứa toàn bộ cấu hình trình bày, phần đầu báo cáo, sáu chương,
tài liệu tham khảo và phụ lục. Không duy trì bản `full`, bản `progress` hoặc các
tệp chương phân mảnh.

## Biên dịch

Cần XeLaTeX và Biber. Chạy từ `docs/thesis-latex/`:

```bash
latexmk -xelatex BaoCaoDATN.tex
```

PDF được tạo tại `build/BaoCaoDATN.pdf`. Toàn bộ `.aux`, `.bcf`, `.bbl`, `.toc`,
`.lof`, `.lot`, `.out`, `.run.xml`, `.blg`, `.log` và các tệp do LaTeX sinh ra
cũng nằm trong `build/`.

Nếu không có `latexmk`, dùng:

```bash
mkdir -p build
xelatex -output-directory=build BaoCaoDATN.tex
biber build/BaoCaoDATN
xelatex -output-directory=build BaoCaoDATN.tex
xelatex -output-directory=build BaoCaoDATN.tex
```

Overleaf: tải lên `BaoCaoDATN.tex` và `references.bib`, sau đó chọn XeLaTeX.

## Lưu ý

- Nếu máy không có Times New Roman, cấu hình tự chuyển sang Liberation Serif.
- Hai sơ đồ trong báo cáo được vẽ trực tiếp bằng TikZ.
- Các số liệu thực nghiệm trong Chương 4 phải được đối chiếu với artifact tương
  ứng trong `outputs/` trước khi sửa.
