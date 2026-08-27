# Hướng dẫn chạy notebook fine-tune PhoBERT trên Colab / Kaggle

Hướng dẫn từng bước để fine-tune PhoBERT sinh mô hình cảm xúc 3 lớp. Làm theo đúng thứ tự,
mỗi mục là một phần trong notebook `phobert_finetune_colab.ipynb`.

---

## PHẦN A — Google Colab (khuyến nghị cho người mới)

### Bước 1: Mở notebook
1. Vào https://colab.research.google.com
2. File > Upload notebook > chọn `phobert_finetune_colab.ipynb`.

### Bước 2: Bật GPU (BẮT BUỘC)
1. Menu **Runtime > Change runtime type**.
2. Hardware accelerator > chọn **T4 GPU** > Save.
3. Nếu không bật, mục 3 sẽ báo lỗi `assert torch.cuda.is_available()`.

### Bước 3: Chạy lần lượt từng cell
Bấm nút Play (hoặc Shift+Enter) từng cell theo thứ tự:

| Mục | Việc | Kỳ vọng |
|---|---|---|
| 1 | Kiểm tra GPU | In ra "GPU: Tesla T4, 15360 MiB" |
| 2 | Cài thư viện | ~1-2 phút. Có thể hiện nút "RESTART SESSION" |
| — | **NẾU hiện RESTART:** bấm restart, rồi chạy TIẾP TỪ MỤC 3 (bỏ qua mục 1-2) | |
| 3 | Xác nhận version | transformers 4.46.3, CUDA: True |
| 4 | Tải seed CafeF | "Đã tải seed CafeF" |
| 5 | Chuẩn bị nhãn | "Chỉ dùng seed CafeF: 999 mẫu", phân bố {0:186,1:249,2:564} |
| 6 | Chia train/val/test | train≈799 val≈100 test≈100 |
| 7 | Tokenize | Tải tokenizer PhoBERT, in "Đã tokenize..." |
| 8 | Nạp mô hình + tạo RUN_DIR | Tải phobert-base (~500MB), in RUN_DIR, cảnh báo "newly initialized" là BÌNH THƯỜNG |
| 9 | **Huấn luyện** | ~5-10 phút trên T4, hiện bảng loss/macro_f1 mỗi epoch; lưu train_log_history.csv |
| 10 | Đánh giá test + lưu report | macro-F1 + per-class + confusion matrix (png/csv) lưu vào RUN_DIR |
| 11 | Lưu model + manifest | model + manifest.json (version, seed, siêu tham số, metric) |
| 12 | Đóng gói zip | Tạo file zip toàn bộ RUN để tải về |
| 13 | Thử inference | In ma trận xác suất 2x3 |

### Bước 4: Lưu kết quả (QUAN TRỌNG - đừng mất session mà mất số)
Notebook tự lưu MỌI artifact vào thư mục `runs/phobert-sentiment-<timestamp>/`:
- `train_log.jsonl` — GHI TỪNG BƯỚC ngay khi train (loss/lr/metric). An toàn: crash giữa chừng vẫn còn số.
- `train_log_history.csv` — loss/metric từng epoch (để debug + vẽ đồ thị học).
- `learning_curve.png` — đường cong train loss + val macro-F1 (chèn thẳng chương 4, xem có hội tụ / overfit không).
- `test_classification_report.txt` + `test_report.json` — chỉ số để trích báo cáo.
- `confusion_matrix.png` + `.csv` — chèn thẳng vào chương 4.
- `misclassified.csv` — các ca dự đoán SAI (text/true/pred/xác suất từng lớp/độ tự tin), sắp theo độ tự tin giảm dần để soi lỗi nặng nhất. Dùng viết phần "phân tích lỗi định tính".
- `test_predictions.csv` — TOÀN BỘ dự đoán test (tái phân tích sau mà không cần chạy lại model).
- `manifest.json` — version, seed, siêu tham số, metric (bằng chứng tái lập, acceptance #8).
- `best/` — mô hình + tokenizer.
- `checkpoints/` — checkpoint mỗi epoch (giữ 3 bản gần nhất) để RESUME.
- `logs/` — TensorBoard log.

**Để KHÔNG mất khi hết session, chọn 1 trong 2:**
- **Cách tốt nhất (Colab):** bỏ chú thích 3 dòng mount Drive ở đầu mục 8, đổi `BASE_OUT`
  thành đường dẫn Drive. Mọi thứ lưu thẳng lên Drive.
- **Cách nhanh:** ở mục 12, bỏ chú thích `files.download(zip_path)` (Colab) để tải zip về máy.
  Kaggle: file zip tự xuất hiện ở tab Output, bấm tải.

### Bước 4b: RESUME khi Colab/Kaggle bị ngắt giữa chừng
Nếu session chết lúc đang train (hết giờ free, mất mạng), KHÔNG phải train lại từ đầu:
1. Đảm bảo thư mục RUN cũ còn (đã mount Drive, hoặc còn trong session Kaggle chưa tắt).
2. Mở lại notebook, chạy mục 1-7 như thường (tải lại thư viện + dữ liệu, phải GIỮ NGUYÊN seed/siêu tham số).
3. Ở **mục 8**, điền `RESUME_RUN_DIR = "runs/phobert-sentiment-<timestamp cũ>"` (đường dẫn RUN cũ).
4. Chạy tiếp mục 8-9. Trainer tự nạp checkpoint epoch gần nhất và train tiếp. Log ghi nối tiếp vào cùng RUN_DIR.

---

## PHẦN B — Kaggle

### Bước 1: Tạo notebook
1. Vào https://www.kaggle.com/code > New Notebook.
2. File > Import Notebook > upload `phobert_finetune_colab.ipynb`.

### Bước 2: Bật GPU + Internet
1. Panel bên phải > **Settings**.
2. Accelerator > **GPU T4 x2** (hoặc P100).
3. Internet > **On** (cần để tải model + seed).

### Bước 3: Chạy tương tự Colab (mục 1-12).
Kaggle hiếm khi cần restart sau cài đặt. Nếu báo xung đột phụ thuộc ở mục 2, cứ chạy tiếp
mục 3 để kiểm tra version; thường vẫn OK.

---

## XỬ LÝ SỰ CỐ THƯỜNG GẶP

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `assert torch.cuda.is_available()` fail ở mục 3 | Chưa bật GPU | Bật GPU (Bước 2) rồi chạy lại từ mục 1 |
| Mục 2 báo cần RESTART | Colab cập nhật numpy/pandas | Restart, chạy tiếp TỪ mục 3 (không chạy lại mục 2) |
| `NameError: torch not defined` | Chạy lẻ cell không theo thứ tự | Chạy lại từ mục 3 |
| Mục 8 cảnh báo "newly initialized weights" | Bình thường (head phân loại mới) | Bỏ qua, đây là điều đúng khi fine-tune |
| Train quá chậm (>30 phút) | Đang chạy CPU không phải GPU | Kiểm tra mục 3 phải in "CUDA: True" |
| `wget` mục 4 lỗi | Kaggle chưa bật Internet | Bật Internet trong Settings |
| macro-F1 thấp (<0.5) | Bình thường với seed nhỏ lệch lớp | Thêm nhãn in-domain, hoặc chấp nhận + báo cáo trung thực |

---

## SAU KHI TRAIN XONG

1. **Ghi lại con số macro-F1** (mục 10) vào journal/plan (plan.md yêu cầu "nêu được con số thí nghiệm gần nhất").
2. **Lưu mô hình** về Drive hoặc tải zip (mục 11).
3. **Bước tiếp theo:** dùng mô hình sinh xác suất cảm xúc cho toàn corpus tin (mục 12 mở rộng):
   upload `articles.parquet` lên, chạy `predict_proba` trên cột title (hoặc title+body),
   lưu kết quả để dùng ở Phase 3 (đặc trưng mã-ngày).

## Nâng chất lượng (tuỳ chọn, nếu còn thời gian)
- **Gán nhãn in-domain:** chạy `stf.sentiment.make_indomain_sample` ở máy, gán tay ~300 bài
  theo guideline, lưu `indomain_labeled.csv`, upload lên notebook (mục 5 tự gộp).
- **Word segmentation:** cài `py_vncorenlp` để tách từ trước tokenize (PhoBERT vốn train trên
  văn bản đã tách từ), thường cải thiện macro-F1.
