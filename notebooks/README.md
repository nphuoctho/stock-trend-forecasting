# Notebook fine-tune PhoBERT (Colab / Kaggle)

Fine-tune `vinai/phobert-base` thành mô hình phân loại cảm xúc 3 lớp
(NEGATIVE / NEUTRAL / POSITIVE) cho tin tài chính tiếng Việt.

## File
- `phobert_finetune_colab.ipynb` — notebook fine-tune, chạy trực tiếp trên Colab/Kaggle.

## Cách chạy

### Google Colab
1. Upload `phobert_finetune_colab.ipynb` lên Colab.
2. Runtime > Change runtime type > **T4 GPU**.
3. Chạy lần lượt từ mục 1. Nếu sau mục 2 Colab báo cần restart, restart rồi chạy tiếp
   TỪ mục 3 (không chạy lại mục 2).

### Kaggle
1. New Notebook > Upload `phobert_finetune_colab.ipynb`.
2. Settings > Accelerator > **GPU** (T4 x2 hoặc P100), và bật **Internet**.
3. Chạy lần lượt.

## Dữ liệu

- **Seed CafeF** (999 tiêu đề, gán nhãn sẵn): notebook tự tải từ repo công khai
  `209sontung/Vietnamese-stock-article-classification`.
- **In-domain (tuỳ chọn):** nếu đã gán nhãn tập in-domain (từ script
  `stf.sentiment.make_indomain_sample`), lưu thành `indomain_labeled.csv`
  (cột `text`, `label`) và upload lên notebook. Notebook tự gộp.

## Lưu ý version (tránh lỗi)

Notebook pin `transformers==4.46.3`, `numpy<2`, `pandas<2.3` để tương thích ổn định
với torch+CUDA có sẵn của Colab/Kaggle. KHÔNG cài lại torch. Nếu Kaggle báo xung đột
phụ thuộc, bật Internet và chạy lại mục 2, hoặc dùng `--no-deps` cho transformers.

## Đầu ra

- Mô hình + tokenizer lưu tại `phobert-sentiment-best/`.
- Báo cáo macro-F1 + per-class trên tập test.
- Hàm `predict_proba` để sinh xác suất 3 lớp cho corpus tin (bước inference tiếp theo).
