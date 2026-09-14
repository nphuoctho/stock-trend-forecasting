# Project context

## Trạng thái hiện tại

- Phạm vi dữ liệu giá: 10 mã VN30, từ 2020-01-01 đến 2025-12-31.
- Dữ liệu tin Vietstock đã thu thập: `data/raw/news/articles.parquet` gồm 10.695 bài; các cột `title` và `body` hiện có dữ liệu.
- Tập CafeF đã chuẩn hóa: 999 mẫu, chỉ có tiêu đề và nhãn ba lớp.
- Mẫu Vietstock nội miền tại `data/labeled/indomain/to_label.csv` có 300 dòng nhưng cột `label` còn trống; chưa thể huấn luyện hoặc đánh giá nội miền.
- Mã PhoBERT hỗ trợ ba biến thể `title`, `context`, `title_context`, cùng ba chiến lược cắt token `head`, `tail`, `head_tail` ở giới hạn 256 token.
- Đã thêm `sentiment-cv` cho stratified 5-fold CV và `sentiment-ablation` cho ma trận 3 x 3. Outer holdout không dùng chọn checkpoint; validation 10% nằm trong outer train.

## Việc tiếp theo

1. Gán nhãn nội miền Vietstock theo `data/labeled/indomain/HUONG_DAN_GAN_NHAN.md`.
2. Chạy từng cấu hình 5-fold trên GPU và lưu các tệp `cv_results.json`/`cv_results.csv`.
3. Chọn cấu hình theo macro-F1 trung bình, kiểm tra độ lệch chuẩn và F1 từng lớp.
4. Dùng checkpoint tốt nhất để suy luận tin Vietstock, sau đó mới tổng hợp đặc trưng theo mã và ngày.

## Nhật ký

### 2026-09-14

- Bổ sung tiền xử lý đầu vào title/context và cắt token đúng ở cấp token.
- Bổ sung cross-validation phân tầng và chạy ablation có lưu kết quả.
- Kiểm thử offline: 18 kiểm thử đạt.
- Chưa chạy huấn luyện PhoBERT thực tế: nhãn nội miền chưa hoàn tất và máy phát triển không có CUDA; không ghi nhận số đo mô hình.
