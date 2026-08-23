# Dự báo xu hướng giá cổ phiếu từ cảm xúc tin tài chính - Stock Trend Forecasting

**TÊN ĐỀ TÀI:** Xây dựng hệ thống dự báo xu hướng biến động giá cổ phiếu dựa trên phân tích cảm xúc tin tức tài chính bằng mô hình Transformer và chuỗi thời gian

**TÊN ĐỀ TÀI (tiếng Anh):** Building a stock price movement forecasting system based on financial news sentiment analysis using Transformer and time-series models

**Cán bộ hướng dẫn:** TS. Đặng Văn Thìn

**Thời gian thực hiện:** Từ ngày 15/07/2026 đến ngày 23/09/2026

**Sinh viên thực hiện:** Nguyễn Phước Thọ – 25410139 – Lớp LT.K2025.2.TTNT

## Môi trường

- Python 3.12, managed by `uv`. Tạo/cập nhật môi trường bằng `uv sync`.
- Chạy script bằng `uv run`, ví dụ: `uv run python src/data/spike_prices.py`.

## Cấu trúc

- `src/data/` - kéo giá (`spike_prices.py`) + tin Vietstock (`spike_news_vietstock.py`).
- `data/` - dữ liệu thô/xử lý (gitignore; tái tạo bằng script).
