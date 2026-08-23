# Dự báo xu hướng giá cổ phiếu từ cảm xúc tin tài chính (PhoBERT + chuỗi thời gian)

Đồ án tốt nghiệp — Nguyễn Phước Thọ, UIT-VNU HCM. GVHD: TS. Đặng Văn Thìn.

Kế hoạch & báo cáo: `../plans/260811-1512-stock-trend-sentiment-system/plan.md`.

## Môi trường
- Python 3.11 (venv qua `uv`). Cài: `uv pip install -r requirements.txt` (thêm sau).
- Chạy script: `uv run python <path>`.

## Cấu trúc
- `src/data/` — kéo giá (`spike_prices.py`) + tin Vietstock (`spike_news_vietstock.py`).
- `data/` — dữ liệu thô/xử lý (gitignore; tái tạo bằng script).

## Trạng thái
Tuần A (15/08/2026): spike khử rủi ro **ĐẠT** — xem `../plans/reports/spike-260815-1935-data-derisk.md`.
