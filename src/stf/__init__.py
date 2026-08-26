"""stf - Stock Trend Forecasting.

Hệ thống dự báo xu hướng giá cổ phiếu HOSE/VN30 dựa trên phân tích cảm xúc
tin tức tài chính (PhoBERT) kết hợp mô hình chuỗi thời gian.

Các package con:
  stf.config    cấu hình tập trung (mã, cửa sổ thời gian, đường dẫn).
  stf.data      thu thập dữ liệu giá và tin tức có timestamp.
  stf.sentiment fine-tune PhoBERT và sinh xác suất cảm xúc 3 lớp.
"""

__version__ = "0.1.0"
