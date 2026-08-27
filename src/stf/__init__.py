"""stf - Stock Trend Forecasting.

Forecasts HOSE/VN30 price trends by combining financial-news sentiment
(PhoBERT) with a time-series model.

Sub-packages:
  stf.config    central config (tickers, date window, paths).
  stf.data      collect price data and timestamped news.
  stf.sentiment fine-tune PhoBERT and produce 3-class sentiment probabilities.
"""

__version__ = "0.1.0"
