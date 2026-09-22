#!/usr/bin/env bash
# Daily forecast job: refresh prices + news, score new articles, predict, resolve.
#
# Schedule after the HOSE close (15:00 ICT), e.g. cron:
#   30 15 * * 1-5  /path/to/stock-trend-forecasting/scripts/daily-forecast.sh
#
# Required env:
#   SENTIMENT_MODEL   checkpoint dir (default: models/sentiment/merged-refit/best)
#   SCORED_NEWS       score-news parquet (default: data/processed/news_sentiment_merged.parquet)
set -euo pipefail
cd "$(dirname "$0")/.."

SENTIMENT_MODEL="${SENTIMENT_MODEL:-models/sentiment/merged-refit/best}"
SCORED_NEWS="${SCORED_NEWS:-data/processed/news_sentiment_merged.parquet}"
ARMS="${ARMS:-lstm_price_sentiment lstm_price logreg_price_sentiment}"

TODAY="$(date +%F)"

echo "[daily] $(date -Is) fetching prices"
uv run python -m stf.cli prices --end "$TODAY"

echo "[daily] $(date -Is) crawling news"
# --refresh until listings carry a per-year coverage watermark: without one,
# a partially walked year is indistinguishable from a complete one and would
# be frozen the moment the calendar rolls over.
uv run python -m stf.cli news --refresh --end "$TODAY"

echo "[daily] $(date -Is) scoring new articles"
uv run python -m stf.cli score-news \
  --model-dir "$SENTIMENT_MODEL" \
  --input-variant title_context \
  --incremental \
  --output "$SCORED_NEWS"

for arm in $ARMS; do
  model_dir="models/forecast/$arm"
  live_dir="outputs/live/$arm"
  extra=()
  case "$arm" in
    *_sentiment) extra=(--news-sentiment "$SCORED_NEWS") ;;
  esac
  if [ ! -f "$model_dir/manifest.json" ]; then
    echo "[daily] refitting missing arm $arm"
    uv run python -m stf.cli forecast-refit \
      --arm "$arm" --model-dir "$model_dir" "${extra[@]}"
  fi
  echo "[daily] $(date -Is) predicting with $arm"
  uv run python -m stf.cli forecast-predict \
    --model-dir "$model_dir" --output-dir "$live_dir" "${extra[@]}"
  uv run python -m stf.cli forecast-resolve \
    --model-dir "$model_dir" --predictions-dir "$live_dir" "${extra[@]}"
done

echo "[daily] $(date -Is) done"
