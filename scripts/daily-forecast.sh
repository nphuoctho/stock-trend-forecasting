#!/usr/bin/env bash
# Daily forecast job: refresh prices + news, score new articles, predict, resolve.
#
# Schedule after the HOSE close (15:00 ICT), e.g. cron:
#   30 15 * * 1-5  /path/to/stock-trend-forecasting/scripts/daily-forecast.sh
#
# Required env:
#   SENTIMENT_MODEL   checkpoint dir (default: models/sentiment/merged-refit/best)
#   SCORED_NEWS       score-news parquet (default: data/processed/news_sentiment_merged.parquet)
#
# Every run appends to logs/daily-YYYYMMDD.log and writes its outcome to
# outputs/live/last_run.json, which the dashboard reads via /api/live/status.
set -euo pipefail
cd "$(dirname "$0")/.."

SENTIMENT_MODEL="${SENTIMENT_MODEL:-models/sentiment/merged-refit/best}"
SCORED_NEWS="${SCORED_NEWS:-data/processed/news_sentiment_merged.parquet}"
ARMS="${ARMS:-lstm_price_sentiment lstm_price logreg_price_sentiment tft_price_sentiment tft_price}"

TODAY="$(date +%F)"
mkdir -p logs outputs/live
exec > >(tee -a "logs/daily-${TODAY}.log") 2>&1

STEP="init"
write_status() {
  local rc="$1"
  STEP="$STEP" RC="$rc" uv run python - <<'PY'
import json, os
from datetime import datetime, timezone
status = {
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "ok": os.environ["RC"] == "0",
    "exit_code": int(os.environ["RC"]),
    "failed_step": None if os.environ["RC"] == "0" else os.environ["STEP"],
}
with open("outputs/live/last_run.json", "w", encoding="utf-8") as fh:
    json.dump(status, fh, ensure_ascii=False, indent=2)
PY
}
trap 'rc=$?; write_status "$rc"; exit $rc' EXIT

echo "[daily] $(date -Is) fetching prices"
STEP="prices"
uv run python -m stf.cli prices --end "$TODAY"

echo "[daily] $(date -Is) crawling news"
# No --refresh: listings record a per-year coverage watermark, so a year is
# reused only when it was walked through its full extent. --end advances daily,
# which re-walks the current year and leaves the archive alone.
STEP="news"
uv run python -m stf.cli news --end "$TODAY"

echo "[daily] $(date -Is) scoring new articles"
STEP="score-news"
uv run python -m stf.cli score-news \
  --model-dir "$SENTIMENT_MODEL" \
  --input-variant title_context \
  --context-chars 400 \
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
    STEP="forecast-refit $arm"
    uv run python -m stf.cli forecast-refit \
      --arm "$arm" --model-dir "$model_dir" "${extra[@]}"
  fi
  echo "[daily] $(date -Is) predicting with $arm"
  STEP="forecast-predict $arm"
  # Exit code 2 = dated file already exists with different content (e.g. a
  # late-crawled article changed today's features). Keep the issued file and
  # continue with the remaining arms instead of aborting the whole loop.
  uv run python -m stf.cli forecast-predict \
    --model-dir "$model_dir" --output-dir "$live_dir" "${extra[@]}" || {
      rc=$?
      if [ "$rc" -eq 2 ]; then
        echo "[daily] $arm: dated prediction already issued; keeping it"
      else
        exit "$rc"
      fi
    }
  STEP="forecast-resolve $arm"
  uv run python -m stf.cli forecast-resolve \
    --model-dir "$model_dir" --predictions-dir "$live_dir" "${extra[@]}"
done

STEP="done"
echo "[daily] $(date -Is) done"
