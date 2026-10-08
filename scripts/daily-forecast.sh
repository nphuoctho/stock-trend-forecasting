#!/usr/bin/env bash
# Daily forecast job: refresh prices + news, score new articles, predict, resolve.
#
# SCHEDULE IT IN THE MORNING, NOT AFTER THE CLOSE.
#
# The price provider publishes a session's close only on the following day, so at
# 15:30 on day D the freshest close available is still D-1 and the job would be
# "predicting" a session that has already traded. Running before the 09:00 ICT
# opening auction uses the overnight publication of D-1 and targets D, which has
# not moved yet -- the only arrangement that produces a prospective forecast:
#   0 8 * * 1-5  /path/to/stock-trend-forecasting/scripts/daily-forecast.sh
#
# `forecast-predict` warns when an issuance cannot count as prospective, and
# `forecast-resolve` reports the prospective/replayed split of every run.
# Required env:
#   SENTIMENT_MODEL   checkpoint dir (default: models/sentiment/merged-refit/best)
#   SCORED_NEWS       score-news parquet (default: data/processed/news_sentiment_merged.parquet)
#
# Every run appends to logs/daily-YYYYMMDD.log and writes its outcome to
# outputs/live/last_run.json, which the dashboard reads via /api/live/status.
# Newly persisted signals, scored news and the job outcome are also appended to
# outputs/live/events.jsonl, which the API streams from /api/events.
set -euo pipefail
cd "$(dirname "$0")/.."

SENTIMENT_MODEL="${SENTIMENT_MODEL:-models/sentiment/merged-refit/best}"
SCORED_NEWS="${SCORED_NEWS:-data/processed/news_sentiment_merged.parquet}"
ARMS="${ARMS:-lstm_price_sentiment lstm_price logreg_price_sentiment tft_price_sentiment tft_price}"

TODAY="$(date +%F)"
mkdir -p logs outputs/live
exec > >(tee -a "logs/daily-${TODAY}.log") 2>&1

STEP="init"
# Appends every committed-but-unlogged event (signals, scored news, job status) to
# outputs/live/events.jsonl. The event log is derived from the artifacts, so this
# is idempotent and safe to run at any time; a failure here never fails the job.
reconcile_events() {
  uv run python -m stf.cli events-reconcile \
    || echo "[daily] warning: events-reconcile failed (rc=$?)" >&2
}

# Writes last_run.json and logs the matching job.status event.
write_status() {
  local rc="$1"
  STEP="$STEP" RC="$rc" uv run python - <<'PY'
import os

from stf import events

rc = int(os.environ["RC"])
events.record_last_run(
    exit_code=rc, failed_step=None if rc == 0 else os.environ["STEP"]
)
PY
}
# trap EXIT is only the fast path: SIGKILL, power loss or killing the process
# group skips it, so the next run (and the API) reconcile from the artifacts too.
trap 'rc=$?; write_status "$rc"; reconcile_events; exit $rc' EXIT

# Recover events a previous run persisted but never got to log.
reconcile_events

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
  # Exit code 3 = dated file already exists with different content (e.g. a
  # late-crawled article changed today's features). Keep the issued file and
  # continue with the remaining arms. Any other failure (missing sentiment
  # file, invalid manifest, hash mismatch) still aborts the job.
  uv run python -m stf.cli forecast-predict \
    --model-dir "$model_dir" --output-dir "$live_dir" "${extra[@]}" || {
      rc=$?
      if [ "$rc" -eq 3 ]; then
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
