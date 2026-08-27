"""Per-ticker Vietstock news scraper: listing -> minute timestamp -> title + body.

Builds on the recipe verified in the spike (finance.vietstock.vn/View/PagingNewsContent),
extended to:
  - Extract the ARTICLE BODY (div itemprop="articleBody" id="vst_detail"), not just the title.
  - Cover the window up to config.DATE_END (default 2026-03-31).
  - Keep minute-level timestamps (itemprop="datePublished") for the 15:00 cutoff in Phase 3.

Built for long background runs and cheap reruns:
  - Cache article HTML so reruns don't re-download.
  - Write parquet incrementally so stopping midway still leaves usable data.
  - listings.parquet: every (ticker, url, date) -> the news->ticker mapping source.
  - articles.parquet: one row per url (minute timestamp + title + body), deduped.

Run:
    uv run python -m stf.cli news                 # crawl everything
    uv run python -m stf.cli news --limit-urls 20 # small check, 20 articles only
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime

import pandas as pd
import requests

from stf import config

BASE = "https://finance.vietstock.vn/View/PagingNewsContent"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.vietstock.vn/"}
SLEEP = 0.4  # polite pacing, matches the spike that ran safely
PAGE_SIZE = 20
MAX_PAGES = 400  # guard against an infinite loop if the paginator repeats the last page

# --- Data-extraction regexes ----------------------------------------------
HREF = re.compile(r"href=(//vietstock\.vn/\d{4}/\d{2}/[^\s\"']+\.htm)", re.I)
DATE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
PUB = re.compile(r"\b(\d{2}/\d{2}/\d{4} \d{2}:\d{2})\b")  # itemprop=datePublished
OG_TITLE = re.compile(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', re.I)
TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
ART_ID = re.compile(r"-(\d+)\.htm", re.I)
# Article body block: <div itemprop="articleBody" id="vst_detail"> ... </div>
BODY_BLOCK = re.compile(
    r'<div[^>]*itemprop=["\']articleBody["\'][^>]*id=["\']vst_detail["\'][^>]*>(.*?)</div>',
    re.I | re.S,
)
# Fallback for longform/feature articles with no vst_detail block: use og:description.
OG_DESC = re.compile(
    r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)', re.I)
TAG = re.compile(r"<[^>]+>")


def _log(msg: str) -> None:
    print(msg, flush=True)


def get(url: str, params: dict | None = None, tries: int = 4) -> requests.Response | None:
    """GET with exponential backoff. Returns None after retries run out (skip the bad article, don't kill the job)."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r
            time.sleep(SLEEP * (2 ** i))  # 429/5xx: back off longer
        except requests.RequestException:
            time.sleep(SLEEP * (2 ** i))
    return None


# --- Phase 1: listing (news -> ticker mapping) ----------------------------

def list_ticker_year(code: str, year: int, *, to_date: str) -> list[tuple[str, str]]:
    """Walk every news page for one ticker in one year. Returns [(url, dd/mm/yyyy date)]."""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for page in range(1, MAX_PAGES + 1):
        r = get(BASE, params={
            "view": 1, "code": code, "type": 1,
            "fromDate": f"{year}-01-01", "toDate": to_date,
            "channelID": 0, "page": page, "pageSize": PAGE_SIZE,
        })
        if r is None:
            break
        hrefs = HREF.findall(r.text)
        dates = DATE.findall(r.text)
        new = [(h, d) for h, d in zip(hrefs, dates + [""] * len(hrefs)) if h not in seen]
        if not new:  # empty page / only articles already seen => year exhausted
            break
        for h, d in new:
            seen.add(h)
            rows.append((h, d))
        time.sleep(SLEEP)
    return rows


def collect_listings(refresh: bool = False) -> pd.DataFrame:
    """Gather news listings for every ticker x year in the config window. The news->ticker mapping source."""
    if config.LISTINGS_PQ.exists() and not refresh:
        df = pd.read_parquet(config.LISTINGS_PQ)
        _log(f"[listings] reused: {len(df)} rows, {df['url'].nunique()} urls, "
             f"{df['ticker'].nunique()} tickers")
        return df

    start_year = int(config.DATE_START[:4])
    end_year = int(config.DATE_END[:4])
    recs: list[dict] = []
    for code in config.TICKERS:
        tot = 0
        for year in range(start_year, end_year + 1):
            # Final year stops at DATE_END (e.g. 2026-03-31); other years go to Dec 31.
            to_date = config.DATE_END if year == end_year else f"{year}-12-31"
            rows = list_ticker_year(code, year, to_date=to_date)
            for href, d in rows:
                recs.append({"ticker": code, "url": "https:" + href,
                             "list_date": d, "year": year})
            tot += len(rows)
            _log(f"[listings] {code} {year}: {len(rows)} articles (ticker running total {tot})")
        config.NEWS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(recs).to_parquet(config.LISTINGS_PQ)  # save incrementally after each ticker
    df = pd.DataFrame(recs)
    _log(f"[listings] done: {len(df)} rows, {df['url'].nunique()} unique urls")
    return df


# --- Phase 2: article content (timestamp + title + body) ------------------

def extract_body(html: str) -> str | None:
    """Extract the article body from the vst_detail block, strip HTML, collapse whitespace.

    Normal articles use the articleBody/vst_detail block (full content). Longform/feature
    articles without it fall back to og:description (a short blurb) so sentiment still has
    some text, instead of leaving it empty and re-crawling forever.
    """
    m = BODY_BLOCK.search(html)
    if m:
        text = TAG.sub(" ", m.group(1))  # drop all HTML tags
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            return text
    # Fallback: og:description for longform/feature articles.
    d = OG_DESC.search(html)
    if d:
        return re.sub(r"\s+", " ", d.group(1)).strip() or None
    return None


def parse_article(html: str) -> dict:
    """Return dict(timestamp_str, title, body) from one article's HTML."""
    m = PUB.search(html)
    ts = m.group(1).strip() if m else None
    t = OG_TITLE.search(html) or TITLE.search(html)
    title = re.sub(r"\s+", " ", t.group(1)).strip() if t else None
    return {"published_at_str": ts, "title": title, "body": extract_body(html)}


def _to_iso(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%d/%m/%Y %H:%M").isoformat()
    except ValueError:
        return None


def _has_body(val) -> bool:
    """True if body holds real content (not None, not NaN, not empty).

    Needed because pandas stores missing cells as NaN (float), and `not float('nan')`
    is False — so an article without a body can look like it has one. Check via pd.isna
    to be safe.
    """
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(val).strip())


def fetch_articles(
    urls: list[str], *, limit_urls: int | None = None, max_new: int | None = None
) -> pd.DataFrame:
    """Fetch minute timestamp + title + body for each url. Caches HTML, writes incrementally.

    Upsert by url: KEEP every old article (including ones with only a title, no body yet)
    and only ADD a body to those still missing one. max_new caps how many bodies get added
    per run (lets cron crawl the whole corpus gradually across runs).
    """
    config.NEWS_HTML_DIR.mkdir(parents=True, exist_ok=True)

    # Load every old article into the store keyed by url (drop nothing).
    store: dict[str, dict] = {}
    if config.ARTICLES_PQ.exists():
        prev = pd.read_parquet(config.ARTICLES_PQ)
        for r in prev.to_dict("records"):
            store[r["url"]] = r
        with_body = sum(1 for r in store.values() if _has_body(r.get("body")))
        _log(f"[articles] loaded {len(store)} existing articles ({with_body} already have a body)")

    if limit_urls is not None:
        urls = urls[:limit_urls]

    def _needs_body(url: str) -> bool:
        rec = store.get(url)
        return rec is None or not _has_body(rec.get("body"))

    def _flush() -> None:
        # Atomic write: write a temp file then os.replace, to avoid corruption on concurrent
        # read/write or a mid-write crash (os.replace is atomic on the same filesystem).
        tmp = config.ARTICLES_PQ.with_suffix(".parquet.tmp")
        pd.DataFrame(list(store.values())).to_parquet(tmp)
        os.replace(tmp, config.ARTICLES_PQ)

    n_new = 0
    for i, url in enumerate(urls, 1):
        if not _needs_body(url):
            continue
        if max_new is not None and n_new >= max_new:
            _log(f"[articles] hit batch of {max_new}, stopping (rest left for next run)")
            break
        aid_m = ART_ID.search(url)
        aid = aid_m.group(1) if aid_m else str(abs(hash(url)))
        cache = config.NEWS_HTML_DIR / f"{aid}.html"
        if cache.exists():
            html = cache.read_text(encoding="utf-8", errors="ignore")
        else:
            r = get(url)
            if r is None:
                continue
            html = r.text
            cache.write_text(html, encoding="utf-8")
            time.sleep(SLEEP)
        parsed = parse_article(html)
        store[url] = {"url": url, "article_id": aid,
                      "published_at": _to_iso(parsed["published_at_str"]),
                      **parsed}
        n_new += 1
        if n_new % 100 == 0:
            _flush()
            with_body = sum(1 for r in store.values() if _has_body(r.get("body")))
            _log(f"[articles] {i}/{len(urls)} | batch {n_new} | body {with_body}/{len(store)}")

    _flush()
    return pd.DataFrame(list(store.values()))


def crawl(
    *, refresh: bool = False, limit_urls: int | None = None, max_new: int | None = None
) -> pd.DataFrame:
    """Run the news pipeline end to end: listing -> articles (ts + title + body)."""
    listings = collect_listings(refresh=refresh)
    urls = listings["url"].drop_duplicates().tolist()
    _log(f"[news] {len(urls)} unique urls to fetch content for")
    return fetch_articles(urls, limit_urls=limit_urls, max_new=max_new)


def summary(articles: pd.DataFrame) -> None:
    """Print a news coverage summary."""
    with_ts = articles["published_at"].notna().sum() if "published_at" in articles else 0
    with_body = articles["body"].notna().sum() if "body" in articles else 0
    _log("\nNews summary")
    _log(f"articles : {len(articles)} total | {with_ts} with timestamp | {with_body} with body")
