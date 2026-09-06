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
from hashlib import sha256
from html.parser import HTMLParser
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from stf import config

BASE = "https://finance.vietstock.vn/View/PagingNewsContent"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.vietstock.vn/"}
SLEEP = 0.4
PAGE_SIZE = 20
MAX_PAGES = 400

_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}

# Listing pages expose the article URL and date in separate fragments.
HREF = re.compile(
    r"""href\s*=\s*["']?((?://|https?://)(?:www\.)?vietstock\.vn/\d{4}/\d{2}/[^\s"'<>]+\.htm)""",
    re.IGNORECASE,
)
DATE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
ART_ID = re.compile(r"-(\d+)\.htm", re.IGNORECASE)


class _ArticleParser(HTMLParser):
    """Read the few article fields needed by the pipeline."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.body_parts: list[str] = []
        self.og_title: str | None = None
        self.og_description: str | None = None
        self.title_parts: list[str] = []
        self.published_parts: list[str] = []
        self._body_depth = 0
        self._capture: str | None = None
        self._capture_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name.lower(): value or "" for name, value in attrs}
        tag = tag.lower()

        if tag == "meta":
            prop = attr_map.get("property", "").lower()
            if prop == "og:title":
                self.og_title = attr_map.get("content")
            elif prop == "og:description":
                self.og_description = attr_map.get("content")

        if self._body_depth:
            if tag not in _VOID_TAGS:
                self._body_depth += 1
            return

        if (
            tag == "div"
            and attr_map.get("id", "").lower() == "vst_detail"
            and attr_map.get("itemprop", "").lower() == "articlebody"
        ):
            self._body_depth = 1
            return

        if self._capture is not None:
            if tag not in _VOID_TAGS:
                self._capture_depth += 1
            return

        if tag == "title":
            self._capture = "title"
            self._capture_depth = 1
        elif tag == "span" and attr_map.get("itemprop", "").lower() == "datepublished":
            self._capture = "published"
            self._capture_depth = 1

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _VOID_TAGS:
            return
        if self._body_depth:
            self._body_depth -= 1
            return
        if self._capture is not None:
            self._capture_depth -= 1
            if self._capture_depth == 0:
                self._capture = None

    def handle_data(self, data: str) -> None:
        if self._body_depth:
            self.body_parts.append(data)
        if self._capture == "title":
            self.title_parts.append(data)
        elif self._capture == "published":
            self.published_parts.append(data)


def _clean_text(value: str | None) -> str | None:
    if not value:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text or None



def _log(msg: str) -> None:
    print(msg, flush=True)


def get(
    url: str, params: dict | None = None, tries: int = 4
) -> requests.Response | None:
    """Fetch one page, retrying transient failures."""
    if tries < 1:
        raise ValueError("tries must be at least 1.")

    for attempt in range(tries):
        try:
            response = requests.get(
                url, params=params, headers=HEADERS, timeout=30
            )
        except requests.RequestException:
            response = None

        if response is not None:
            if response.status_code == 200:
                response.encoding = "utf-8"
                return response
            if response.status_code < 500 and response.status_code != 429:
                return None

        if attempt + 1 < tries:
            time.sleep(SLEEP * (2**attempt))
    return None


# --- Phase 1: listing (news -> ticker mapping) ----------------------------


def list_ticker_year(code: str, year: int, *, to_date: str) -> list[tuple[str, str]]:
    """Walk every news page for one ticker in one year. Returns [(url, dd/mm/yyyy date)]."""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for page in range(1, MAX_PAGES + 1):
        r = get(
            BASE,
            params={
                "view": 1,
                "code": code,
                "type": 1,
                "fromDate": f"{year}-01-01",
                "toDate": to_date,
                "channelID": 0,
                "page": page,
                "pageSize": PAGE_SIZE,
            },
        )
        if r is None:
            break
        hrefs = HREF.findall(r.text)
        dates = DATE.findall(r.text)
        new = [
            (h, d) for h, d in zip(hrefs, dates + [""] * len(hrefs)) if h not in seen
        ]
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
        _log(
            f"[listings] reused: {len(df)} rows, {df['url'].nunique()} urls, "
            f"{df['ticker'].nunique()} tickers"
        )
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
                url = href if href.startswith("http") else f"https:{href}"
                recs.append(
                    {
                        "ticker": code,
                        "url": url,
                        "list_date": d,
                        "year": year,
                    }
                )
            tot += len(rows)
            _log(
                f"[listings] {code} {year}: {len(rows)} articles (ticker running total {tot})"
            )
        config.NEWS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(recs).to_parquet(
            config.LISTINGS_PQ
        )  # save incrementally after each ticker
    df = pd.DataFrame(recs)
    _log(f"[listings] done: {len(df)} rows, {df['url'].nunique()} unique urls")
    return df


# --- Phase 2: article content (timestamp + title + body) ------------------


def _parse_article_html(html: str) -> _ArticleParser:
    parser = _ArticleParser()
    parser.feed(html)
    parser.close()
    return parser


def extract_body(html: str) -> str | None:
    """Return the full article body, falling back to the page description."""
    parser = _parse_article_html(html)
    body = _clean_text(" ".join(parser.body_parts))
    return body or _clean_text(parser.og_description)


def parse_article(html: str) -> dict[str, str | None]:
    """Extract the timestamp, title and body stored for one article."""
    parser = _parse_article_html(html)
    title = parser.og_title or _clean_text(" ".join(parser.title_parts))
    timestamp = _clean_text(" ".join(parser.published_parts))
    return {
        "published_at_str": timestamp,
        "title": _clean_text(title),
        "body": _clean_text(" ".join(parser.body_parts))
        or _clean_text(parser.og_description),
    }


def _to_iso(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        local = datetime.strptime(ts, "%d/%m/%Y %H:%M")
    except ValueError:
        return None
    return local.replace(tzinfo=ZoneInfo(config.TIMEZONE)).isoformat()


def _normalize_stored_timestamp(value: str | None) -> str | None:
    """Upgrade old naive timestamps to the study timezone."""
    if not value:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(config.TIMEZONE)
    else:
        timestamp = timestamp.tz_convert(config.TIMEZONE)
    return timestamp.isoformat()


def _has_body(val) -> bool:
    """Return whether a stored body contains text."""
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
    """Fetch article pages, reusing cached HTML and stored records."""
    config.NEWS_HTML_DIR.mkdir(parents=True, exist_ok=True)

    store: dict[str, dict] = {}
    if config.ARTICLES_PQ.exists():
        prev = pd.read_parquet(config.ARTICLES_PQ)
        for record in prev.to_dict("records"):
            record["published_at"] = _normalize_stored_timestamp(
                record.get("published_at")
            )
            store[record["url"]] = record
        with_body = sum(_has_body(record.get("body")) for record in store.values())
        _log(
            f"[articles] loaded {len(store)} existing articles "
            f"({with_body} already have a body)"
        )

    if limit_urls is not None:
        urls = urls[:limit_urls]

    def _needs_body(url: str) -> bool:
        rec = store.get(url)
        return rec is None or not _has_body(rec.get("body"))

    def _flush() -> None:
        # Replace in one operation so a crash cannot leave a partial parquet.
        tmp = config.ARTICLES_PQ.with_suffix(".parquet.tmp")
        pd.DataFrame(list(store.values())).to_parquet(tmp)
        os.replace(tmp, config.ARTICLES_PQ)

    n_new = 0
    for i, url in enumerate(urls, 1):
        if not _needs_body(url):
            continue
        if max_new is not None and n_new >= max_new:
            _log(
                f"[articles] hit batch of {max_new}, stopping (rest left for next run)"
            )
            break
        aid_m = ART_ID.search(url)
        aid = aid_m.group(1) if aid_m else sha256(url.encode()).hexdigest()[:16]
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
        store[url] = {
            "url": url,
            "article_id": aid,
            "published_at": _to_iso(parsed["published_at_str"]),
            **parsed,
        }
        n_new += 1
        if n_new % 100 == 0:
            _flush()
            with_body = sum(1 for r in store.values() if _has_body(r.get("body")))
            _log(
                f"[articles] {i}/{len(urls)} | batch {n_new} | body {with_body}/{len(store)}"
            )

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
    with_ts = (
        articles["published_at"].notna().sum() if "published_at" in articles else 0
    )
    with_body = (
        sum(_has_body(value) for value in articles["body"])
        if "body" in articles
        else 0
    )
    _log("\nNews summary")
    _log(
        f"articles : {len(articles)} total | {with_ts} with timestamp | {with_body} with body"
    )
