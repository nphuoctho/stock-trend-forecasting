"""Per-ticker Vietstock news scraper: listing -> minute timestamp -> title + body.

Builds on the recipe verified in the spike (finance.vietstock.vn/View/PagingNewsContent),
extended to:
  - Extract the ARTICLE BODY (div itemprop="articleBody" id="vst_detail"), not just the title.
  - Cover the window up to config.DATE_END (default 2025-12-31).
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

import json
import os
import re
import time
from datetime import datetime
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from stf import config

BASE = "https://finance.vietstock.vn/View/PagingNewsContent"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.vietstock.vn/"}
SLEEP = 0.4
PAGE_SIZE = 20
MAX_PAGES = 400

# Every outbound request must be https to exactly one of these Vietstock hosts.
ALLOWED_HOSTS = frozenset({"finance.vietstock.vn", "vietstock.vn", "www.vietstock.vn"})

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

HREF = re.compile(
    r"""href\s*=\s*["']?((?://|https?://)(?:www\.)?vietstock\.vn/\d{4}/\d{2}/[^\s"'<>]+\.htm)""",
    re.IGNORECASE,
)
DATE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
ART_ID = re.compile(r"-(\d+)\.htm", re.IGNORECASE)


def _listing_rows(html: str) -> list[tuple[str, str]]:
    """Pair each discovered article URL with the nearest date in its listing fragment."""
    matches = list(HREF.finditer(html))
    rows: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        left = matches[index - 1].end() if index else 0
        right = matches[index + 1].start() if index + 1 < len(matches) else len(html)
        fragment = html[left:right]
        dates = list(DATE.finditer(fragment))
        if not dates:
            date = ""
        else:
            offset = match.start() - left
            date = min(dates, key=lambda item: abs(item.start() - offset)).group(1)
        rows.append((match.group(1), date))
    return rows


class _ArticleParser(HTMLParser):
    """Read the few article fields needed by the pipeline."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.body_parts: list[str] = []
        self.og_title: str | None = None
        self.og_description: str | None = None
        self.title_parts: list[str] = []
        self.published_parts: list[str] = []
        self.meta_published: str | None = None
        self._body_depth = 0
        self._capture: str | None = None
        self._capture_depth = 0
        self._skip_depth = 0  # inside <style>/<script>: drop CSS/JS payload

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name.lower(): value or "" for name, value in attrs}
        tag = tag.lower()

        if tag == "meta":
            prop = attr_map.get("property", "").lower()
            if prop == "og:title":
                self.og_title = attr_map.get("content")
            elif prop == "og:description":
                self.og_description = attr_map.get("content")
            elif attr_map.get("itemprop", "").lower() == "datepublished":
                self.meta_published = attr_map.get("content")

        if self._skip_depth:
            if tag not in _VOID_TAGS:
                self._skip_depth += 1
            return

        if tag in ("style", "script"):
            self._skip_depth = 1
            return

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
        elif tag in ("span", "div") and attr_map.get("itemprop", "").lower() == "datepublished":
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
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if self._body_depth:
            self._body_depth -= 1
            return
        if self._capture is not None:
            self._capture_depth -= 1
            if self._capture_depth == 0:
                self._capture = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._body_depth:
            self.body_parts.append(data)
        if self._capture == "title":
            self.title_parts.append(data)
        elif self._capture == "published":
            self.published_parts.append(data)


# <style>/<script> blocks (with their CSS/JS payload), HTML comments, and any
# residual tags are removed deterministically. Ordering matters: strip comments
# and style/script bodies before generic tags so their inner text never leaks.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_STYLE_SCRIPT = re.compile(r"<(style|script)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")


def clean_html(value: str | None) -> str | None:
    """Strip HTML/CSS markup from a text fragment, keeping readable words.

    Deterministically removes HTML comments, ``<style>``/``<script>`` blocks
    (including their CSS/JS payload), and any remaining tags, then unescapes
    entities and collapses whitespace. Only markup is discarded; title and body
    word content is preserved. Returns ``None`` when nothing readable remains.
    """
    if not value:
        return None
    text = _HTML_COMMENT.sub(" ", value)
    text = _STYLE_SCRIPT.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _clean_text(value: str | None) -> str | None:
    if not value:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text or None



def _log(msg: str) -> None:
    print(msg, flush=True)


def _atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` via temp file + ``os.replace``.

    A crash mid-write can never leave a partial/corrupt parquet at ``path``.
    """
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` via temp file + ``os.replace``."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _is_allowed_url(url: str) -> bool:
    """Return whether ``url`` is https and targets an exact allowed Vietstock host."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS


def get(
    url: str, params: dict | None = None, tries: int = 4
) -> requests.Response | None:
    """Fetch one page, retrying transient failures.

    Every request is restricted to https and an exact allowed Vietstock host. This is
    the single network choke point, so it applies equally to freshly discovered URLs
    and to URLs read back from a cached listings parquet -- neither can bypass
    validation. Redirects are never followed automatically: a redirect response is
    treated as a failure rather than silently retargeted.
    """
    if tries < 1:
        raise ValueError("tries must be at least 1.")
    if not _is_allowed_url(url):
        _log(f"[fetch] refusing disallowed url: {url}")
        return None

    for attempt in range(tries):
        try:
            response = requests.get(
                url, params=params, headers=HEADERS, timeout=30, allow_redirects=False
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


# News listings


def list_ticker_year(
    code: str, year: int, *, to_date: str
) -> tuple[list[tuple[str, str]], bool]:
    """Walk every news page for one ticker in one year.

    Returns ``(rows, complete)``. ``complete`` is True only when pagination ended
    on a page that genuinely had nothing new. A fetch failure or hitting
    ``MAX_PAGES`` yields whatever was collected with ``complete=False``, so a
    caller cannot mistake a truncated walk for a finished one.
    """
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    complete = False
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
            _log(f"[listings] {code} {year}: fetch failed on page {page}; walk incomplete")
            break
        new = [
            (href, date)
            for href, date in _listing_rows(r.text)
            if href not in seen
        ]
        if not new:  # empty page / only articles already seen => year exhausted
            complete = True
            break
        for h, d in new:
            seen.add(h)
            rows.append((h, d))
        time.sleep(SLEEP)
    else:
        _log(f"[listings] {code} {year}: hit MAX_PAGES={MAX_PAGES}; walk incomplete")
    return rows, complete


def _coverage_path() -> Path:
    return config.NEWS_DIR / "listings_coverage.json"


def _read_coverage() -> dict[str, str]:
    """Year -> the date that year was walked through, from the last complete run."""
    path = _coverage_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def collect_listings(refresh: bool = False, end: str | None = None) -> pd.DataFrame:
    """Gather news listings for every ticker x year in the config window. The news->ticker mapping source.

    Without ``refresh`` a year is reused only when a recorded watermark proves it
    was walked through its full extent. Row presence alone cannot prove that: a
    year that was the end year at cache time stopped at that run's ``to_date``,
    and would otherwise be frozen mid-year once the calendar rolls over.
    """
    end = end or config.DATE_END
    start_year = int(config.DATE_START[:4])
    end_year = int(end[:4])
    if end_year < start_year:
        raise ValueError(
            f"end {end!r} precedes the study window start {config.DATE_START!r}."
        )

    def _wanted_through(year: int) -> str:
        return end if year == end_year else f"{year}-12-31"

    def _within_window(frame: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows whose listing date lies inside [DATE_START, end].

        Filtering by ``year`` alone is not enough: a cache walked through December
        still carries rows past a mid-year ``--end``, so the flag would be ignored
        on the reuse path. ``list_date`` is Vietstock's ``dd/mm/yyyy``.
        """
        if frame.empty:
            return frame.reset_index(drop=True)
        listed = pd.to_datetime(frame["list_date"], format="%d/%m/%Y", errors="coerce")
        keep = listed.between(pd.Timestamp(config.DATE_START), pd.Timestamp(end))
        return frame[keep.fillna(False)].reset_index(drop=True)

    cached: pd.DataFrame | None = None
    years = list(range(start_year, end_year + 1))
    if config.LISTINGS_PQ.exists() and not refresh:
        cached = pd.read_parquet(config.LISTINGS_PQ)
        coverage = _read_coverage()
        covered = {
            year
            for year in range(start_year, end_year + 1)
            if coverage.get(str(year), "") >= _wanted_through(year)
        }
        years = [year for year in range(start_year, end_year + 1) if year not in covered]
        if cached.empty:
            cached = None
        elif not years:
            df = _within_window(cached)
            _log(
                f"[listings] reused: {len(df)} rows, {df['url'].nunique()} urls, "
                f"every year covered through its requested end"
            )
            return df
        else:
            _log(
                f"[listings] reusing {len(cached)} cached rows; "
                f"re-walking years {years}"
            )

    recs: list[dict] = []
    # A year earns its watermark only if every ticker's walk terminated on an empty
    # page. One transient fetch failure anywhere in the year withholds it, so the
    # next run re-walks instead of freezing a gap.
    year_complete = {year: True for year in years}
    # Drop the watermark for every year about to be rewritten, before the first
    # partial write. The parquet is replaced after each ticker, so a crash between
    # here and the end would otherwise leave partial rows that a stale watermark
    # still vouches for.
    if years:
        config.NEWS_DIR.mkdir(parents=True, exist_ok=True)
        stale = _read_coverage()
        for year in years:
            stale.pop(str(year), None)
        _atomic_write_json(_coverage_path(), stale)
    for code in config.TICKERS:
        tot = 0
        for year in years:
            # Final year stops at the requested end; other years go to Dec 31.
            to_date = end if year == end_year else f"{year}-12-31"
            rows, complete = list_ticker_year(code, year, to_date=to_date)
            year_complete[year] = year_complete[year] and complete
            for href, d in rows:
                if href.startswith("//"):
                    url = f"https:{href}"
                elif href.startswith("http://"):
                    url = f"https://{href[len('http://'):]}"
                else:
                    url = href
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
        if cached is not None and len(cached):
            kept = cached[~cached["year"].isin(years)]
            merged = pd.concat([kept, pd.DataFrame(recs)], ignore_index=True)
        else:
            merged = pd.DataFrame(recs)
        _atomic_to_parquet(merged, config.LISTINGS_PQ)  # save incrementally after each ticker

    if cached is not None and len(cached):
        df = pd.concat(
            [cached[~cached["year"].isin(years)], pd.DataFrame(recs)],
            ignore_index=True,
        )
    else:
        df = pd.DataFrame(recs)
    # Drop rows outside the configured window so the artifact keeps meaning
    # "listings in [DATE_START, end]" even after a run with a later --end.
    df = _within_window(df)
    # The watermark is written only here, after every ticker finished every year in
    # this run, and only for years whose every walk ended on an empty page. A crash
    # or a transient fetch failure therefore leaves the old watermark, so the
    # affected year is re-walked next time instead of being mistaken for complete.
    coverage = _read_coverage()
    for year in years:
        if year_complete.get(year):
            coverage[str(year)] = _wanted_through(year)
        else:
            _log(f"[listings] {year}: walk incomplete, watermark withheld")
    _atomic_write_json(_coverage_path(), coverage)
    _log(f"[listings] done: {len(df)} rows, {df['url'].nunique()} unique urls")
    return df


def join_listings_articles(
    listings: pd.DataFrame, articles: pd.DataFrame
) -> pd.DataFrame:
    """Attach article content to every listed ticker without collapsing mappings.

    An article may be listed under multiple tickers. The returned frame therefore
    keeps one row per ``(ticker, url)`` and only deduplicates article storage rows.
    """
    required_listings = {"ticker", "url"}
    required_articles = {"url", "published_at", "title", "body"}
    if not required_listings <= set(listings.columns):
        raise ValueError("listings needs 'ticker' and 'url' columns.")
    if not required_articles <= set(articles.columns):
        raise ValueError("articles needs url, published_at, title and body columns.")
    links = listings.loc[:, ["ticker", "url"]].drop_duplicates(["ticker", "url"])
    content = articles.drop_duplicates("url", keep="last")
    return links.merge(content, on="url", how="inner", validate="many_to_one")


def load_ticker_articles() -> pd.DataFrame:
    """Load the persisted listing/content join used by the forecasting panel."""
    if not config.LISTINGS_PQ.exists() or not config.ARTICLES_PQ.exists():
        raise FileNotFoundError("Both listings.parquet and articles.parquet are required.")
    return join_listings_articles(
        pd.read_parquet(config.LISTINGS_PQ),
        pd.read_parquet(config.ARTICLES_PQ),
    )


# Article content


def _parse_article_html(html: str) -> _ArticleParser:
    parser = _ArticleParser()
    parser.feed(html)
    parser.close()
    return parser


def extract_body(html: str) -> str | None:
    """Return the full article body, falling back to the page description.

    The joined body/description passes through :func:`clean_html`, the
    HTML/CSS normalization boundary, so any residual markup or entities are
    removed without altering the readable body semantics.
    """
    parser = _parse_article_html(html)
    body = clean_html(" ".join(parser.body_parts))
    return body or clean_html(parser.og_description)


def parse_article(html: str) -> dict[str, str | None]:
    """Extract the timestamp, title and body stored for one article."""
    parser = _parse_article_html(html)
    title = clean_html(parser.og_title) or clean_html(" ".join(parser.title_parts))
    timestamp = _clean_text(" ".join(parser.published_parts)) or parser.meta_published
    return {
        "published_at_str": timestamp,
        "title": title,
        "body": clean_html(" ".join(parser.body_parts))
        or clean_html(parser.og_description),
    }


def _to_iso(ts: str | None) -> str | None:
    if not ts:
        return None
    # ISO 8601 (from meta tag or already-normalized strings)
    if "T" in ts or (len(ts) >= 10 and ts[4] == "-" and ts[7] == "-"):
        iso = pd.to_datetime(ts, errors="coerce")
        if not pd.isna(iso):
            if iso.tzinfo is None:
                iso = iso.tz_localize(config.TIMEZONE)
            else:
                iso = iso.tz_convert(config.TIMEZONE)
            return iso.isoformat()
    # Vietstock formats: DD/MM/YYYY HH:MM or DD-MM-YYYY HH:MM[:SS][+TZ]
    for fmt in ("%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M:%S%z", "%d-%m-%Y %H:%M"):
        try:
            local = datetime.strptime(ts, fmt)
        except ValueError:
            continue
        if local.tzinfo is None:
            local = local.replace(tzinfo=ZoneInfo(config.TIMEZONE))
        return local.isoformat()
    return None

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
    urls: list[str],
    *,
    limit_urls: int | None = None,
    max_new: int | None = None,
    retry_failed: bool = False,
) -> pd.DataFrame:
    """Fetch article pages, reusing cached HTML and stored records."""
    config.NEWS_HTML_DIR.mkdir(parents=True, exist_ok=True)

    store: dict[str, dict] = {}
    if config.ARTICLES_PQ.exists():
        prev = pd.read_parquet(config.ARTICLES_PQ)
        for record in prev.to_dict("records"):
            normalized = {
                key: record.get(key)
                for key in (
                    "url",
                    "article_id",
                    "published_at",
                    "title",
                    "body",
                    "parse_failed",
                )
            }
            normalized["published_at"] = _normalize_stored_timestamp(
                normalized.get("published_at")
            )
            store[normalized["url"]] = normalized
        with_body = sum(_has_body(record.get("body")) for record in store.values())
        _log(
            f"[articles] loaded {len(store)} existing articles "
            f"({with_body} already have a body)"
        )

    if limit_urls is not None:
        urls = urls[:limit_urls]

    def _needs_body(url: str) -> bool:
        rec = store.get(url)
        if rec is None or not _has_body(rec.get("body")):
            return True
        if rec.get("published_at"):
            return False
        # Timestamp missing. `parse_failed` records that a parser already tried this
        # cached page and produced nothing, so a permanently timestamp-less article
        # stops re-entering the queue every run. `retry_failed` (set by --refresh) is
        # the escape hatch after a parser change.
        return retry_failed or not rec.get("parse_failed")

    def _flush() -> None:
        _atomic_to_parquet(pd.DataFrame(list(store.values())), config.ARTICLES_PQ)

    n_parsed = 0  # records written this run, for flush cadence and progress
    n_fetched = 0  # network fetches, the only thing --batch limits
    for i, url in enumerate(urls, 1):
        if not _needs_body(url):
            continue
        aid_m = ART_ID.search(url)
        aid = aid_m.group(1) if aid_m else sha256(url.encode()).hexdigest()[:16]
        cache = config.NEWS_HTML_DIR / f"{aid}.html"
        if cache.exists():
            html = cache.read_text(encoding="utf-8", errors="ignore")
        else:
            # The budget limits network work. Re-parsing cached HTML is free, so
            # charging it here would let a handful of stuck records consume the
            # whole batch every run and starve genuinely new articles.
            if max_new is not None and n_fetched >= max_new:
                _log(
                    f"[articles] hit batch of {max_new}, stopping (rest left for next run)"
                )
                break
            r = get(url)
            if r is None:
                continue
            html = r.text
            cache.write_text(html, encoding="utf-8")
            n_fetched += 1
            time.sleep(SLEEP)
        parsed = parse_article(html)
        published_at = _to_iso(parsed["published_at_str"])
        store[url] = {
            "url": url,
            "article_id": aid,
            "published_at": published_at,
            "title": parsed["title"],
            "body": parsed["body"],
            # Set when this parser found no timestamp. Cleared automatically on any
            # re-parse that succeeds; `news --refresh` forces that re-parse.
            "parse_failed": published_at is None,
        }
        n_parsed += 1
        if n_parsed % 100 == 0:
            _flush()
            with_body = sum(1 for r in store.values() if _has_body(r.get("body")))
            _log(
                f"[articles] {i}/{len(urls)} | parsed {n_parsed} | "
                f"fetched {n_fetched} | body {with_body}/{len(store)}"
            )

    _flush()
    return pd.DataFrame(list(store.values()))


def crawl(
    *,
    refresh: bool = False,
    limit_urls: int | None = None,
    max_new: int | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Run the news pipeline end to end: listing -> articles (ts + title + body)."""
    listings = collect_listings(refresh=refresh, end=end)
    urls = listings["url"].drop_duplicates().tolist()
    _log(f"[news] {len(urls)} unique urls to fetch content for")
    return fetch_articles(
        urls, limit_urls=limit_urls, max_new=max_new, retry_failed=refresh
    )


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
