"""Scraper tin Vietstock per-ticker: listing -> timestamp phút -> tiêu đề + nội dung.

Kế thừa recipe đã xác minh ở spike (finance.vietstock.vn/View/PagingNewsContent),
mở rộng:
  - Trích NỘI DUNG BÀI (div itemprop="articleBody" id="vst_detail"), không chỉ tiêu đề.
  - Cửa sổ tới config.DATE_END (mặc định 31/03/2026).
  - Timestamp tới phút (itemprop="datePublished") cho cutoff 15:00 ở Phase 3.

Thiết kế cho chạy nền dài + chạy lại rẻ:
  - Cache HTML trang bài -> chạy lại không tải lại.
  - Ghi parquet tăng dần -> dừng giữa chừng vẫn còn dữ liệu dùng được.
  - listings.parquet: mọi (mã, url, ngày) -> nguồn ánh xạ tin->mã.
  - articles.parquet: mỗi url 1 dòng (timestamp phút + tiêu đề + nội dung), đã dedup.

Chạy:
    uv run python -m stf.cli news                 # crawl toàn bộ
    uv run python -m stf.cli news --limit-urls 20 # verify nhỏ, chỉ 20 bài
"""

from __future__ import annotations

import re
import time
from datetime import datetime

import pandas as pd
import requests

from stf import config

BASE = "https://finance.vietstock.vn/View/PagingNewsContent"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.vietstock.vn/"}
SLEEP = 0.4  # nhịp lịch sự, khớp spike đã chạy an toàn
PAGE_SIZE = 20
MAX_PAGES = 400  # chặn vòng lặp vô hạn nếu paginator lặp lại trang cuối

# --- Regex trích dữ liệu --------------------------------------------------
HREF = re.compile(r"href=(//vietstock\.vn/\d{4}/\d{2}/[^\s\"']+\.htm)", re.I)
DATE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
PUB = re.compile(r"\b(\d{2}/\d{2}/\d{4} \d{2}:\d{2})\b")  # itemprop=datePublished
OG_TITLE = re.compile(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', re.I)
TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
ART_ID = re.compile(r"-(\d+)\.htm", re.I)
# Khối nội dung bài: <div itemprop="articleBody" id="vst_detail"> ... </div>
BODY_BLOCK = re.compile(
    r'<div[^>]*itemprop=["\']articleBody["\'][^>]*id=["\']vst_detail["\'][^>]*>(.*?)</div>',
    re.I | re.S,
)
TAG = re.compile(r"<[^>]+>")


def _log(msg: str) -> None:
    print(msg, flush=True)


def get(url: str, params: dict | None = None, tries: int = 4) -> requests.Response | None:
    """GET với backoff lũy thừa. Trả None sau khi hết lượt (bỏ bài lỗi, không chết job)."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r
            time.sleep(SLEEP * (2 ** i))  # 429/5xx: lùi lâu hơn
        except requests.RequestException:
            time.sleep(SLEEP * (2 ** i))
    return None


# --- Pha 1: listing (ánh xạ tin -> mã) ------------------------------------

def list_ticker_year(code: str, year: int, *, to_date: str) -> list[tuple[str, str]]:
    """Duyệt hết trang tin của 1 mã trong 1 năm. Trả [(url, ngày dd/mm/yyyy)]."""
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
        if not new:  # trang rỗng / chỉ lặp bài đã thấy => hết năm
            break
        for h, d in new:
            seen.add(h)
            rows.append((h, d))
        time.sleep(SLEEP)
    return rows


def collect_listings(refresh: bool = False) -> pd.DataFrame:
    """Gom danh sách tin toàn bộ mã x năm trong cửa sổ config. Nguồn ánh xạ tin->mã."""
    if config.LISTINGS_PQ.exists() and not refresh:
        df = pd.read_parquet(config.LISTINGS_PQ)
        _log(f"[listings] dùng lại: {len(df)} dòng, {df['url'].nunique()} url, "
             f"{df['ticker'].nunique()} mã")
        return df

    start_year = int(config.DATE_START[:4])
    end_year = int(config.DATE_END[:4])
    recs: list[dict] = []
    for code in config.TICKERS:
        tot = 0
        for year in range(start_year, end_year + 1):
            # Năm cuối chỉ lấy tới DATE_END (vd 2026-03-31), năm khác tới 31/12.
            to_date = config.DATE_END if year == end_year else f"{year}-12-31"
            rows = list_ticker_year(code, year, to_date=to_date)
            for href, d in rows:
                recs.append({"ticker": code, "url": "https:" + href,
                             "list_date": d, "year": year})
            tot += len(rows)
            _log(f"[listings] {code} {year}: {len(rows)} bài (luỹ kế mã {tot})")
        config.NEWS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(recs).to_parquet(config.LISTINGS_PQ)  # lưu tăng dần sau mỗi mã
    df = pd.DataFrame(recs)
    _log(f"[listings] XONG: {len(df)} dòng, {df['url'].nunique()} url duy nhất")
    return df


# --- Pha 2: nội dung bài (timestamp + tiêu đề + body) ---------------------

def extract_body(html: str) -> str | None:
    """Trích nội dung bài từ khối vst_detail, strip HTML, gom whitespace."""
    m = BODY_BLOCK.search(html)
    if not m:
        return None
    text = TAG.sub(" ", m.group(1))  # bỏ mọi thẻ HTML
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def parse_article(html: str) -> dict:
    """Trả dict(timestamp_str, title, body) từ HTML 1 bài."""
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


def fetch_articles(urls: list[str], *, limit_urls: int | None = None) -> pd.DataFrame:
    """Lấy timestamp phút + tiêu đề + nội dung cho mỗi url. Cache HTML, ghi tăng dần."""
    config.NEWS_HTML_DIR.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if config.ARTICLES_PQ.exists():
        prev = pd.read_parquet(config.ARTICLES_PQ)
        # Chỉ coi là "đã xong" khi bài đã có cột body (dữ liệu cũ chỉ có title thì cào lại body).
        has_body = "body" in prev.columns
        for r in prev.to_dict("records"):
            if has_body and r.get("body"):
                done[r["url"]] = r
        _log(f"[articles] đã có {len(done)} bài (có body) từ lần trước, bỏ qua")

    if limit_urls is not None:
        urls = urls[:limit_urls]

    recs = list(done.values())
    n_new = 0
    for i, url in enumerate(urls, 1):
        if url in done:
            continue
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
        recs.append({"url": url, "article_id": aid,
                     "published_at": _to_iso(parsed["published_at_str"]),
                     **parsed})
        n_new += 1
        if n_new % 100 == 0:
            pd.DataFrame(recs).to_parquet(config.ARTICLES_PQ)
            with_ts = sum(1 for r in recs if r.get("published_at"))
            with_body = sum(1 for r in recs if r.get("body"))
            _log(f"[articles] {i}/{len(urls)} | mới {n_new} | ts {with_ts} | body {with_body}")
    df = pd.DataFrame(recs)
    config.NEWS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.ARTICLES_PQ)
    return df


def crawl(*, refresh: bool = False, limit_urls: int | None = None) -> pd.DataFrame:
    """Chạy pipeline tin đầu-cuối: listing -> articles (ts + title + body)."""
    listings = collect_listings(refresh=refresh)
    urls = listings["url"].drop_duplicates().tolist()
    _log(f"[news] {len(urls)} url duy nhất để lấy nội dung")
    return fetch_articles(urls, limit_urls=limit_urls)


def summary(articles: pd.DataFrame) -> None:
    """In tổng kết coverage tin."""
    with_ts = articles["published_at"].notna().sum() if "published_at" in articles else 0
    with_body = articles["body"].notna().sum() if "body" in articles else 0
    _log("\n==== TỔNG KẾT TIN ====")
    _log(f"articles : {len(articles)} bài | {with_ts} có timestamp | {with_body} có nội dung")
