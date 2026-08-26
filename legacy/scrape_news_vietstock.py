"""Scraper tin Vietstock per-ticker cho Phase 1 — 10 mã VN30, 2020–2025.

Sinh corpus tin có timestamp tới PHÚT (cần cho cutoff 15:00 ở Phase 3) và giữ
ánh xạ tin→mã (Vietstock trả tin đã gắn sẵn mã, không phải đoán mapping).

Recipe đã xác minh ở spike (src/data/spike_news_vietstock.py). robots.txt của
Vietstock (kiểm 22/08) không chặn /View/PagingNewsContent hay trang bài *.htm.

Thiết kế cho chạy nền dài + chạy lại rẻ:
  - Cache HTML trang bài vào data/raw/news/html/ (R8): chạy lại không tải lại.
  - Ghi parquet tăng dần: dừng giữa chừng vẫn còn dữ liệu dùng được.
  - listings.parquet giữ mọi (mã, url, ngày) — nguồn ánh xạ tin→mã.
  - articles.parquet giữ mỗi url 1 dòng: timestamp phút + tiêu đề (đã dedup).

Chạy: uv run python src/data/scrape_news_vietstock.py
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

TICKERS = ["FPT", "GAS", "HPG", "MBB", "MWG", "TCB", "VCB", "VHM", "VIC", "VNM"]
YEARS = range(2020, 2026)  # 2020..2025

BASE = "https://finance.vietstock.vn/View/PagingNewsContent"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.vietstock.vn/"}
SLEEP = 0.4  # nhịp lịch sự, khớp với spike đã chạy an toàn
PAGE_SIZE = 20
MAX_PAGES = 400  # chặn vòng lặp vô hạn nếu paginator lặp lại trang cuối

HREF = re.compile(r"href=(//vietstock\.vn/\d{4}/\d{2}/[^\s\"']+\.htm)", re.I)
DATE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
PUB = re.compile(r"\b(\d{2}/\d{2}/\d{4} \d{2}:\d{2})\b")  # itemprop="datePublished"
OG_TITLE = re.compile(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', re.I)
TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
ART_ID = re.compile(r"-(\d+)\.htm", re.I)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "news"
HTML_DIR = OUT / "html"
LISTINGS_PQ = OUT / "listings.parquet"
ARTICLES_PQ = OUT / "articles.parquet"


def _log(msg: str) -> None:
    print(msg, flush=True)


def get(url: str, params: dict | None = None, tries: int = 4) -> requests.Response | None:
    """GET với backoff. Trả None nếu hỏng sau các lần thử (bỏ qua bài lỗi, không chết cả job)."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r
            # 429/5xx: lùi lâu hơn rồi thử lại
            time.sleep(SLEEP * (2 ** i))
        except requests.RequestException:
            time.sleep(SLEEP * (2 ** i))
    return None


def list_ticker_year(code: str, year: int) -> list[tuple[str, str]]:
    """Duyệt hết các trang tin của 1 mã trong 1 năm. Trả [(url, ngày dd/mm/yyyy)]."""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for page in range(1, MAX_PAGES + 1):
        r = get(BASE, params={
            "view": 1, "code": code, "type": 1,
            "fromDate": f"{year}-01-01", "toDate": f"{year}-12-31",
            "channelID": 0, "page": page, "pageSize": PAGE_SIZE,
        })
        if r is None:
            break
        hrefs = HREF.findall(r.text)
        dates = DATE.findall(r.text)
        new = [(h, d) for h, d in zip(hrefs, dates + [""] * len(hrefs)) if h not in seen]
        if not new:  # trang rỗng hoặc chỉ lặp lại bài đã thấy => hết năm
            break
        for h, d in new:
            seen.add(h)
            rows.append((h, d))
        time.sleep(SLEEP)
    return rows


def collect_listings(refresh: bool = False) -> pd.DataFrame:
    """Pha 1: gom danh sách tin toàn bộ mã × năm. Nguồn ánh xạ tin→mã."""
    if LISTINGS_PQ.exists() and not refresh:
        df = pd.read_parquet(LISTINGS_PQ)
        _log(f"[listings] dùng lại {LISTINGS_PQ.name}: {len(df)} dòng, "
             f"{df['url'].nunique()} url, {df['ticker'].nunique()} mã")
        return df
    recs = []
    for code in TICKERS:
        tot = 0
        for year in YEARS:
            rows = list_ticker_year(code, year)
            for href, d in rows:
                recs.append({"ticker": code, "url": "https:" + href,
                             "list_date": d, "year": year})
            tot += len(rows)
            _log(f"[listings] {code} {year}: {len(rows)} bài (luỹ kế mã {tot})")
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(recs).to_parquet(LISTINGS_PQ)  # lưu tăng dần sau mỗi mã
    df = pd.DataFrame(recs)
    _log(f"[listings] XONG: {len(df)} dòng, {df['url'].nunique()} url duy nhất")
    return df


def parse_article(html: str) -> tuple[str | None, str | None]:
    """Trả (timestamp 'dd/mm/yyyy hh:mm', tiêu đề)."""
    m = PUB.search(html)
    ts = m.group(1).strip() if m else None
    t = OG_TITLE.search(html) or TITLE.search(html)
    title = re.sub(r"\s+", " ", t.group(1)).strip() if t else None
    return ts, title


def fetch_articles(urls: list[str]) -> pd.DataFrame:
    """Pha 2: lấy timestamp phút + tiêu đề cho mỗi url. Cache HTML, ghi tăng dần."""
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if ARTICLES_PQ.exists():
        done = {r["url"]: r for r in pd.read_parquet(ARTICLES_PQ).to_dict("records")}
        _log(f"[articles] đã có {len(done)} bài từ lần chạy trước, bỏ qua")
    recs = list(done.values())
    n_new = 0
    for i, url in enumerate(urls, 1):
        if url in done:
            continue
        aid_m = ART_ID.search(url)
        aid = aid_m.group(1) if aid_m else str(abs(hash(url)))
        cache = HTML_DIR / f"{aid}.html"
        if cache.exists():
            html = cache.read_text(encoding="utf-8", errors="ignore")
        else:
            r = get(url)
            if r is None:
                continue
            html = r.text
            cache.write_text(html, encoding="utf-8")
            time.sleep(SLEEP)
        ts, title = parse_article(html)
        iso = None
        if ts:
            try:
                iso = datetime.strptime(ts, "%d/%m/%Y %H:%M").isoformat()
            except ValueError:
                iso = None
        recs.append({"url": url, "article_id": aid, "published_at_str": ts,
                     "published_at": iso, "title": title})
        n_new += 1
        if n_new % 100 == 0:
            pd.DataFrame(recs).to_parquet(ARTICLES_PQ)
            with_ts = sum(1 for r in recs if r["published_at"])
            _log(f"[articles] {i}/{len(urls)} | mới {n_new} | có timestamp {with_ts}")
    df = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ARTICLES_PQ)
    return df


def main() -> int:
    refresh = "--refresh" in sys.argv
    listings = collect_listings(refresh=refresh)
    urls = listings["url"].drop_duplicates().tolist()
    _log(f"[main] {len(urls)} url duy nhất để lấy timestamp")
    articles = fetch_articles(urls)

    with_ts = articles["published_at"].notna().sum()
    _log("\n==== TỔNG KẾT ====")
    _log(f"listings : {len(listings)} dòng (ánh xạ tin→mã), {listings['ticker'].nunique()} mã")
    _log(f"articles : {len(articles)} bài duy nhất, {with_ts} có timestamp phút")
    _log(f"file     : {LISTINGS_PQ}\n           {ARTICLES_PQ}")
    ok = with_ts >= 2000
    _log(f"\nCỔNG 18/08 (≥2000 tin có timestamp): {'ĐẠT' if ok else 'CHƯA ĐẠT'} ({with_ts})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
