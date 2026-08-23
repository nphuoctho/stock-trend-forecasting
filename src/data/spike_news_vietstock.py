"""Spike: chứng minh nguồn tin Vietstock per-ticker, có lọc khoảng ngày, lùi tới 2020.

Recipe (dùng lại cho scraper chính thức Phase 1):
  GET https://finance.vietstock.vn/View/PagingNewsContent
    view=1 & code=<TICKER> & type=1 & fromDate=YYYY-MM-DD & toDate=YYYY-MM-DD
    & channelID=0 & page=<N> & pageSize=20
  - fromDate/toDate BẮT BUỘC định dạng ISO (yyyy-mm-dd); dd/mm/yyyy bị bỏ qua.
  - Danh sách trả về ngày dd/mm/yyyy + href //vietstock.vn/YYYY/MM/<slug>.htm
  - Timestamp tới PHÚT nằm trên trang bài: itemprop="datePublished" -> "dd/mm/yyyy hh:mm"
    (cần cho cutoff 15:00 ở Phase 3).

Chạy: uv run python src/data/spike_news_vietstock.py
"""

from __future__ import annotations

import re
import sys
import time

import requests

BASE = "https://finance.vietstock.vn/View/PagingNewsContent"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.vietstock.vn/"}
HREF = re.compile(r"href=(//vietstock\.vn/\d{4}/\d{2}/[^\s\"']+\.htm)", re.I)
DATE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
# Timestamp hiển thị dạng "dd/mm/yyyy hh:mm" (thẻ có itemprop="datePublished").
PUB = re.compile(r"\b(\d{2}/\d{2}/\d{4} \d{2}:\d{2})\b")


def list_page(code: str, year: int, page: int) -> list[tuple[str, str]]:
    """Trả (href, ngày) cho 1 trang tin của mã trong 1 năm."""
    params = {
        "view": 1, "code": code, "type": 1,
        "fromDate": f"{year}-01-01", "toDate": f"{year}-12-31",
        "channelID": 0, "page": page, "pageSize": 20,
    }
    r = requests.get(BASE, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    hrefs = HREF.findall(r.text)
    dates = DATE.findall(r.text)
    return list(zip(hrefs, dates + [""] * len(hrefs)))


def article_timestamp(href: str) -> str | None:
    """Lấy timestamp tới phút từ trang bài."""
    url = "https:" + href if href.startswith("//") else href
    r = requests.get(url, headers=HEADERS, timeout=30)
    m = PUB.search(r.text)
    return m.group(1).strip() if m else None


def main() -> int:
    code = "HPG"
    seen: set[str] = set()
    for year in range(2020, 2026):
        for page in range(1, 4):
            for href, _ in list_page(code, year, page):
                seen.add(href)
            time.sleep(0.4)  # nhịp lịch sự
        print(f"{code} {year}: luỹ kế {len(seen)} bài (3 trang/năm)")
    sample = next(iter(seen))
    ts = article_timestamp(sample)
    print(f"\nMẫu timestamp: {ts}  <-  {sample}")
    ok = len(seen) >= 200 and ts is not None
    print(f"\nSPIKE NEWS: {'PASS' if ok else 'FAIL'} ({len(seen)} bài, có timestamp phút={ts is not None})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
