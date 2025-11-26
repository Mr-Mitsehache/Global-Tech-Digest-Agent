# app/digest_service.py
from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Any

import httpx
import feedparser

from .config import AI_FEEDS, CYBER_FEEDS, ITEMS_PER_FEED, DIGESTS_DIR

import logging

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# 1) Async helpers สำหรับดึงข่าว (ยังเผื่อใช้กรณีอื่น / CLI)
# -------------------------------------------------------------------


async def _fetch_feed_entries(url: str, limit: int) -> List[Dict[str, Any]]:
    """ดึง RSS feed แบบ async ใช้ httpx + feedparser"""
    logger.info("DigestService: fetching RSS (async) from %s (limit=%d)", url, limit)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except Exception:
        logger.exception("DigestService: HTTP error when fetching %s", url)
        return []

    feed = feedparser.parse(resp.text)

    entries: List[Dict[str, Any]] = []
    for e in feed.entries[:limit]:
        entries.append(
            {
                "title": getattr(e, "title", "").strip(),
                "summary": getattr(e, "summary", getattr(e, "description", "")).strip(),
                "link": getattr(e, "link", "").strip(),
                "source": url,
            }
        )

    logger.info(
        "DigestService: fetched %d entries (async) from %s", len(entries), url
    )
    return entries


async def collect_ai_news() -> List[Dict[str, Any]]:
    """ดึงข่าวจาก AI_FEEDS แบบ async (ใช้ใน CLI/โหมดอื่นได้)"""
    logger.info(
        "DigestService: collect_ai_news from %d feeds (async)", len(AI_FEEDS)
    )
    tasks = [_fetch_feed_entries(url, ITEMS_PER_FEED) for url in AI_FEEDS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    items: List[Dict[str, Any]] = []
    for feed_url, res in zip(AI_FEEDS, results):
        if isinstance(res, Exception):
            logger.error("[AI_FEED] error from %s: %s", feed_url, res)
            continue
        items.extend(res)

    logger.info(
        "DigestService: collect_ai_news done, total_items=%d", len(items)
    )
    return items


async def collect_cyber_news() -> List[Dict[str, Any]]:
    """ดึงข่าวจาก CYBER_FEEDS แบบ async (ใช้ใน CLI/โหมดอื่นได้)"""
    logger.info(
        "DigestService: collect_cyber_news from %d feeds (async)",
        len(CYBER_FEEDS),
    )
    tasks = [_fetch_feed_entries(url, ITEMS_PER_FEED) for url in CYBER_FEEDS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    items: List[Dict[str, Any]] = []
    for feed_url, res in zip(CYBER_FEEDS, results):
        if isinstance(res, Exception):
            logger.error("[CYBER_FEED] error from %s: %s", feed_url, res)
            continue
        items.extend(res)

    logger.info(
        "DigestService: collect_cyber_news done, total_items=%d", len(items)
    )
    return items


# -------------------------------------------------------------------
# 2) ฟังก์ชันหลักที่ collector_agent ใช้ (sync + rule-based)
# -------------------------------------------------------------------


def fetch_rss_items(feed_url: str, max_items: int = 10) -> List[Dict[str, Any]]:
    """
    ดึงรายการข่าวจาก RSS feed แล้วแปลงเป็น list ของ dict แบบมาตรฐาน
    ใช้โดย collector_agent

    ตัวอย่างโครงสร้าง:
    {
        "title": "...",
        "summary": "...",
        "link": "https://...",
        "published": "2025-11-24T10:30:00",
        "source": "ชื่อเว็บ / feed title",
    }
    """
    logger.info(
        "DigestService: fetch_rss_items from %s (max_items=%d)",
        feed_url,
        max_items,
    )

    try:
        feed = feedparser.parse(feed_url)
    except Exception:
        logger.exception("DigestService: feedparser.parse failed for %s", feed_url)
        return []

    items: List[Dict[str, Any]] = []
    source_name = feed.feed.get("title", feed_url)

    for entry in feed.entries[:max_items]:
        published_raw = (
            entry.get("published")
            or entry.get("updated")
            or ""
        )

        items.append(
            {
                "title": entry.get("title", "").strip(),
                "summary": (
                    entry.get("summary")
                    or entry.get("description")
                    or ""
                ).strip(),
                "link": entry.get("link", ""),
                "published": published_raw,
                "source": source_name,
            }
        )

    logger.info(
        "DigestService: fetched %d items from %s", len(items), source_name
    )
    return items


# -------------------------------------------------------------------
# 3) ตัวช่วยเก่า (ยังเก็บไว้เผื่อใช้, แต่อาจไม่ใช้ใน graph ใหม่)
# -------------------------------------------------------------------


def _build_news_bullet_block(title: str, items: List[Dict[str, Any]]) -> str:
    """แปลง list ข่าว -> markdown bullet block (ใช้ใน CLI ได้)"""
    logger.debug(
        "DigestService: build_news_bullet_block '%s' (items=%d)",
        title,
        len(items),
    )
    lines = [f"### {title}", ""]
    for it in items:
        lines.append(f"- **{it.get('title', '')}**")
        if it.get("summary"):
            lines.append(f"  - สรุปย่อ: {it['summary']}")
        if it.get("link"):
            lines.append(f"  - ลิงก์: {it['link']}")
        lines.append("")
    return "\n".join(lines)


def _save_digest_to_file(md_text: str) -> Path:
    """
    legacy helper: เซฟ markdown ลงไฟล์ .md
    ตอนนี้เราใช้ HTML เป็นหลัก แต่เผื่อ CLI mode ที่อยากได้ .md
    """
    today = date.today().isoformat()
    DIGESTS_DIR.mkdir(parents=True, exist_ok=True)
    path = DIGESTS_DIR / f"{today}.md"
    path.write_text(md_text, encoding="utf-8")
    logger.info("DigestService: [ARCHIVE] Saved markdown digest to %s", path)
    return path


# -------------------------------------------------------------------
# 4) สร้าง / เซฟ Digest เป็น HTML (ใช้จริงใน summarizer + Web UI)
# -------------------------------------------------------------------


def generate_digest_markdown(
    ai_html: str,
    cyber_html: str,
    summary_html: str,
    date_str: str | None = None,
) -> str:
    """สร้าง HTML รวมทั้ง 3 หมวดในสไตล์รายงาน"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    logger.debug(
        "DigestService: generate_digest_markdown date=%s (len ai=%d, cyber=%d, summary=%d)",
        date_str,
        len(ai_html),
        len(cyber_html),
        len(summary_html),
    )

    return f"""
<h1>Global Tech Digest – AI & Cybersecurity</h1>
<p><em>ประจำวันที่ {date_str}</em></p>

<hr/>

<section class="ai-wrapper">
  {ai_html}
</section>

<hr/>

<section class="cyber-wrapper">
  {cyber_html}
</section>

<hr/>

<section class="summary-wrapper">
  {summary_html}
</section>
""".strip()


def save_digest_markdown(
    ai_html: str,
    cyber_html: str,
    summary_html: str,
    date_str: str | None = None,
) -> Path:
    """
    เซฟ digest เป็นไฟล์ HTML ในโฟลเดอร์ digests/ แล้วคืน path กลับไป
    ใช้โดย summarizer_agent
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    DIGESTS_DIR.mkdir(parents=True, exist_ok=True)
    # 1) ไฟล์รวมฉบับเต็ม
    full_html = generate_digest_markdown(ai_html, cyber_html, summary_html, date_str)
    full_path = DIGESTS_DIR / f"{date_str}.html"
    full_path.write_text(full_html, encoding="utf-8")
    logger.info("DigestService: saved full digest %s", full_path)

    # 2) partials แยกหมวด (ให้ web / RAG โหลดง่าย)
    ai_path = DIGESTS_DIR / f"{date_str}.ai.html"
    cyber_path = DIGESTS_DIR / f"{date_str}.cyber.html"
    summary_path = DIGESTS_DIR / f"{date_str}.summary.html"

    ai_path.write_text(ai_html, encoding="utf-8")
    cyber_path.write_text(cyber_html, encoding="utf-8")
    summary_path.write_text(summary_html, encoding="utf-8")

    logger.info(
        "DigestService: saved partials for %s (ai=%s, cyber=%s, summary=%s)",
        date_str,
        ai_path.name,
        cyber_path.name,
        summary_path.name,
    )

    return full_path


# -------------------------------------------------------------------
# 5) Archive utilities (ใช้โดย Web UI + RAG)
# -------------------------------------------------------------------


def list_digest_files() -> List[Path]:
    """คืน list ของไฟล์ digest .html ทั้งหมด (เรียงใหม่สุดก่อน)"""
    DIGESTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DIGESTS_DIR.glob("*.html"), reverse=True)
    logger.info("DigestService: list_digest_files -> %d files", len(files))
    return files


def load_digest_html(date_str: str) -> str | None:
    """โหลด digest HTML ตามวันที่ (YYYY-MM-DD) ถ้าไม่มีคืน None"""
    DIGESTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DIGESTS_DIR / f"{date_str}.html"

    if not file_path.exists():
        logger.warning(
            "DigestService: load_digest_html not found for date=%s", date_str
        )
        return None

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        logger.exception(
            "DigestService: failed to read digest file %s", file_path
        )
        return None

    logger.info(
        "DigestService: loaded digest for %s from %s (len=%d)",
        date_str,
        file_path,
        len(content),
    )
    return content

def load_digest_parts(date_str: str) -> tuple[str | None, str | None, str | None]:
    """
    โหลด HTML เฉพาะส่วนสำหรับวันนั้น:
    - ai_html
    - cyber_html
    - summary_html
    ถ้าไฟล์ไหนไม่มี จะคืน None กลับมาแทน
    """
    DIGESTS_DIR.mkdir(parents=True, exist_ok=True)

    def _read_partial(suffix: str) -> str | None:
        path = DIGESTS_DIR / f"{date_str}.{suffix}.html"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    ai_html = _read_partial("ai")
    cyber_html = _read_partial("cyber")
    summary_html = _read_partial("summary")

    logger.info(
        "DigestService: load_digest_parts(%s) -> ai=%s cyber=%s summary=%s",
        date_str,
        bool(ai_html),
        bool(cyber_html),
        bool(summary_html),
    )

    return ai_html, cyber_html, summary_html