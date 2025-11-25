# app/agents/collector_agent.py
from __future__ import annotations

from typing import Dict, Any, List

import logging

from ..states.digest_state import DigestState
from ..config import AI_FEEDS, CYBER_FEEDS
from ..digest_service import fetch_rss_items

logger = logging.getLogger(__name__)


def collector_node(state: DigestState) -> DigestState:
    """
    ดึงข่าวจาก RSS feeds ทั้งฝั่ง AI และ Cyber
    แล้วใส่ tag category คร่าว ๆ ให้แต่ละข่าว
    """
    logger.info(
        "Collector: start fetching RSS items (ai_feeds=%d, cyber_feeds=%d)",
        len(AI_FEEDS),
        len(CYBER_FEEDS),
    )

    items: List[Dict[str, Any]] = []

    # ดึงจากทุก feed (ทั้ง AI และ Cyber)
    for feed in AI_FEEDS + CYBER_FEEDS:
        try:
            feed_items = fetch_rss_items(feed_url=feed, max_items=10)
            logger.info(
                "Collector: fetched %d items from feed=%s",
                len(feed_items),
                feed,
            )
            items.extend(feed_items)
        except Exception:
            # กันกรณี feed ใด feed หนึ่งพัง แต่ยังให้ตัวอื่นทำงานต่อได้
            logger.exception("Collector: error while fetching feed=%s", feed)

    logger.info("Collector: total items before labeling = %d", len(items))

    # ติด label category แบบ rule-based ง่าย ๆ
    ai_count = 0
    cyber_count = 0

    for item in items:
        source = (item.get("source") or "").lower()
        title = (item.get("title") or "").lower()

        if (
            "security" in source
            or "hacker" in source
            or "bleepingcomputer" in source
            or "cisa" in title
            or "ransomware" in title
        ):
            item["category"] = "cyber"
            cyber_count += 1
        else:
            item["category"] = "ai"
            ai_count += 1

    logger.info(
        "Collector: labeled items -> ai=%d, cyber=%d (total=%d)",
        ai_count,
        cyber_count,
        len(items),
    )

    # graph state จะเอา key 'raw_items' ไปใช้ต่อใน node ถัดไป
    return {"raw_items": items}
