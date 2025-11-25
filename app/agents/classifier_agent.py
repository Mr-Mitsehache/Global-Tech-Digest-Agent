# app/agents/classifier_agent.py
from __future__ import annotations

from typing import Dict, Any, List

from ..states.digest_state import DigestState
import logging
logger = logging.getLogger(__name__)


def classifier_node(state: DigestState) -> DigestState:
    """
    แบ่ง raw_items ออกเป็น ai_items / cyber_items
    (ตอนนี้ใช้ rule-based จาก field 'category' ที่ collector ใส่มา)
    ภายหลังอยากใช้ LLM ช่วยจัดหมวดเพิ่มก็มาเปลี่ยนที่ node นี้ node เดียว
    """
    raw: List[Dict[str, Any]] = state.get("raw_items", []) or []

    ai_items: List[Dict[str, Any]] = []
    cyber_items: List[Dict[str, Any]] = []

    logger.info(
        "Classifier: start classify ai=%d, cyber=%d",
        len(ai_items), len(cyber_items),
    )

    for item in raw:
        if item.get("category") == "cyber":
            cyber_items.append(item)
        else:
            ai_items.append(item)

    logger.info(
        "Classifier: finished classify (ai=%d, cyber=%d)",
        len(ai_items), len(cyber_items),
    )

    return {
        "ai_items": ai_items,
        "cyber_items": cyber_items,
    }
