# app/states/digest_state.py
from __future__ import annotations

from typing import TypedDict, List, Dict, Any


class DigestState(TypedDict, total=False):
    """
    state กลางที่ทุก node ใช้ร่วมกันในกราฟสรุปข่าว
    total=False = field ไม่จำเป็นต้องมีทุก key ในทุกช่วง
    """
    raw_items: List[Dict[str, Any]]      # ข่าวดิบทั้งหมดจาก RSS
    ai_items: List[Dict[str, Any]]       # ข่าวที่จัดเป็นหมวด AI/LLM
    cyber_items: List[Dict[str, Any]]    # ข่าวหมวด Cybersecurity
    ai_html: str                         # HTML/Markdown หมวด AI
    cyber_html: str                      # HTML/Markdown หมวด Cyber
    summary_html: str                    # สรุปภาพรวมประจำวัน
