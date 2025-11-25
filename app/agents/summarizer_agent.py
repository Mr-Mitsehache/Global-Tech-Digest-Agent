# app/agents/summarizer_agent.py
from __future__ import annotations

import logging
from typing import Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from app.llm import get_chat_model
from app.digest_service import save_digest_markdown
from app.states.digest_state import DigestState

logger = logging.getLogger(__name__)

# ---------- UTIL / GUARDRAILS ----------


def _strip_code_fence(text: str) -> str:
    """ตัด ```...``` ออกจากรอบ ๆ ข้อความ (กันโมเดลตอบเป็น code block)"""
    if not isinstance(text, str):
        return str(text)

    s = text.strip()
    if not s.startswith("```"):
        return s

    lines = s.splitlines()

    # ตัดบรรทัดแรกถ้าเป็น ``` หรือ ```html, ```markdown ฯลฯ
    first = lines[0]
    if first.startswith("```"):
        lines = lines[1:]

    # ตัดบรรทัดสุดท้ายถ้าเป็น ```
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _fallback_if_too_short(
    text: str, section_name: str, min_chars: int = 400
) -> str:
    """ถ้า LLM ตอบสั้นผิดปกติ หรือว่าง -> ใส่ข้อความ fallback ที่อ่านรู้เรื่องแทน"""
    if not text or len(text.strip()) < min_chars:
        logger.warning(
            "Summarizer: fallback for section '%s' (len=%d < %d)",
            section_name,
            len(text.strip()) if text else 0,
            min_chars,
        )
        return (
            f"<p><strong>หมายเหตุ:</strong> "
            f"วันนี้ระบบยังไม่สามารถดึงข่าวในหมวด {section_name} ได้ครบถ้วน "
            f"หรือโมเดลตอบสั้นผิดปกติ จึงแสดงข้อความแจ้งเตือนแทน.</p>"
        )
    return text


def _message_to_str(resp: Any) -> str:
    """
    แปลงผลลัพธ์จาก LLM (AIMessage หรือ object แปลก ๆ) ให้เป็น string เดียว
    รองรับทั้งกรณี content เป็น str หรือ list ของส่วนต่าง ๆ (เช่น {'type': 'text', 'text': '...'})
    """
    if isinstance(resp, AIMessage):
        content = resp.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                # google genai มักจะให้ dict ที่มี key 'text'
                if isinstance(part, dict) and "text" in part:
                    parts.append(str(part["text"]))
                else:
                    parts.append(str(part))
            return "\n".join(parts)
    return str(resp)


# ---------- PROMPTS ----------

AI_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
คุณคือผู้ช่วยเขียน "รายงานสรุปข่าวเทคโนโลยีด้าน AI / LLM / Agentic" สำหรับผู้บริหาร

ข้อกำหนดรูปแบบเอาต์พุต:
- ตอบเป็น **HTML เท่านั้น**
- ใช้ tag แค่: <section>, <h2>, <h3>, <p>, <ul>, <li>, <strong>, <em>
- แบ่งย่อหน้าให้ชัดเจน อ่านง่าย ไม่เขียนติดเป็นก้อนเดียว
- ใช้ภาษาไทย สั้น กระชับ แบบรายงาน ไม่ใช่แชตคุยเล่น

รูปแบบรายงานที่ต้องการ (ห้ามใส่ ```markdown หรือ ```html):
1. บทนำสั้น ๆ 1 ย่อหน้า
2. หัวข้อย่อย 2–4 หัวข้อ (เช่น โครงสร้างพื้นฐาน, แอปพลิเคชัน, policy)
3. แต่ละหัวข้อ:
   - สรุปแนวโน้มภาพรวม 2–10 บรรทัด
   - bullet ข่าวสำคัญ 2–5 bullet (ชื่อข่าว + ใจความ)
4. สรุปท้ายหมวด 1 ย่อหน้า

โครงสร้างที่ต้องการ:

<section class="ai-section">
  <h2>หมวดที่ 1: ข่าวด้าน AI / LLM / Agentic</h2>

  <p>ย่อหน้าเปิด บอกภาพรวมเทรนด์วันนี้</p>

  <h3>โครงสร้างพื้นฐาน / เครื่องมือ (Infra / Tooling)</h3>
  <p>สรุปข่าวที่เกี่ยวข้องกับ infra, tooling, platform, model hosting ฯลฯ</p>

  <h3>การประยุกต์ใช้และผลิตภัณฑ์ (Applications / Products)</h3>
  <p>สรุปการใช้งานจริงในธุรกิจ / ผลิตภัณฑ์ใหม่ที่น่าสนใจ</p>

  <h3>นโยบายและผลกระทบต่อสังคม (Policy / Social Impact)</h3>
  <p>ถ้ามีประเด็นด้านนโยบาย, จริยธรรม, ผลกระทบต่อสังคมให้สรุปสั้น ๆ</p>
</section>

เนื้อหาให้ดึงจากรายการข่าวที่ผู้ใช้ส่งมาให้เท่านั้น
ไม่ต้องใส่ข้อมูลเพิ่มเองเกินจากนั้นมากนัก เน้นการจัดระเบียบและสรุปให้เข้าใจง่าย
            """,
        ),
        (
            "human",
            """
ข่าวฝั่ง AI / LLM / Agentic วันนี้ (list ของข่าวพร้อมรายละเอียด):

{ai_items}

โปรดเขียน HTML รายงานตามโครงสร้างที่กำหนดด้านบน
            """,
        ),
    ]
)

CYBER_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
คุณคือผู้ช่วยเขียน "รายงานสรุปข่าวด้าน Cybersecurity" สำหรับผู้บริหาร

ข้อกำหนดรูปแบบเอาต์พุต:
- ตอบเป็น HTML เท่านั้น
- ใช้ tag แค่: <section>, <h2>, <h3>, <p>, <ul>, <li>, <strong>, <em>
- แบ่งย่อหน้าให้อ่านง่าย

โครงสร้างที่ต้องการ:

<section class="cyber-section">
  <h2>หมวดที่ 2: ข่าวด้าน Cybersecurity</h2>

  <p>ย่อหน้าเปิด สรุปภาพรวมสถานการณ์ด้านความปลอดภัยวันนี้</p>

  <h3>ภัยคุกคามและแคมเปญโจมตีสำคัญ</h3>
  <ul>
    <li>รายการสรุปภัยคุกคาม/แคมเปญที่สำคัญ พร้อมคำอธิบายสั้น ๆ</li>
  </ul>

  <h3>ช่องโหว่และแพตช์สำคัญ</h3>
  <p>สรุปช่องโหว่ที่ถูกพูดถึง / ถูกโจมตี และสถานะการแพตช์</p>

  <h3>ข้อเสนอแนะสำหรับองค์กร</h3>
  <p>แนะนำสิ่งที่องค์กรควรระวังหรือควรทำในระยะสั้น</p>
</section>
            """,
        ),
        (
            "human",
            """
ข่าวฝั่ง Cybersecurity วันนี้:

{cyber_items}

โปรดเขียน HTML รายงานตามโครงสร้างที่กำหนดด้านบน
            """,
        ),
    ]
)

OVERALL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
คุณคือผู้ช่วยสรุป "ภาพรวมเทรนด์ประจำวัน" จากข่าว AI และ Cybersecurity

รูปแบบ:
- ตอบเป็น HTML
- ใช้ <section>, <h2>, <p>, <ul>, <li> ก็พอ
- เน้นสรุปภาพรวม ไม่ลงรายละเอียดเท่าหมวดหลัก

โครงสร้าง:

<section class="summary-section">
  <h2>สรุปภาพรวมวันนี้</h2>
  <p>ย่อหน้าแรก: วันนี้เทรนด์สำคัญคืออะไร</p>
  <p>ย่อหน้าถัดไป: ความเสี่ยงหลักและโอกาสที่ควรจับตา</p>
  <ul>
    <li>ข้อสังเกตหรือ Insight สำคัญข้อที่ 1</li>
    <li>ข้อสังเกตหรือ Insight สำคัญข้อที่ 2</li>
  </ul>
</section>
            """,
        ),
        (
            "human",
            """
ข่าวฝั่ง AI / LLM / Agentic:

{ai_items}

ข่าวฝั่ง Cybersecurity:

{cyber_items}

โปรดเขียนสรุปภาพรวมตามโครงสร้างที่กำหนดด้านบน
            """,
        ),
    ]
)


async def _run_section_chain(chain, variables: dict, section_name: str) -> str:
    """helper เรียก LLM + ทำความสะอาด text + fallback + log length"""
    try:
        logger.info("Summarizer: calling LLM for section '%s'", section_name)
        resp = await chain.ainvoke(variables)
    except Exception:
        logger.exception("Summarizer: LLM call failed for section '%s'", section_name)
        return _fallback_if_too_short("", section_name)

    raw = _message_to_str(resp)
    cleaned = _strip_code_fence(raw)
    cleaned = _fallback_if_too_short(cleaned, section_name)
    logger.info(
        "Summarizer: section '%s' produced HTML length=%d",
        section_name,
        len(cleaned),
    )
    return cleaned


# ---------- NODE FUNCTION ----------


async def summarizer_node(state: DigestState) -> DigestState:
    """
    รับรายการข่าวจาก collector/classifier แล้ว:
    - เรียก LLM สร้าง HTML รายงาน 3 ส่วน: AI, Cyber, Summary
    - เซฟลงไฟล์ผ่าน save_digest_markdown
    - คืนค่า state ที่เพิ่ม ai_html, cyber_html, summary_html
    """
    ai_items = state.get("ai_items", [])
    cyber_items = state.get("cyber_items", [])

    logger.info(
        "Summarizer: start (ai_items=%d, cyber_items=%d)",
        len(ai_items),
        len(cyber_items),
    )

    model = get_chat_model()  # Gemini (กำหนดใน app.llm)

    # ถ้าทั้งสองฝั่งไม่มีข่าวเลย ก็ไม่ต้องเปลือง LLM
    if not ai_items and not cyber_items:
        logger.warning(
            "Summarizer: both ai_items and cyber_items are empty, using pure fallback"
        )
        ai_html = _fallback_if_too_short("", "ข่าวด้าน AI / LLM / Agentic")
        cyber_html = _fallback_if_too_short("", "ข่าวด้าน Cybersecurity")
        summary_html = _fallback_if_too_short("", "สรุปภาพรวม")
    else:
        # 1) สรุปฝั่ง AI
        if ai_items:
            ai_chain = AI_SUMMARY_PROMPT | model
            ai_html = await _run_section_chain(
                ai_chain,
                {"ai_items": ai_items},
                "ข่าวด้าน AI / LLM / Agentic",
            )
        else:
            logger.warning("Summarizer: no AI items, using fallback section")
            ai_html = _fallback_if_too_short("", "ข่าวด้าน AI / LLM / Agentic")

        # 2) สรุปฝั่ง Cyber
        if cyber_items:
            cyber_chain = CYBER_SUMMARY_PROMPT | model
            cyber_html = await _run_section_chain(
                cyber_chain,
                {"cyber_items": cyber_items},
                "ข่าวด้าน Cybersecurity",
            )
        else:
            logger.warning("Summarizer: no Cyber items, using fallback section")
            cyber_html = _fallback_if_too_short("", "ข่าวด้าน Cybersecurity")

        # 3) สรุปภาพรวม
        summary_chain = OVERALL_SUMMARY_PROMPT | model
        summary_html = await _run_section_chain(
            summary_chain,
            {"ai_items": ai_items, "cyber_items": cyber_items},
            "สรุปภาพรวม",
        )

    # อัพเดต state ให้ web เอาไปแสดง
    state["ai_html"] = ai_html
    state["cyber_html"] = cyber_html
    state["summary_html"] = summary_html

    # เซฟเป็นไฟล์ markdown/html ตามที่เรากำหนดใน digest_service
    try:
        save_digest_markdown(ai_html, cyber_html, summary_html)
        logger.info("Summarizer: digest saved to archive successfully")
    except Exception:
        logger.exception("Summarizer: failed to save digest archive")

    return {
        "ai_html": ai_html,
        "cyber_html": cyber_html,
        "summary_html": summary_html,
    }
