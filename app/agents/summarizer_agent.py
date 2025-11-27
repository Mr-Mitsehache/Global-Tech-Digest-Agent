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
เนื้อหามาจาก:
- บทความเชิงลึก / research note
- ข่าวเปิดตัวผลิตภัณฑ์ / แพลตฟอร์ม / เครื่องมือ
- บทวิเคราะห์แนวโน้มในอุตสาหกรรม

ข้อกำหนดรูปแบบเอาต์พุต:
- ตอบเป็น **HTML เท่านั้น**
- ใช้ tag แค่: <section>, <h2>, <h3>, <p>, <ul>, <li>, <strong>, <em>
- แบ่งย่อหน้าให้ชัดเจน อ่านง่าย ไม่เขียนติดเป็นก้อนเดียว
- ใช้ภาษาไทย แบบรายงานมืออาชีพ กระชับแต่ครบประเด็น
- ถ้าเนื้อหาไม่มีหัวข้อบางอย่างจริง ๆ ให้ข้ามหัวข้อนั้นไปได้ ไม่ต้องยัดเอง

โครงสร้างที่ต้องการ:

<section class="ai-section">
  <h2>หมวดที่ 1: ข่าวด้าน AI / LLM / Agentic</h2>

  <p>ย่อหน้าเปิด: สรุปภาพรวมเทรนด์ AI วันนี้ เช่น เน้น research, infra, หรือ product</p>

  <h3>1. งานวิจัยและโมเดล (Research / Models)</h3>
  <p>สรุปประเด็นจากบทความเชิงลึก / paper / blog ด้านโมเดลและเทคนิคใหม่ ๆ</p>
  <ul>
    <li><strong>หัวข้อข่าว / บทความ</strong> – ใจความสำคัญ 1–2 ประโยค</li>
  </ul>

  <h3>2. โครงสร้างพื้นฐานและเครื่องมือ (Infra / Tooling / Platforms)</h3>
  <p>สรุปข่าวที่เกี่ยวกับแพลตฟอร์ม, tooling, inference infra, retriever, vector DB ฯลฯ</p>
  <ul>
    <li><strong>หัวข้อข่าว</strong> – อธิบายว่ามีผลต่อ developer / org อย่างไร</li>
  </ul>

  <h3>3. การประยุกต์ใช้และธุรกิจ (Applications / Business Impact)</h3>
  <p>สรุปการใช้งานจริงในภาคธุรกิจ, ผลิตภัณฑ์ใหม่, use case ที่น่าสนใจ</p>
  <ul>
    <li><strong>หัวข้อข่าว</strong> – ใจความ และผลกระทบต่อผู้ใช้ / ตลาด</li>
  </ul>

  <h3>4. นโยบายและผลกระทบต่อสังคม (Policy / Governance / Social Impact)</h3>
  <p>ถ้าไม่มีข่าวด้าน policy จริง ๆ ให้เขียนสั้น ๆ ว่ายังไม่มีประเด็น policy สำคัญในวันนี้</p>

  <p><strong>สรุปท้ายหมวด:</strong> สรุป insight สำคัญ 2–3 ประเด็น ที่ผู้บริหารควรโฟกัสจากข่าวทั้งหมดวันนี้</p>
</section>

ข้อจำกัดสำคัญ:
- ใช้ข้อมูลเฉพาะจากรายการข่าวที่ให้เท่านั้น
- ห้ามสร้างชื่อบริษัท, โมเดล, หรือตัวเลขขึ้นมาเอง ถ้าไม่มีในข่าว
- ถ้าข่าวเน้นด้านใดด้านหนึ่ง (เช่น มีแต่ infra กับ research) ให้เน้นส่วนนั้น ไม่ต้องฝืนให้ครบทุกหัวข้อ
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
คุณคือผู้ช่วยเขียน "รายงานสรุปข่าวด้าน Cybersecurity" สำหรับผู้บริหารและทีม Security

เนื้อหาที่ได้รับอาจผสมกันระหว่าง:
- ข่าวเหตุโจมตีจริง (APT, ransomware, data breach)
- Advisory / bulletin จากหน่วยงานรัฐ (เช่น CISA) ที่มี CVE, vendor, severity
- บทความเล่าเคส / story-based, podcast, analysis

ข้อกำหนดรูปแบบเอาต์พุต:
- ตอบเป็น HTML เท่านั้น
- ใช้ tag แค่: <section>, <h2>, <h3>, <p>, <ul>, <li>, <strong>, <em>
- แบ่งย่อหน้าให้อ่านง่าย
- เน้นมุม "แล้วองค์กรควรสนใจอะไร/ทำอะไร" ไม่ใช่แค่เล่าเรื่อง

โครงสร้างที่ต้องการ:

<section class="cyber-section">
  <h2>หมวดที่ 2: ข่าวด้าน Cybersecurity</h2>

  <p>ย่อหน้าเปิด: สรุปภาพรวมสถานการณ์ด้านความปลอดภัยในช่วงข่าวชุดนี้ เช่น เน้น ransomware, ช่องโหว่, data leak ฯลฯ</p>

  <h3>1. แคมเปญโจมตีและภัยคุกคามสำคัญ</h3>
  <ul>
    <li><strong>ชื่อ/คำอธิบายสั้น ๆ ของแคมเปญ</strong> – ใครโจมตีใคร, เทคนิคหลัก, เป้าหมาย</li>
  </ul>

  <h3>2. ช่องโหว่และแพตช์สำคัญ (Vulnerabilities & Patches)</h3>
  <p>สรุปช่องโหว่ที่ถูกพูดถึงหรือถูกโจมตี (ระบุ vendor / product / CVE ถ้ามีในข่าว)</p>
  <ul>
    <li><strong>ชื่อช่องโหว่หรือผลิตภัณฑ์</strong> – ใจความ, ระดับความเสี่ยงโดยประมาณจากข่าว, สถานะแพตช์ (ถ้าข่าวระบุ)</li>
  </ul>

  <h3>3. เหตุการณ์ข้อมูลรั่วไหล / Policy / Regulation</h3>
  <p>สรุป data breach, กรณีโดนแฮ็กองค์กรใหญ่, หรือประเด็นด้านกฎระเบียบที่เกี่ยวข้อง</p>

  <h3>4. ข้อเสนอแนะสำหรับองค์กร</h3>
  <p>สรุปเป็น bullet 3–5 ข้อ ว่าองค์กรควรตรวจสอบ/ทำอะไร เช่น พิจารณาแพตช์, ตรวจ config, เตรียม incident response</p>
</section>

ข้อจำกัด:
- ให้ใช้ข้อมูลเฉพาะจากข่าวที่ส่งให้เท่านั้น ห้ามเดาตัวเลข CVE หรือ vendor ถ้าไม่มีในข่าว
- ถ้าไม่มีข่าวในบางหัวข้อให้เขียนสั้น ๆ ว่า "ในชุดข่าวนี้ยังไม่พบประเด็นด้าน X ที่เด่นชัด"
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
ผู้อ่านคือผู้บริหารที่ต้องการมองภาพใหญ่ ทั้งโอกาส (AI) และความเสี่ยง (Cyber)

รูปแบบ:
- ตอบเป็น HTML
- ใช้ <section>, <h2>, <p>, <ul>, <li> เท่านั้น
- ไม่ต้องลงรายละเอียดเท่าหมวดหลัก แต่ต้องมี insight ชัดเจน

โครงสร้าง:

<section class="summary-section">
  <h2>สรุปภาพรวมวันนี้</h2>
  <p>ย่อหน้าแรก: วันนี้เทรนด์สำคัญจากฝั่ง AI และ Cyber คืออะไร (อย่างละ 1–2 ประเด็น)</p>
  <p>ย่อหน้าถัดไป: เชื่อมโยงว่าโอกาสจาก AI มาพร้อมความเสี่ยงอะไรจากข่าวด้าน Cyber</p>
  <ul>
    <li>Insight สำคัญข้อที่ 1 – เช่น โอกาสทางธุรกิจหรือการเพิ่ม productivity</li>
    <li>Insight สำคัญข้อที่ 2 – เช่น ความเสี่ยงที่เพิ่มขึ้นจากช่องโหว่หรือแคมเปญโจมตี</li>
    <li>Insight สำคัญข้อที่ 3 – ข้อคิดเชิงกลยุทธ์ที่องค์กรควรนำไปใช้</li>
  </ul>
</section>

ข้อจำกัด:
- สรุปจากรายการข่าวที่ให้เท่านั้น
- ไม่ต้องทวนทุกข่าว ให้ดึงเฉพาะประเด็นหลักและความสัมพันธ์ระหว่าง AI กับ Cyber
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
