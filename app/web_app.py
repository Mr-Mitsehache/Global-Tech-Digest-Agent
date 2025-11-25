# app/web_app.py
from __future__ import annotations

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.graphs.digest_graph import build_digest_graph
from .config import TEMPLATES_DIR
from .logging_config import setup_logging

from .rag_service import rag_answer
from .digest_service import list_digest_files, load_digest_html


app = FastAPI()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
setup_logging()
import logging
logger = logging.getLogger(__name__)

digest_app = build_digest_graph()
logger.info("Digest graph created")

# สร้าง / compile กราฟครั้งเดียว
_digest_graph = build_digest_graph()
digest_app = _digest_graph.compile()


def split_sections(md_text: str):
    """
    แยก markdown ออกเป็น 4 ส่วน:
    - header_md: ก่อน '## หมวดที่ 1'
    - ai_md: '## หมวดที่ 1' -> ก่อน '## หมวดที่ 2'
    - cyber_md: '## หมวดที่ 2' -> ก่อน '### สรุปภาพรวมวันนี้'
    - summary_md: ตั้งแต่ '### สรุปภาพรวมวันนี้' เป็นต้นไป
    """
    t = md_text

    i1 = t.find("## หมวดที่ 1")
    i2 = t.find("## หมวดที่ 2")
    i3 = t.find("### สรุปภาพรวมวันนี้")

    if i1 == -1:
        return t.strip(), "", "", ""

    header_md = t[:i1].strip()

    if i2 == -1:
        return header_md, t[i1:].strip(), "", ""

    ai_md = t[i1:i2].strip()

    if i3 == -1:
        return header_md, ai_md, t[i2:].strip(), ""

    cyber_md = t[i2:i3].strip()
    summary_md = t[i3:].strip()

    return header_md, ai_md, cyber_md, summary_md


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    หน้า Digest วันนี้ – ใช้กราฟหลาย node รัน pipeline
    """
    logger.info("Running digest graph for homepage")
    state = await digest_app.ainvoke({})
    logger.info("Digest graph finished")

    ai_html = state.get("ai_html", "ไม่มีข้อมูล")
    cyber_html = state.get("cyber_html", "ไม่มีข้อมูล")
    summary_html = state.get("summary_html", "")

    logger.info(
        "HTTP / : got sections len(ai)=%d, len(cyber)=%d, len(summary)=%d",
        len(ai_html), len(cyber_html), len(summary_html),
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "ai_html": ai_html,
            "cyber_html": cyber_html,
            "summary_html": summary_html,
        },
    )


@app.get("/qa", response_class=HTMLResponse)
async def qa_page(request: Request, q: str | None = None):
    answer = None
    sources = []

    if q:
        try:
            answer, sources = rag_answer(q)
        except Exception as e:
            answer = f"เกิดข้อผิดพลาดขณะประมวลผลคำถาม: {e}"

    return templates.TemplateResponse(
        "qa.html",
        {
            "request": request,
            "question": q or "",
            "answer": answer,
            "sources": sources,
        },
    )

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/archive")
async def archive_list(request: Request):
    files = list_digest_files()
    logger.info("HTTP /archive : found %d digest files", len(files))
    items = []
    for path in files:
        date_str = path.stem  # 2025-11-24
        items.append({"date": date_str})
    return templates.TemplateResponse(
        "archive_list.html",
        {"request": request, "items": items},
    )


@app.get("/archive/{date_str}")
async def archive_detail(request: Request, date_str: str):
    logger.info("HTTP /archive/%s : loading digest", date_str)
    html = load_digest_html(date_str)
    if html is None:
        logger.warning("HTTP /archive/%s : digest not found", date_str)
        raise HTTPException(status_code=404, detail="ไม่พบ Digest วันที่ระบุ")

    return templates.TemplateResponse(
        "archive_detail.html",
        {"request": request, "date_str": date_str, "digest_html": html},
    )
