# app\rag_service.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import logging
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .config import DIGESTS_DIR, RAG_STORE_DIR
from .llm import get_chat_model, get_embedding_model
from .digest_service import list_digest_files, load_digest_html

logger = logging.getLogger(__name__)

# จะเก็บ FAISS index ไว้ตรงนี้
VECTOR_STORE_PATH = RAG_STORE_DIR / "digest_faiss"


# ---------- Helper: HTML -> Text ----------

def _html_to_plaintext(html: str) -> str:
    """แปลง HTML รายงาน Digest ให้เป็น text สะอาด ๆ สำหรับทำ RAG"""
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    # ล้างบรรทัดว่างเกิน
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _build_documents() -> List[Document]:
    """
    โหลด digest HTML ทุกไฟล์ใน digests/
    แล้วแปลงเป็น Document สำหรับ LangChain
    """
    files = list_digest_files()
    if not files:
        logger.warning("RAG: ไม่พบไฟล์ digest ใน %s", DIGESTS_DIR)

    docs: List[Document] = []
    for path in files:
        try:
            html = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("RAG: อ่านไฟล์ %s ไม่ได้: %s", path, e)
            continue

        text = _html_to_plaintext(html)
        if not text.strip():
            logger.warning("RAG: ไฟล์ %s ไม่มีเนื้อหา", path)
            continue

        # ใช้ชื่อไฟล์ (yyyy-mm-dd.html) เป็น metadata date
        date_str = path.stem

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "date": date_str,
                },
            )
        )

    logger.info("RAG: สร้าง documents จาก digest แล้ว %d รายการ", len(docs))
    return docs


# ---------- Build / Load Vector Store ----------

def build_vector_store(force_rebuild: bool = False) -> FAISS:
    """
    สร้างหรือโหลด FAISS index จาก digest ทั้งหมด
    ถ้า force_rebuild=True จะ rebuild ใหม่เสมอ
    """
    embeddings = get_embedding_model()

    if VECTOR_STORE_PATH.exists() and not force_rebuild:
        logger.info("RAG: โหลด FAISS index จาก %s", VECTOR_STORE_PATH)
        return FAISS.load_local(
            str(VECTOR_STORE_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    logger.info("RAG: สร้าง FAISS index ใหม่จาก digest HTML")
    docs = _build_documents()
    if not docs:
        raise RuntimeError(
            "RAG: ยังไม่มี digest ให้สร้าง index (โฟลเดอร์ digests/ ว่างเปล่า)"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)
    logger.info("RAG: สร้าง chunk แล้ว %d ชิ้น", len(chunks))

    vs = FAISS.from_documents(chunks, embeddings)

    VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(VECTOR_STORE_PATH))
    logger.info("RAG: บันทึก FAISS index ที่ %s แล้ว", VECTOR_STORE_PATH)

    return vs


_vector_store: FAISS | None = None


def get_vector_store() -> FAISS:
    """lazy load + cache vector store ไว้ในโปรเซส"""
    global _vector_store
    if _vector_store is None:
        _vector_store = build_vector_store(force_rebuild=False)
    return _vector_store


# ---------- Prompt สำหรับ QA ----------

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
คุณคือผู้ช่วยวิเคราะห์ข่าวเทคโนโลยีจาก Global Tech Digest

ข้อกำหนด:
- ตอบเป็นภาษาไทยแบบชัดเจน สั้น-กระชับแต่ครบประเด็น
- ใช้ข้อมูลจาก "บริบท" ที่ให้เท่านั้น ห้ามเดาเกินข้อมูล
- ถ้าบริบทไม่มีข้อมูลเพียงพอ ให้ตอบตรง ๆ ว่า Digest ที่มีอยู่ยังไม่ครอบคลุม
- ถ้ามีวันที่ในบริบท ให้ช่วยอ้างอิงวันที่หรือช่วงเวลาให้ด้วย (ถ้าเหมาะสม)
""",
        ),
        (
            "human",
            """
คำถามของผู้ใช้:
{question}

บริบทจาก Digest (อาจหลายวันผสมกัน):
{context}

โปรดตอบคำถามจากบริบทด้านบนเท่านั้น
ถ้าไม่พบข้อมูลที่เกี่ยวข้องมากพอ ให้ตอบว่า "ใน Digest ที่เก็บไว้ยังไม่มีข้อมูลเพียงพอเกี่ยวกับคำถามนี้"
""",
        ),
    ]
)


# ---------- Public: rag_answer ----------

async def rag_answer(question: str) -> Tuple[str, List[Document]]:
    """
    ฟังก์ชันหลักสำหรับ RAG QA:
    - รับคำถาม
    - ดึงบริบทจาก digest ที่เก็บไว้
    - ให้ LLM สรุปคำตอบจากบริบท
    คืน:
    - answer: str
    - docs: list[Document] ที่ถูก retrieve
    """
    clean_q = (question or "").strip()
    if not clean_q:
        raise ValueError("question must not be empty")

    logger.info("RAG: answer question = %s", clean_q)

    vs = get_vector_store()
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    # LangChain รุ่นใหม่ retriever เป็น Runnable → ใช้ ainvoke
    docs = await retriever.ainvoke(clean_q)
    if not docs:
        logger.warning("RAG: ไม่พบเอกสารที่เกี่ยวข้องกับคำถามนี้")
        return (
            "ใน Digest ที่เก็บไว้ยังไม่พบข้อมูลที่เกี่ยวข้องกับคำถามนี้มากพอ จึงไม่สามารถตอบได้อย่างมั่นใจ.",
            [],
        )


    # รวม context แบบอ่านง่าย + มีวันที่
    context_parts: List[str] = []
    for d in docs:
        date = d.metadata.get("date") or d.metadata.get("source") or "ไม่ทราบวันที่"
        context_parts.append(f"[{date}]\n{d.page_content}")
    context_text = "\n\n---\n\n".join(context_parts)

    model = get_chat_model()
    chain = RAG_PROMPT | model
    resp = await chain.ainvoke(
        {
            "question": clean_q,
            "context": context_text,
        }
    )
    answer = getattr(resp, "content", str(resp))

    logger.info("RAG: ตอบคำถามเรียบร้อย (len=%d)", len(answer or ""))

    return answer, docs
