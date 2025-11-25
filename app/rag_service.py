# app/rag_service.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .config import DIGESTS_DIR, RAG_STORE_DIR
from .llm import get_embedding_model, get_chat_model
import logging
logger = logging.getLogger(__name__)


def load_digest_documents() -> List[Document]:
    docs: List[Document] = []
    for path in sorted(DIGESTS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "date": path.stem,
                    "path": str(path),
                },
            )
        )
    return docs


def build_rag_index() -> None:
    """อ่านทุก digest/* แล้วสร้าง/ทับ Chroma index ใน rag_store/."""
    docs = load_digest_documents()
    if not docs:
        print("[RAG] ไม่มี digest ให้สร้าง index")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)
    print(f"[RAG] {len(docs)} files -> {len(chunks)} chunks")

    embeddings = get_embedding_model()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(RAG_STORE_DIR),
    )
    print(f"[RAG] index สร้างเสร็จที่ {RAG_STORE_DIR}")


def _get_vectorstore() -> Chroma:
    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=str(RAG_STORE_DIR),
        embedding_function=embeddings,
    )


qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "คุณเป็นผู้เชี่ยวชาญด้าน AI และ Cybersecurity.\n"
            "คุณจะได้รับ context ซึ่งเป็นสรุปข่าวรายวันจากหลาย ๆ วัน.\n"
            "ตอบคำถามโดยอ้างอิงจาก context เท่านั้น ถ้าข้อมูลไม่พอให้บอกตรง ๆ.\n"
            "ตอบเป็นภาษาไทย ย่อยง่าย เป็นย่อหน้า + bullet เมื่อเหมาะสม.",
        ),
        (
            "human",
            "คำถาม:\n{question}\n\n"
            "context จากข่าวเก่า:\n{context}",
        ),
    ]
)


def rag_answer(question: str):
    """ตอบคำถามจาก archive ด้วย RAG. คืน (answer, sources)."""
    vs = _get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "ไม่มีข้อมูลใน archive ที่เกี่ยวข้องกับคำถามนี้มากพอ", []

    context = ""
    sources: List[Dict] = []
    for i, d in enumerate(docs, start=1):
        date = d.metadata.get("date", "ไม่ทราบวันที่")
        snippet = d.page_content[:600]
        context += f"[{i}] ({date}) {snippet}\n\n"
        sources.append(
            {
                "index": i,
                "date": date,
                "path": d.metadata.get("path"),
                "snippet": snippet,
            }
        )

    chain = qa_prompt | get_chat_model()
    res = chain.invoke({"question": question, "context": context})
    answer = res.content if hasattr(res, "content") else str(res)
    return answer, sources
