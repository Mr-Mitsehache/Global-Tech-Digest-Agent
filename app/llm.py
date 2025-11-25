# app/llm.py
from __future__ import annotations

import logging
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from .config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

# central config สำหรับ LLM ทั้งไฟล์
MODEL_NAME = "gemini-2.5-pro"
EMBED_MODEL_NAME = "text-embedding-004"
TEMPERATURE = 0.4


def _ensure_api_key() -> None:
    """กันเคสลืมตั้ง GOOGLE_API_KEY ให้ fail แบบมี log ชัด ๆ"""
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY is missing or empty")
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. "
            "กรุณาตั้งค่าใน .env หรือ environment variables ก่อนรันเซิร์ฟเวอร์"
        )


def get_chat_model() -> ChatGoogleGenerativeAI:
    """คืน chat model (Gemini) สำหรับสรุปข่าว / QA."""
    _ensure_api_key()
    logger.info(
        "Creating ChatGoogleGenerativeAI model=%s temperature=%s",
        MODEL_NAME,
        TEMPERATURE,
    )
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """คืน embedding model สำหรับ RAG."""
    _ensure_api_key()
    logger.info(
        "Creating GoogleGenerativeAIEmbeddings model=%s",
        EMBED_MODEL_NAME,
    )
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
    )
