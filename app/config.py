from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv  # <<< เพิ่มบรรทัดนี้
import logging
logger = logging.getLogger(__name__)

# โหลดค่าในไฟล์ .env เข้ามาเป็น environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(env_path)  # <<< และบรรทัดนี้

# --- API keys / env ---

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY ยังไม่ถูกตั้งใน environment/.env")

# --- Paths ---
DIGESTS_DIR = BASE_DIR / "digests"
DIGESTS_DIR.mkdir(parents=True, exist_ok=True)

RAG_STORE_DIR = BASE_DIR / "rag_store"
RAG_STORE_DIR.mkdir(exist_ok=True)

TEMPLATES_DIR = BASE_DIR / "templates"

# --- RSS feeds ---

AI_FEEDS = [
    "https://news.ycombinator.com/rss",
    "https://www.technologyreview.com/feed/",
    # เติม feed ด้าน AI/LLM/Agent ที่นายชอบได้เลย
]

CYBER_FEEDS = [
    "https://feeds.feedburner.com/TheHackersNews",
    "https://www.bleepingcomputer.com/feed/",
    # เติม feed ด้าน security เพิ่มได้
]

# จำนวนข่าวต่อ feed ที่จะดึงมาใช้สร้าง digest
ITEMS_PER_FEED = 5
