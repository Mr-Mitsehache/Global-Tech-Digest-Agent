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
DIGESTS_DIR.mkdir(exist_ok=True)

RAG_STORE_DIR = BASE_DIR / "rag_store"
RAG_STORE_DIR.mkdir(exist_ok=True)

TEMPLATES_DIR = BASE_DIR / "templates"

# --- RSS feeds ---

AI_FEEDS = [
    "https://www.technologyreview.com/feed/", #เทคโนโลยี
    "https://magazine.sebastianraschka.com/feed", #การวิจัย Machine Learning และ AI
    "https://ai-techpark.com/category/ai/feed/", #AI, ML, IoT, ข่าวความปลอดภัยทางไซเบอร์และการวิเคราะห์แนวโน้ม, บทสัมภาษณ์
    "https://www.artificialintelligence-news.com/feed/rss/", #ข่าวปัญญาประดิษฐ์
    "https://www.wired.com/feed/tag/ai/latest/rss",
]

CYBER_FEEDS = [
    "https://feeds.feedburner.com/TheHackersNews", #สายข่าว Cybersecurity ที่ดังสุด ๆ
    "https://www.cisa.gov/cybersecurity-advisories/all.xml", #US-CERT / CISA (ถ้าอยากเน้น advisory)
    "https://podcast.darknetdiaries.com/", #เล่าเคส hack จริง ๆ, story-based
    "https://krebsonsecurity.com/feed/", #สืบสวนคดี
    "https://www.securityweek.com/feed/", #SecurityWeek
    
]

# จำนวนข่าวต่อ feed ที่จะดึงมาใช้สร้าง digest
ITEMS_PER_FEED = 5
