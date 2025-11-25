# Global Tech Digest Agent (AI & Cybersecurity)

โปรเจกต์นี้คือ **เว็บสรุปข่าวเทคโนโลยีด้าน AI / LLM / Agentic และ Cybersecurity**  
ทำงานด้วย:

- FastAPI + Jinja2 templates (เว็บ)
- LangChain + Gemini 2.5 (สรุปข่าว / ตอบคำถาม)
- RSS feeds (ดึงข่าวจริง)
- ChromaDB (RAG จาก digest ที่บันทึกไว้)
- โครงสร้างโค้ดแยกเป็นสัดส่วน (config, LLM, digest, RAG, web)

เหมาะไว้ใช้เป็น “ข่าวกรอง AI & Cyber ส่วนตัว” + ใช้เป็น sandbox ฝึก LangChain / RAG / FastAPI ไปพร้อมกัน

---

## ฟีเจอร์หลัก

- 🔎 ดึงข่าวจาก RSS feeds ที่เกี่ยวกับ:
  - AI / LLM / Agentic
  - Cybersecurity
- 📰 สร้างสรุปข่าวรายวัน เป็นภาษาไทย อ่านง่าย แบ่งหมวด:
  - หมวดที่ 1: ข่าวด้าน AI / LLM / Agentic (มีหมวดย่อย Infra, Apps, Policy ฯลฯ)
  - หมวดที่ 2: ข่าวด้าน Cybersecurity (APT, Ransomware, Zero-day, Privacy ฯลฯ)
  - สรุปภาพรวมประจำวัน
- 📂 เก็บสรุปทั้งหมดเป็นไฟล์ Markdown ในโฟลเดอร์ `digests/` (เป็น archive ส่วนตัว)
- 🧠 ใช้ RAG (Chroma + LangChain) สร้างหน้า **Q&A trend**
  - ถามแนว “ในช่วงที่ผ่านมา นักลงทุนนิยมลงทุน AI ด้านไหนมากที่สุด?”
  - หรือ “เทรนด์การโจมตี ransomware จากข่าวเก่าที่เก็บไว้เป็นยังไงบ้าง?”
- 🌐 UI หน้าเว็บเป็น Dark theme + Tailwind ผ่าน CDN
  - หน้า Digest วันนี้: `/`
  - หน้า Q&A trend: `/qa`

---

## โครงสร้างโปรเจกต์

```text
project-root/
├─ app/
│  ├─ __init__.py
│  ├─ config.py           # env, path ต่าง ๆ, RSS feed list
│  ├─ llm.py              # สร้าง Chat + Embedding model (Gemini)
│  ├─ digest_service.py   # ดึง RSS + สร้างสรุปข่าว + เซฟ archive
│  ├─ rag_service.py      # RAG index + RAG QA จาก archive
│  └─ web_app.py          # FastAPI routes ("/", "/qa")
│
├─ templates/
│  ├─ index.html          # หน้า Digest วันนี้
│  └─ qa.html             # หน้า Q&A trend จาก archive
│
├─ digests/               # ไฟล์ .md รายวัน (archive)
├─ rag_store/             # โฟลเดอร์ Chroma vector store (สร้างโดยสคริปต์)
│
├─ scripts/
│  └─ build_rag_index.py  # สคริปต์สร้าง / อัปเดต RAG index
│
├─ requirements.txt
└─ .env                   # GOOGLE_API_KEY ฯลฯ
