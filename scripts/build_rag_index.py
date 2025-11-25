# scripts/build_rag_index.py
from __future__ import annotations

import sys
from pathlib import Path

# ให้แน่ใจว่า project-root อยู่ใน sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag_service import build_rag_index  # noqa: E402


if __name__ == "__main__":
    build_rag_index()
