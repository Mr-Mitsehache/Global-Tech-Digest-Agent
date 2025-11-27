# debug_feeds.py
from app.config import AI_FEEDS, CYBER_FEEDS, ITEMS_PER_FEED
from app.digest_service import fetch_rss_items

def main():
    print("=== AI FEEDS ===")
    for url in AI_FEEDS:
        items = fetch_rss_items(url, max_items=ITEMS_PER_FEED)
        print(f"[AI] {url} -> {len(items)} items")
        for it in items[:3]:
            print("   -", it["title"])
        print()

    print("=== CYBER FEEDS ===")
    for url in CYBER_FEEDS:
        items = fetch_rss_items(url, max_items=ITEMS_PER_FEED)
        print(f"[CYBER] {url} -> {len(items)} items")
        for it in items[:3]:
            print("   -", it["title"])
        print()

if __name__ == "__main__":
    main()
