import requests, time
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import NEWSAPI_KEY
from .store import upsert_article

# NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

def _headers():
    return {"X-Api-Key": NEWSAPI_KEY} if NEWSAPI_KEY else {}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _fetch(query: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "page": page,
    }
    r = requests.get(NEWSAPI_ENDPOINT, params=params, headers=_headers(), timeout=20)
    r.raise_for_status()
    return r.json()

def ingest_query(query: str, max_articles: int = 20) -> List[str]:
    if not NEWSAPI_KEY:
        return []
    ids = []
    page, got = 1, 0
    while got < max_articles:
        data = _fetch(query=query, page=page, page_size=min(100, max_articles - got))
        articles = data.get("articles", [])
        if not articles: break
        for a in articles:
            doc = {
                "url": a.get("url"),
                "title": a.get("title"),
                "source": (a.get("source") or {}).get("name"),
                "published_at": a.get("publishedAt"),
                "text": (a.get("content") or "")[:10000],
                "raw": a,
                "status": "ingested"
            }
            if not doc["url"]:
                continue
            doc_id = upsert_article(doc)
            ids.append(doc_id)
            got += 1
            if got >= max_articles: break
        page += 1
        time.sleep(0.2)
    return ids