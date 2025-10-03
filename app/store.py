import os, json, hashlib
from typing import Dict, Any, Iterable, List
from .config import STORE_PATH, DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(STORE_PATH):
    open(STORE_PATH, "a").close()

def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

def upsert_article(doc: Dict[str, Any]) -> str:
    doc_id = _hash_url(doc["url"])
    doc["id"] = doc_id
    docs = list(iter_docs())
    seen = {d["id"]: d for d in docs}
    seen[doc_id] = doc
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        for d in seen.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return doc_id

def iter_docs() -> Iterable[Dict[str, Any]]:
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            yield json.loads(line)

def get_all() -> List[Dict[str, Any]]:
    return list(iter_docs())