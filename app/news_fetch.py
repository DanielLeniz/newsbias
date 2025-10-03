from typing import Dict, Optional
import re, httpx, trafilatura
from readability import Document
import tldextract

UA = "Mozilla/5.0 (compatible; BiasDemo/1.0)"

async def fetch_html(url: str, timeout: float = 12.0) -> str:
    async with httpx.AsyncClient(follow_redirects=True, headers={"User-Agent": UA}, timeout=timeout) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.text

def _clean(s: Optional[str]) -> str:
    if not s: return ""
    return re.sub(r"\s+", " ", s).strip()

def _domain(url: str) -> str:
    e = tldextract.extract(url)
    return ".".join([p for p in [e.domain, e.suffix] if p])

async def extract_article(url: str) -> Dict[str, str]:
    html = await fetch_html(url)

    # try trafilatura
    meta = trafilatura.extract_metadata(html)
    body = trafilatura.extract(html, include_comments=False, include_tables=False, favor_precision=True)
    if body:
        return {
            "url": url,
            "source": _domain(url),
            "title": _clean(meta.title if meta else None),
            "text": _clean(body),
        }

    # fallback: readability
    doc = Document(html)
    title = _clean(doc.short_title())
    text = _clean(re.sub(r"<[^>]+>", " ", doc.summary(html_partial=True)))
    return {"url": url, "source": _domain(url), "title": title, "text": text}
