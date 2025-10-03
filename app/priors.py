from __future__ import annotations
import os, csv, re
from typing import Dict, Optional

try:
    import tldextract
except Exception:
    tldextract = None

from .config import ALLSIDES_PRIORS_PATH

# accept many header variants
_NAME_KEYS   = ("source","outlet","name","publication","organization","site_name","brand","source_name")
_DOMAIN_KEYS = ("domain","site","website","host","url","homepage") 
_RATING_KEYS = ("rating","bias","allsides_rating","allsides_bias","allsides")
_NOTES_KEYS  = ("notes","summary","desc","description","about")
_LINK_KEYS   = ("link","source_url","allsides_url","ref","reference","page","page_url")

# minimal domain to name fallback for common outlets 
DOMAIN_TO_NAME = {
    "cnn.com": "CNN",
    "foxnews.com": "Fox News",
    "apnews.com": "Associated Press",
    "associatedpress.com": "Associated Press",
    "nytimes.com": "New York Times",
    "wsj.com": "Wall Street Journal",
}

_PRIORS_BY_DOMAIN: Dict[str, Dict[str,str]] = {}
_PRIORS_BY_NAME: Dict[str, Dict[str,str]] = {}
_LOADED = False

def _norm_domain(x: str) -> str:
    if not x: return ""
    s = x.strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    s = s.split("/")[0]
    if tldextract is not None and "." in s:
        ext = tldextract.extract("http://" + s if not s.startswith("http") else s)
        s = ".".join([p for p in (ext.domain, ext.suffix) if p])
    return s

def _norm_name(x: str) -> str:
    return (x or "").strip().lower()

def _first(d: dict, keys: tuple[str,...]) -> Optional[str]:
    for k in keys:
        if k in d and d[k]: return str(d[k])
    lk = {k.lower(): k for k in d.keys()}
    for want in keys:
        if want in lk and d[lk[want]]: return str(d[lk[want]])
    return None

def _load() -> None:
    global _LOADED, _PRIORS_BY_DOMAIN, _PRIORS_BY_NAME
    if _LOADED: return
    _LOADED = True

    path = ALLSIDES_PRIORS_PATH
    if not path or not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name   = _first(row, _NAME_KEYS)   or ""
            domain = _first(row, _DOMAIN_KEYS) or ""    
            rating = _first(row, _RATING_KEYS) or ""
            notes  = _first(row, _NOTES_KEYS)  or ""
            link   = _first(row, _LINK_KEYS)   or ""

            name_n   = _norm_name(name)
            domain_n = _norm_domain(domain)

            if not (name_n or domain_n or rating):
                continue

            record = {
                "source": name or domain_n or "",
                "domain": domain_n,
                "rating": rating,
                "origin": "AllSides",
            }
            if notes: record["notes"] = notes
            if link:  record["url"]   = link

            if domain_n:
                _PRIORS_BY_DOMAIN[domain_n] = record
            if name_n:
                _PRIORS_BY_NAME[name_n] = record

def get_prior_for_source(source_or_domain: str) -> Optional[Dict[str,str]]:
    _load()
    s = source_or_domain or ""

    # try domain directly
    dom = _norm_domain(s)
    if dom and dom in _PRIORS_BY_DOMAIN:
        return _PRIORS_BY_DOMAIN[dom]

    # if we got a domain, map it to an outlet name and try name lookup
    if dom and dom in DOMAIN_TO_NAME:
        nm = _norm_name(DOMAIN_TO_NAME[dom])
        rec = _PRIORS_BY_NAME.get(nm)
        if rec:
            # fill in domain if missing
            if not rec.get("domain"):
                rec = dict(rec)
                rec["domain"] = dom
            return rec

    # try name directly ("CNN")
    nm = _norm_name(s)
    if nm and nm in _PRIORS_BY_NAME:
        return _PRIORS_BY_NAME[nm]

    # if it was a full URL and we didn't catch above
    if s.startswith("http"):
        dom2 = _norm_domain(s)
        if dom2 in _PRIORS_BY_DOMAIN:
            return _PRIORS_BY_DOMAIN[dom2]
        if dom2 in DOMAIN_TO_NAME:
            nm2 = _norm_name(DOMAIN_TO_NAME[dom2])
            rec = _PRIORS_BY_NAME.get(nm2)
            if rec:
                if not rec.get("domain"):
                    rec = dict(rec)
                    rec["domain"] = dom2
                return rec

    return None
