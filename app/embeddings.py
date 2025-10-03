from typing import List
import hashlib, math, struct

def _hash_to_vec(text: str, dim: int = 16) -> List[float]:
    if not text:
        return [0.0]*dim
    h = hashlib.sha256(text.encode("utf-8")).digest()
    buf = (h * ((dim*4)//len(h) + 1))[:dim*4]
    vec = list(struct.unpack("!%df" % dim, bytes(buf)))
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v/norm for v in vec]

def embed_title_lede(title: str, text: str, dim: int = 16) -> List[float]:
    lede = (text or "")[:300]
    return _hash_to_vec((title or "") + " || " + lede, dim=dim)