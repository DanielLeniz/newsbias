from __future__ import annotations
import os, re, logging
from typing import Dict

log = logging.getLogger("uvicorn.error")

# feature flag; turn LLM summarization on/off via env
USE_LLM = str(os.getenv("SUMMARY_ENABLED", "0")).lower() in {"1", "true", "yes"}

SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_MAX_INPUT_CHARS = 12000   # trim very long articles to keep latency/cost low
_DEFAULT_MAX_WORDS = 100   # target summary length

def _extractive_fallback(text: str, max_words: int = _DEFAULT_MAX_WORDS) -> str:
    """
    Lightweight summary: take first few sentences until ~N words
    """
    text = (text or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)[:_MAX_INPUT_CHARS]
    sents = re.split(r"(?<=[\.\!\?])\s+", text)
    out, wc = [], 0
    for s in sents:
        w = len(s.split())
        out.append(s)
        wc += w
        if wc >= max_words:
            break
    return " ".join(out).strip()

def summarize(text: str, max_words: int = _DEFAULT_MAX_WORDS) -> Dict[str, str]:
    """
    Returns {"text": "<summary>"}.
    Uses OpenAI when SUMMARY_ENABLED=1 and OPENAI_API_KEY is set
    """
    text = (text or "").strip()
    if not text:
        return {"text": ""}

    # if LLM disabled or no key, use fallback
    if not USE_LLM or not OPENAI_API_KEY:
        return {"text": _extractive_fallback(text, max_words)}

    # try OpenAI; on any error, fall back 
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # keep prompt short; pass article as the main content
        system_msg = (
            "You are a neutral news assistant. Write a concise, faithful summary in third person. "
            "Avoid opinionated adjectives, speculation, or instructions; no bullet points unless asked. "
            "Include who/what/when/where, and key context if essential. Keep it objective."
        )
        user_msg = (
            f"Summarize the following article in ~{max_words} words.\n\n"
            f"--- ARTICLE START ---\n{ text[:_MAX_INPUT_CHARS] }\n--- ARTICLE END ---"
        )

        resp = client.chat.completions.create(
            model=SUMMARY_MODEL,
            temperature=0.2,
            max_tokens=300,            
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        summary = (resp.choices[0].message.content or "").strip()
        summary = re.sub(r"\s+", " ", summary)
        return {"text": summary}
    except Exception as e:
        log.warning(f"LLM summarization failed; falling back. Error: {e}")
        return {"text": _extractive_fallback(text, max_words)}
