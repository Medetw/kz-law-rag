"""Download act texts from adilet.zan.kz (the official legal database of Kazakhstan).

Only ``/api/documents/by-ngr/{NGR}`` is used: the endpoint the public act page itself calls, explicitly
``Allow``-ed for all user agents in https://adilet.zan.kz/robots.txt. One request per act and language,
cached on disk, with a pause between requests. Answers of this project always link back to the article on adilet.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx

from kzlaw.corpus import ADILET_LANG, Code, Lang

log = logging.getLogger(__name__)

API = "https://adilet.zan.kz/api/documents/by-ngr/{ngr}?language={lang}"
USER_AGENT = "kz-law-rag/0.1 (+https://github.com/Medetw/kz-law-rag; cites articles with links to adilet)"
PAUSE_S = 3.0


def raw_path(raw_dir: Path, code: Code, lang: Lang) -> Path:
    return raw_dir / f"{code.ngr}_{ADILET_LANG[lang]}.json"


def fetch_html(code: Code, lang: Lang, raw_dir: Path, client: httpx.Client | None = None) -> str:
    """Return the act HTML, downloading it once and caching the JSON response."""
    path = raw_path(raw_dir, code, lang)
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        own = client is None
        http = client or httpx.Client(timeout=120, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            log.info("downloading %s (%s) from adilet", code.ngr, lang)
            resp = http.get(API.format(ngr=code.ngr, lang=ADILET_LANG[lang]))
            resp.raise_for_status()
            path.write_bytes(resp.content)
            time.sleep(PAUSE_S)
        finally:
            if own:
                http.close()
    doc = json.loads(path.read_text(encoding="utf-8"))["document"]
    text: str = doc["text_content"]
    if not text:
        raise ValueError(f"{code.ngr} ({lang}): adilet returned no text")
    return text
