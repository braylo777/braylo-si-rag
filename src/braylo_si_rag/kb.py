"""Knowledge base (public site copy only), built by scripts/ingest.py into data/kb.json."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KB = json.loads((ROOT / "data" / "kb.json").read_text(encoding="utf-8"))
MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
EMBED = "@cf/baai/bge-base-en-v1.5"  # 768-dim, matches the Vectorize index
SITE = "https://braylo-si.pages.dev"


def cap(s, n=600):
    return isinstance(s, str) and bool(s.strip()) and len(s) <= n
