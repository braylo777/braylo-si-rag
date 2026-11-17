"""Builds the knowledge base: sources/*.{md,txt} -> chunks (<=700 chars, heading-aware, 1-line overlap) -> data/kb.json + evals/kb.json.
sources/ holds only the public site pages (home EN/ES, book, terms, privacy) from the latest preview build; the street address is replaced.
Usage: python3 scripts/ingest.py [--out DIR]"""

import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TITLES = {
    "site_index": "Site: home (EN)",
    "site_index.es": "Site: home (ES)",
    "site_book": "Site: book a call",
    "site_terms": "Site: terms",
    "site_privacy": "Site: privacy",
}
_JS_WS = "\\t\\n\\v\\f\\r \\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff"
_TRIM = re.compile(f"^[{_JS_WS}]+|[{_JS_WS}]+$")
_utf16 = lambda s: len(s.encode("utf-16-le")) // 2


def _slice16(s, n):
    return s.encode("utf-16-le")[: n * 2].decode("utf-16-le", "ignore")


def chunk(text, mx=700):
    lines = [l for l in (_TRIM.sub("", x) for x in text.split("\n")) if l]
    out, cur, head = [], [], ""

    def flush():
        if cur:
            out.append(
                (head + "\n" if head and cur[0] != head else "") + "\n".join(cur)
            )

    for l in lines:
        if re.match(r"^#{1,4} ", l):
            flush()
            cur = []
            head = re.sub(r"^#+ ", "", l)
            cur.append(head)
            continue
        if _utf16("\n".join(cur)) + _utf16(l) > mx and cur:
            last = cur[-1]
            flush()
            cur = [last] if _utf16(last) < 200 else []
        cur.append(_slice16(l, mx) if _utf16(l) > mx else l)
    flush()
    return out


def build(root=ROOT):
    files = sorted(p.name for p in (root / "sources").iterdir())
    chunks = []
    for f in files:
        base = re.sub(r"\.(md|txt)$", "", f)
        doc = TITLES.get(base) or re.sub(
            r"[_-]", " ", re.sub(r"^\d{4}-\d\d-\d\d_", "", base)
        )
        for i, text in enumerate(
            chunk((root / "sources" / f).read_text(encoding="utf-8"))
        ):
            chunks.append(
                {
                    "id": f"{re.sub('[^a-z0-9]+', '-', base, flags=re.I).lower()}#{i + 1}",
                    "doc": doc,
                    "source": "sources/" + f,
                    "lang": "es" if base.endswith(".es") else "en",
                    "text": text,
                }
            )
    return {"built_from": files, "chunks": chunks}


if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT
    kb = build()
    (out / "data").mkdir(exist_ok=True)
    (out / "data" / "kb.json").write_text(
        json.dumps(kb, separators=(",", ":"), ensure_ascii=False)
    )
    print(len(kb["built_from"]), "files ->", len(kb["chunks"]), "chunks")
