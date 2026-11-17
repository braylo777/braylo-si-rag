"""Hybrid retrieval: BM25 (in-Worker) + Workers AI embeddings in Vectorize, fused with reciprocal rank fusion (RRF, k=60).
If the vector side is unavailable (no binding, quota, error) it degrades to BM25 and says so."""

from .bm25 import bm25_index
from .kb import EMBED, KB
from .obs import to_fixed

_bm25 = bm25_index(KB["chunks"])
BY_ID = {c["id"]: c for c in KB["chunks"]}


def rrf(lists, k=60):
    s = {}
    for lst in lists:
        for i, cid in enumerate(lst):
            s[cid] = s.get(cid, 0) + 1 / (k + i + 1)
    return [
        {"id": cid, "score": sc}
        for cid, sc in sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def bm25_ids(q, n=20):
    return [h["id"] for h in _bm25(q) if h["score"] > 0][:n]


def vector_ids(host, q, n=20):
    e = host.ai(EMBED, {"text": [q]})
    r = host.vectors_query(e["data"][0], n)
    return [m["id"] for m in r["matches"]]


def hybrid(host, q, k=5):
    lex = bm25_ids(q)
    vec, vector_error = None, None
    if host.has_ai and host.has_vectors:
        try:
            vec = vector_ids(host, q)
        except Exception as e:
            vector_error = str(e)
    ranked = (
        rrf([lex, vec])
        if vec is not None
        else [{"id": cid, "score": 1 / (60 + i + 1)} for i, cid in enumerate(lex)]
    )
    return {
        "mode": "hybrid" if vec is not None else "bm25",
        "vector_error": vector_error,
        "hits": [
            {**BY_ID.get(h["id"], {}), "score": to_fixed(h["score"], 5)}
            for h in ranked[:k]
        ],
    }
