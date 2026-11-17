"""BM25 lexical retriever (k1=1.5, b=0.75). No model calls."""
import math
from .metrics import tokens


def bm25_index(chunks, k1=1.5, b=0.75):
    docs = [tokens(c["text"]) for c in chunks]
    N = len(docs)
    avg = sum(len(d) for d in docs) / N
    df = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1

    def search(query):
        q = tokens(query)
        out = []
        for i, d in enumerate(docs):
            s = 0
            for t in q:
                f = d.count(t)
                if not f:
                    continue
                n = df[t]
                s += math.log(1 + (N - n + 0.5) / (n + 0.5)) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * len(d) / avg))
            out.append({"id": chunks[i]["id"], "score": s})
        return sorted(out, key=lambda h: (-h["score"], h["id"]))

    return search
