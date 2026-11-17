"""Retrieval and faithfulness metrics. Pure functions, no model calls."""
import math, re

STOP = set("a an and are as at be by can for from how in is it its of on or so that the their them then this to was what when where which who why with does do into than such each name two own other".split(" "))
_TOK = re.compile(r"[a-z0-9]+")
_STEM = re.compile(r"(ing|es|s)$")


def tokens(s):
    return [_STEM.sub("", t, count=1) for t in _TOK.findall(str(s).lower()) if len(t) > 2 and t not in STOP]


def precision_at_k(ranked, relevant, k):
    R = set(relevant)
    return sum(1 for i in ranked[:k] if i in R) / k


def recall_at_k(ranked, relevant, k):
    R = set(relevant)
    return sum(1 for i in ranked[:k] if i in R) / len(R) if R else 0


def reciprocal_rank(ranked, relevant):
    R = set(relevant)
    for i, x in enumerate(ranked):
        if x in R:
            return 1 / (i + 1)
    return 0


def mean(xs):
    return sum(xs) / len(xs) if xs else 0


def r3(x):
    """JS Math.round(x * 1000) / 1000 (half rounds up)."""
    return math.floor(x * 1000 + 0.5) / 1000


def citation_overlap(answer, cited_texts):
    """Share of the answer's content tokens that appear in the chunks it cites."""
    a = tokens(answer)
    src = {t for c in cited_texts for t in tokens(c)}
    return sum(1 for t in a if t in src) / len(a) if a else 0


def auc(scores, labels):
    """ROC AUC via pairwise comparison (ties count half)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1 if p > n else 0.5 if p == n else 0
    return s / (len(pos) * len(neg)) if pos and neg else 0
