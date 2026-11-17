"""Offline evals (no model calls). python evals/run.py -> evals/results.json (port of the old run.mjs)."""

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from braylo_si_rag.agent import SCOPE_MIN, answer, on_topic, redact_pii  # noqa: E402
from braylo_si_rag.guards import input_guard  # noqa: E402
from braylo_si_rag.kb import KB  # noqa: E402
from braylo_si_rag.metrics import (  # noqa: E402
    citation_overlap, mean, precision_at_k, r3, recall_at_k, reciprocal_rank,
)
from braylo_si_rag.obs import pct  # noqa: E402
from braylo_si_rag.retrieve import BY_ID, bm25_ids  # noqa: E402
from braylo_si_rag.tools import route_intent  # noqa: E402

SET = json.loads((ROOT / "data" / "queries.json").read_text(encoding="utf-8"))
GUARD = json.loads((ROOT / "data" / "guard-cases.agent14.json").read_text(encoding="utf-8"))


def relevant(must):
    return [c["id"] for c in KB["chunks"] if all(m.lower() in c["text"].lower() for m in must)]


L = [{**r, "relevant": relevant(r["must"])} for r in SET["retrieval"]]
empty = [q["id"] for q in L if not q["relevant"]]
if empty:
    raise SystemExit(f"queries with no relevant chunk: {empty}")


def score(qs, rank):
    o = {"n": len(qs)}
    for k in (1, 3, 5):
        o[f"precision@{k}"] = r3(mean([precision_at_k(rank(q), q["relevant"], k) for q in qs]))
        o[f"recall@{k}"] = r3(mean([recall_at_k(rank(q), q["relevant"], k) for q in qs]))
    o["hit_rate_at_5"] = r3(mean([1 if any(i in q["relevant"] for i in rank(q)[:5]) else 0 for q in qs]))
    o["mrr"] = r3(mean([reciprocal_rank(rank(q), q["relevant"]) for q in qs]))
    return o


def bm(q):
    return bm25_ids(q["q"], 50)


rel_counts = [len(q["relevant"]) for q in L]
retrieval = {
    "bm25": {
        "all": score(L, bm),
        "dev": score([q for q in L if q["split"] == "dev"], bm),
        "test": score([q for q in L if q["split"] == "test"], bm),
    },
    "relevant_per_query": {"min": min(rel_counts), "max": max(rel_counts), "mean": r3(mean(rel_counts))},
    "vector_bge_base": "not yet run (needs Workers AI embeddings; quota resets ~7 PM CT Oct 28)",
    "hybrid_rrf": "not yet run (needs vector side)",
}

scope_cases = SET["scope"] + [{"q": q["q"], "on": True, "split": q["split"]} for q in L]


def sc(split):
    c = [x for x in scope_cases if x["split"] == split]
    ok = sum(1 for x in c if on_topic(x["q"])["on"] == x["on"])
    off, on = [x for x in c if not x["on"]], [x for x in c if x["on"]]
    return {
        "n": len(c),
        "accuracy": r3(ok / len(c)),
        "off_topic_blocked": f"{sum(1 for x in off if not on_topic(x['q'])['on'])}/{len(off)}",
        "on_topic_false_blocks": f"{sum(1 for x in on if not on_topic(x['q'])['on'])}/{len(on)}",
    }


att = [c for c in GUARD if c["label"] == "attack"]
ben = [c for c in GUARD if c["label"] != "attack"]


def blocked(c):
    return input_guard(c["prompt"] + " " + (c.get("doc") or ""))["blocked"]


injection = {
    "source": "agent-14 evals/cases.json",
    "attacks_blocked": f"{sum(1 for c in att if blocked(c))}/{len(att)}",
    "benign_false_positives": f"{sum(1 for c in ben if blocked(c))}/{len(ben)}",
    "braylo_queries_false_positives": f"{sum(1 for q in L if input_guard(q['q'])['blocked'])}/{len(L)}",
    "note": "Rules were tuned on the agent-14 set, so its numbers are optimistic.",
}


def pii_ok(p):
    o = redact_pii(p["in"])
    return o == p["in"] if p.get("same") else p["must_not"] not in o


pii = {"n": len(SET["pii"]), "passed": sum(1 for p in SET["pii"] if pii_ok(p))}


def routed(t):
    r = route_intent(t["q"])
    return r["tool"] if r else None


tools = {
    "router": "rules (fallback path)",
    "n": len(SET["tools"]),
    "accuracy": r3(sum(1 for t in SET["tools"] if routed(t) == t["tool"]) / len(SET["tools"])),
    "misses": [t["q"] for t in SET["tools"] if routed(t) != t["tool"]],
    "llm_function_calling": "not yet run (needs Workers AI)",
}


class NoModelHost:
    has_ai = False
    has_vectors = False
    has_kv = False

    def now(self):
        return time.perf_counter() * 1000

    def epoch(self):
        return int(time.time() * 1000)

    def trace_id(self):
        return "0" * 32

    def limited(self):
        return False


e2e, spans, totals = [], {}, []
for q in L:
    r = answer(NoModelHost(), q["q"], no_model=True)
    cited = [BY_ID[c["id"]]["text"] for c in r.get("citations") or []]
    text = r.get("answer") or ""
    import re

    e2e.append({
        "id": q["id"],
        "contains_gold": 1 if all(m.lower() in text.lower() for m in q["must"]) else 0,
        "overlap": citation_overlap(re.sub(r"\[\d+\]", "", text), cited),
        "blocked": r.get("blocked"),
    })
    totals.append(r["duration_ms"])
    for s in r["spans"]:
        spans.setdefault(s["name"], []).append(s["duration_ms"])

extractive = {
    "n": len(e2e),
    "answer_contains_gold_phrase": r3(mean([e["contains_gold"] for e in e2e])),
    "citation_overlap_mean": r3(mean([e["overlap"] for e in e2e])),
    "wrongly_blocked": [e["id"] for e in e2e if e["blocked"]],
    "note": "Extractive answers copy sentences from cited chunks, so citation overlap is near 1 by construction; the gold-phrase rate is the informative number.",
}
latency = {
    "where": f"box CPU, Python {platform.python_version()}, extractive path, {len(L)} queries (not Workers edge latency)",
    "total_ms": {"p50": pct(totals, 50), "p95": pct(totals, 95)},
    "per_span_ms": {k: {"p50": pct(v, 50), "p95": pct(v, 95)} for k, v in spans.items()},
}
out = {
    "project": "braylo-si-rag",
    "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
    "kb": {"files": len(KB["built_from"]), "chunks": len(KB["chunks"])},
    "retrieval": retrieval,
    "scope_guard": {"threshold_bm25": SCOPE_MIN, "dev": sc("dev"), "test": sc("test")},
    "injection_guard": injection,
    "pii_redaction": pii,
    "tool_routing": tools,
    "extractive_answers": extractive,
    "latency": latency,
    "not_yet_run": {
        "reason": "Workers AI free daily quota exhausted until ~7 PM CT Oct 28, 2026",
        "items": [
            "vector (bge-base) and hybrid RRF retrieval metrics",
            "LLM function-calling accuracy",
            "LLM answer faithfulness (agent-11 judge)",
            "live Workers latency p50/p95 and token counts",
        ],
    },
}
dest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "evals" / "results.json"
dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out["retrieval"]["bm25"]["all"]), out["injection_guard"], out["extractive_answers"]["answer_contains_gold_phrase"])
