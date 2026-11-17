"""HTTP routes. Pure router: returns (status, json_or_None, headers); status 0 = serve the page."""

import json
from pathlib import Path
from urllib.parse import parse_qs

from .agent import answer
from .kb import KB, cap
from .obs import recent, store, summarize
from .retrieve import hybrid
from .tools import TOOL_SCHEMAS, TOOLS

ROOT = Path(__file__).resolve().parents[2]
RESULTS = json.loads((ROOT / "evals" / "results.json").read_text(encoding="utf-8"))
RING = []  # per-isolate fallback when no KV is bound


def handle(host, method, path, query="", body=None):
    if path == "/":
        return 0, None, {}
    if path == "/api/health":
        return (
            200,
            {
                "ok": True,
                "chunks": len(KB["chunks"]),
                "ai": host.has_ai,
                "vectors": host.has_vectors,
            },
            {},
        )
    if path == "/api/tools":
        return 200, TOOL_SCHEMAS, {}
    if path == "/api/eval":
        return 200, RESULTS, {}
    if path == "/api/metrics":
        t = recent(host.kv) if host.kv is not None else RING
        return (
            200,
            {
                "n": len(t),
                "by_route": summarize(t),
                "store": "kv" if host.kv is not None else "isolate memory",
            },
            {},
        )
    if path == "/api/search":
        q = (parse_qs(query, keep_blank_values=True).get("q") or [""])[0]
        if not cap(q):
            return 400, {"error": "q required (<=600 chars)"}, {}
        return 200, hybrid(host, q, 5), {}
    if method != "POST":
        return 404, {"error": "not found"}, {}
    if host.limited():
        return 429, {"error": "rate limited: 5 requests/min"}, {}
    b = body if isinstance(body, dict) else {}
    if path == "/api/tool":
        name = b.get("name")
        f = TOOLS.get(name) if isinstance(name, str) else None
        if not f:
            return 400, {"error": "unknown tool"}, {}
        return 200, f(b.get("args") or {}), {}
    if path == "/api/chat":
        if not cap(b.get("message")):
            return 400, {"error": "message required (<=600 chars)"}, {}
        r = answer(host, b["message"])
        trace = {
            k: r.get(k)
            for k in (
                "trace_id",
                "route",
                "status",
                "ts",
                "duration_ms",
                "spans",
                "tokens",
                "cost",
            )
        }
        if host.kv is not None:
            try:
                store(host.kv, trace)
            except Exception:
                pass
        else:
            RING.insert(0, trace)
            del RING[200:]
        return r["status"], r, {"x-trace-id": r["trace_id"]}
    return 404, {"error": "not found"}, {}
