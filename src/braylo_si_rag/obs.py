"""Observability: trace IDs, spans, token counting, published-price cost estimate, percentiles, KV trace store.
Pricing source: developers.cloudflare.com/workers-ai/platform/pricing (read Oct 27, 2026). ESTIMATE only."""
import json, math
from decimal import ROUND_HALF_UP, Decimal

MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
PRICING = {MODEL: {"in_neurons_per_m": 26668, "out_neurons_per_m": 204805}, "usd_per_1k_neurons": 0.011}
IDX, MAX = "idx", 200
dumps = lambda o: json.dumps(o, separators=(",", ":"), ensure_ascii=False)


def to_fixed(x, d):
    """JS Number(x.toFixed(d)): exact decimal expansion, ties round up."""
    return float(Decimal(x).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def round2(x):
    return math.floor(x * 100 + 0.5) / 100


class Tracer:
    def __init__(self, route, now, trace_id, epoch):
        self.route, self.now, self.trace_id, self.epoch = route, now, trace_id, epoch
        self.t0, self.spans = now(), []

    def span(self, name, fn, attrs=None):
        s = {"name": name, "start_ms": round2(self.now() - self.t0), "attrs": attrs if attrs is not None else {}}
        try:
            return fn(s)
        except Exception as e:
            s["error"] = str(e)
            raise
        finally:
            s["duration_ms"] = round2(self.now() - self.t0 - s["start_ms"])
            self.spans.append(s)

    def finish(self, status, **extra):
        return {"trace_id": self.trace_id, "route": self.route, "status": status, "ts": self.epoch(), "duration_ms": round2(self.now() - self.t0), "spans": self.spans, **extra}


def est_tokens(s):
    return math.ceil(len(str(s or "").encode("utf-16-le")) / 2 / 4)


def cost_estimate(model, tin, tout):
    p = PRICING.get(model) if model != "usd_per_1k_neurons" else None
    if not p:
        return None
    neurons = (tin * p["in_neurons_per_m"] + tout * p["out_neurons_per_m"]) / 1e6
    return {"neurons": to_fixed(neurons, 3), "usd": to_fixed(neurons / 1000 * PRICING["usd_per_1k_neurons"], 8), "label": "estimate from published neuron pricing; free tier pays $0"}


def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1)]


def summarize(traces):
    by = {}
    for t in traces:
        by.setdefault(t["route"], []).append(t)
    out = {}
    for r, ts in by.items():
        d = [t["duration_ms"] for t in ts]
        ok = [t for t in ts if t["status"] < 400]
        cost = sum((t.get("cost") or {}).get("usd", 0) or 0 for t in ts)
        tok = [t for t in ts if t.get("tokens")]
        out[r] = {"n": len(ts), "errors": len(ts) - len(ok), "p50_ms": pct(d, 50), "p95_ms": pct(d, 95),
                  "avg_tokens_in": to_fixed(sum(t["tokens"]["in"] for t in tok) / len(tok), 1) if tok else None,
                  "avg_tokens_out": to_fixed(sum(t["tokens"]["out"] for t in tok) / len(tok), 1) if tok else None,
                  "est_cost_usd_per_req": to_fixed(cost / len(tok), 8) if tok else None}
    return out


def store(kv, trace):
    if kv is None:
        return
    kv.put("t:" + trace["trace_id"], dumps(trace))
    raw = kv.get(IDX)
    cur = json.loads(raw) if raw else []
    cur.insert(0, trace["trace_id"])
    nxt = list(dict.fromkeys(cur))[:MAX]
    kv.put(IDX, dumps(nxt))
    for drop in cur[MAX:]:
        kv.delete("t:" + drop)


def recent(kv):
    if kv is None:
        return []
    raw = kv.get(IDX)
    rows = [kv.get("t:" + i) for i in (json.loads(raw) if raw else [])]
    return [json.loads(r) for r in rows if r]
