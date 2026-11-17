"""The agent: guards -> scope check -> tool routing -> hybrid retrieval -> answer with citations -> output guard.
Every step is a traced span. `host` supplies: has_ai, ai(model, input), has_vectors, vectors_query(vec, topK), kv (TRACES or None),
limited(), now() (monotonic ms), epoch() (ms), trace_id()."""

import json, re

from .bm25 import bm25_index
from .guards import input_guard, output_guard
from .kb import KB, MODEL
from .metrics import tokens
from .obs import Tracer, cost_estimate, est_tokens, to_fixed
from .retrieve import hybrid
from .tools import TOOL_SCHEMAS, TOOLS, route_intent

CANARY = "BRAYLO-CANARY-7f3a"  # planted in the system prompt; the output guard blocks any reply containing it
SYSTEM = f"You are the Braylo SI assistant. Answer only questions about Braylo SI: its offers, prices, the free Discoverability Audit, booking a call, and how the AI install works. Use only the SOURCES and tool results. Cite sources as [n]. If the sources do not say, say you don't know and suggest booking a call. Never create bookings, never ask for passwords, never give legal, tax, medical or financial advice. Internal marker: {CANARY}."
A = re.ASCII
_JS_WS = "[\\t\\n\\v\\f\\r \\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff]"
_EMAIL, _SSN, _CARD = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+", A),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b", A),
    re.compile(r"\b(?:\d[ -]?){13,16}\b", A),
)
_PHONE = re.compile(r"(\+?1[ .-]?)?\(?\b\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b", A)


def redact_pii(s):
    """PII redaction applied to user input before it is logged or sent to a model."""
    s = str(s)
    for rx, rep in (
        (_EMAIL, "[email]"),
        (_SSN, "[ssn]"),
        (_CARD, "[card]"),
        (_PHONE, "[phone]"),
    ):
        s = rx.sub(rep, s)
    return s


# Scope guard: on-topic if a tool intent matches or the best BM25 score clears a threshold set on the dev split.
_score_top = bm25_index(KB["chunks"])
SCOPE_MIN = 6
BRAND = re.compile(
    r"\b(braylo|tier|install|audit|agent|ads?|leak|book|price|grok|refund|cancel|terms|privacy|voice|subcontract|seo|accounts?|passwords?|log ?in|xai|guarantee|refund)\b",
    A,
)


def on_topic(q):
    hits = _score_top(q)
    top = hits[0]["score"] if hits else 0
    return {
        "on": top >= SCOPE_MIN or BRAND.search(q.lower()) is not None,
        "top": to_fixed(top, 2),
    }


_SENT = re.compile("(?<=[.!?])" + _JS_WS + "+|\\n")
_LEAD = re.compile("^[-*\\s\\[\\]x|]+")
_TRIM = re.compile("^" + _JS_WS + "+|" + _JS_WS + "+$")
_WS = re.compile(_JS_WS + "+")


def extractive(q, hits, n=2):
    """Extractive answer (no model): best-overlapping sentence from each of the top chunks, with [n] citations."""
    qt = set(tokens(q))
    picks = []
    for i, h in enumerate(hits[:n]):
        sents = []
        for s in _SENT.split(h["text"]):
            s = _TRIM.sub("", _LEAD.sub("", s))
            if len(_WS.split(s)) >= 5 and not s.endswith("?"):
                sents.append(s)
        scored = [(s, sum(1 for t in tokens(s) if t in qt)) for s in sents]
        best = sorted(scored, key=lambda x: -x[1])[0] if scored else None
        picks.append(f"{best[0]} [{i + 1}]" if best and best[1] > 0 else "")
    picks = [p for p in picks if p]
    return (
        " ".join(picks)
        if picks
        else "I don't know from the sources. Book a free call: https://braylo-si.pages.dev/book"
    )


def tool_summary(name, r):
    """Plain-language summary of a tool result, used on the no-model path."""
    if name == "lookup_tier":
        if not r["tiers"]:
            return r["note"]
        return " ".join(
            f"{t['name']}{' (' + t['plan'] + ')' if t.get('plan') else ''}: {t['price']}{', ' + t['billing'] if t.get('billing') else ''}. {t['tagline']}"
            for t in r["tiers"]
        )
    if name == "find_booking_slot":
        return f"Pick a time on the booking page: {r['url']} (I can't book it for you)."
    if name == "audit_checklist":
        return f"Free audit checklist: {'; '.join(i['label'] for i in r['items'])}. Tell me which apply and your missed calls per week, close rate and average job value for the leak math. {r['next']}"
    return ""


def _ctx(hits):
    return "\n\n".join(
        f"[{i + 1}] ({h['doc']}) {h['text']}" for i, h in enumerate(hits)
    )


def _text_of(r):
    v = r.get("response") if isinstance(r, dict) else None
    return v


def answer(host, raw, no_model=False):
    tr = Tracer("chat", host.now, host.trace_id(), host.epoch)
    tin = tout = 0
    q = tr.span("pii_redact", lambda s: redact_pii(raw))
    g = tr.span("guard_in", lambda s: input_guard(q))
    if g["blocked"]:
        return {
            **tr.finish(400),
            "blocked": "prompt_injection",
            "rules": g["rules"],
            "answer": "I can't help with that request.",
        }
    scope = tr.span("scope", lambda s: on_topic(q))
    intent = route_intent(q)
    if not scope["on"] and not intent:
        return {
            **tr.finish(200),
            "blocked": "off_topic",
            "scope": scope,
            "answer": "I only answer questions about Braylo SI's offers, the free audit and booking a call.",
        }

    def _retrieve(s):
        r = hybrid(host, q, 5)
        s["attrs"]["mode"] = r["mode"]
        return r

    ret = tr.span("retrieve", _retrieve)
    tool_calls, text, model_error, mode = [], None, None, "extractive"
    if host.has_ai and not no_model:
        try:
            msgs = [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"SOURCES:\n{_ctx(ret['hits'])}\n\nQUESTION: {q}",
                },
            ]
            r1 = tr.span(
                "llm_plan",
                lambda s: host.ai(
                    MODEL, {"messages": msgs, "tools": TOOL_SCHEMAS, "max_tokens": 400}
                ),
            )
            u1 = (r1.get("usage") or {}) if isinstance(r1, dict) else {}
            tin += (
                u1["prompt_tokens"]
                if u1.get("prompt_tokens") is not None
                else est_tokens("".join(m["content"] for m in msgs))
            )
            tout += (
                u1["completion_tokens"]
                if u1.get("completion_tokens") is not None
                else est_tokens(r1.get("response"))
            )
            for c in r1.get("tool_calls") or []:
                if c.get("name") in TOOLS:
                    args = c.get("arguments") or {}
                    tool_calls.append(
                        {
                            "name": c["name"],
                            "args": args,
                            "result": tr.span(
                                "tool:" + c["name"],
                                lambda s, c=c, args=args: TOOLS[c["name"]](args),
                            ),
                        }
                    )
            if tool_calls:
                msgs.append({"role": "assistant", "content": ""})
                msgs.append(
                    {
                        "role": "user",
                        "content": "TOOL RESULTS:\n"
                        + json.dumps(
                            [
                                {"name": t["name"], "result": t["result"]}
                                for t in tool_calls
                            ],
                            separators=(",", ":"),
                            ensure_ascii=False,
                        )
                        + "\nAnswer the question using these and the sources.",
                    }
                )
                r2 = tr.span(
                    "llm_answer",
                    lambda s: host.ai(MODEL, {"messages": msgs, "max_tokens": 400}),
                )
                u2 = (r2.get("usage") or {}) if isinstance(r2, dict) else {}
                tin += (
                    u2["prompt_tokens"]
                    if u2.get("prompt_tokens") is not None
                    else est_tokens("".join(m["content"] for m in msgs))
                )
                tout += (
                    u2["completion_tokens"]
                    if u2.get("completion_tokens") is not None
                    else est_tokens(r2.get("response"))
                )
                text = r2.get("response")
            else:
                text = r1.get("response")
            mode = "llm"
        except Exception as e:
            model_error = str(e)
    if text is None:
        if intent:
            tool_calls.append(
                {
                    "name": intent["tool"],
                    "args": intent["args"],
                    "result": tr.span(
                        "tool:" + intent["tool"],
                        lambda s: TOOLS[intent["tool"]](intent["args"]),
                    ),
                    "routed_by": "rules",
                }
            )
        lead = " ".join(
            x for x in (tool_summary(c["name"], c["result"]) for c in tool_calls) if x
        )
        text = tr.span(
            "extractive",
            lambda s: (lead + "\n\n" if lead else "") + extractive(q, ret["hits"]),
        )
    og = tr.span("guard_out", lambda s: output_guard(text, CANARY, SYSTEM))
    if og["blocked"]:
        text = "I can't share that."
    cites = [
        {
            "n": i + 1,
            "id": h.get("id"),
            "doc": h.get("doc"),
            "source": h.get("source"),
            "snippet": h.get("text", "")[:220],
        }
        for i, h in enumerate(ret["hits"])
    ]
    fin = tr.finish(
        200,
        tokens={"in": tin, "out": tout} if mode == "llm" else None,
        cost=cost_estimate(MODEL, tin, tout) if mode == "llm" else None,
    )
    return {
        **fin,
        "mode": mode,
        "retrieval": ret["mode"],
        "vector_error": ret["vector_error"],
        "model_error": model_error,
        "answer": text,
        "tool_calls": tool_calls,
        "citations": cites,
        "output_guard": og["rules"],
    }
