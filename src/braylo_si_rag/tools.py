"""Agent tools (function calling). All deterministic: no model, no side effects, no bookings created."""

import math, re, unicodedata

from .kb import SITE

A = re.ASCII
# Source of truth: How We Help section of the site preview (worker v22v).
TIERS = [
    {
        "step": 1,
        "name": "Discoverability Audit",
        "es": "Auditoría de Visibilidad",
        "price": "Free",
        "tagline": "See where you're leaking.",
        "includes": [
            "Your monthly revenue leak, in dollars, and where it's coming from.",
            "A step-by-step fix plan your AI agents can follow.",
        ],
    },
    {
        "step": 2,
        "name": "Agent Infrastructure Install",
        "price": "$497",
        "billing": "one-time",
        "tagline": "Lay a digital foundation.",
        "includes": [
            "Your own AI agent team, set up and linked to your accounts.",
            "Your Google, Facebook, X and Grok accounts set up, with your SEO fixed.",
            "Phone, email and security cleaned up, and you own all of it.",
        ],
    },
    {
        "step": 3,
        "name": "Autonomous Ads Campaign",
        "plan": "Basic",
        "price": "$500/wk",
        "tagline": "Stay in front of buyers.",
        "includes": [
            "3 ready-to-post ads or clips every week, delivered to your Drive.",
            "Copy written for you, posted from your own ad accounts.",
        ],
    },
    {
        "step": 4,
        "name": "Inbound Closing Engine",
        "plan": "Pro",
        "price": "$1,500/wk",
        "tagline": "Turn leads into booked jobs.",
        "includes": [
            "A voice AI agent that answers, qualifies and books every lead.",
            "Automatic follow-up so no lead goes cold.",
        ],
    },
    {
        "step": 5,
        "name": "Subcontractor Coordination",
        "plan": "VIP",
        "price": "$2,000/wk",
        "tagline": "Step into a broker role.",
        "includes": [
            "Subs recruited, onboarded and scheduled for you.",
            "Our agents run day-to-day communication with your crews.",
        ],
    },
]
SRC = "sources/site_index.txt"
CHECKS = [
    ("has_website", "Website that loads on mobile and states your service area"),
    (
        "google_profile",
        "Google Business Profile claimed, with hours, photos and services",
    ),
    ("reviews_30plus", "30+ recent reviews, with replies"),
    (
        "listings_consistent",
        "Same name, address and phone on Google, Facebook, Yelp and directories",
    ),
    (
        "answers_missed_calls",
        "Missed calls get an answer or a text back within 5 minutes",
    ),
    ("fast_web_reply", "Web leads get a reply within 5 minutes"),
    ("quote_followup", "Every open quote gets a follow-up"),
]


def fold(s):
    s = "" if s is None else str(s)
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower()) if not "̀" <= c <= "ͯ"
    )


def _r(p):
    return re.compile(p, A)


_ALL = _r(r"\b(all|every|ladder|tiers|plans|pricing|prices|precios|planes)\b")
_AUDIT, _INSTALL, _ADS = (
    _r(r"audit|auditoria|free|gratis"),
    _r(r"install|497|instala"),
    _r(r"\bads?\b|anuncio|basic"),
)
_VOICE, _VIP = _r(r"voice|closing|pro\b|leads"), _r(r"vip|sub(contractor)?s?\b|crew")


def lookup_tier(query=""):
    q = fold(query)
    if _ALL.search(q):
        return {"tiers": TIERS, "source": SRC}
    hit = []
    for t in TIERS:
        keys = [
            k
            for k in (t["name"], t.get("es"), t.get("plan"), t["price"], str(t["step"]))
            if k
        ]
        if (
            any(fold(k) in q for k in keys)
            or (_AUDIT.search(q) and t["step"] == 1)
            or (_INSTALL.search(q) and t["step"] == 2)
            or (_ADS.search(q) and t["step"] == 3)
            or (_VOICE.search(q) and t["step"] == 4)
            or (_VIP.search(q) and t["step"] == 5)
        ):
            hit.append(t)
    if hit:
        return {"tiers": hit, "source": SRC}
    return {
        "tiers": [],
        "note": "No matching tier. Ask about: " + ", ".join(t["name"] for t in TIERS),
        "source": SRC,
    }


def find_booking_slot(lang="en"):
    url = SITE + ("/book.es" if lang == "es" else "/book")
    return {
        "url": url,
        "note": "Pick an open time on the booking page. This assistant does not create bookings.",
        "source": "sources/site_book.txt",
    }


def _n(a, k):
    v = a.get(k)
    return (
        v
        if isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
        and v >= 0
        else None
    )


def js_num(v):
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e21:
        return str(int(v))
    return str(v) if not isinstance(v, float) else repr(v)


def audit_checklist(a=None):
    """Free Discoverability Audit checklist. Leak math only uses numbers the owner gives; unknowns are "not checked" (never invent numbers)."""
    a = a or {}
    items = [
        {
            "check": k,
            "label": label,
            "status": (
                "pass"
                if a.get(k) is True
                else "gap" if a.get(k) is False else "not checked"
            ),
        }
        for k, label in CHECKS
    ]
    calls, job, rate = (
        _n(a, "missed_calls_per_week"),
        _n(a, "avg_job_value"),
        _n(a, "close_rate"),
    )
    if calls is not None and job is not None and rate is not None and rate <= 1:
        leak = {
            "monthly_usd": math.floor(calls * rate * job * 52 / 12 + 0.5),
            "math": f"{js_num(calls)} missed calls/wk × {js_num(rate)} close rate × ${js_num(job)} avg job × 52/12",
        }
    else:
        leak = {
            "monthly_usd": None,
            "math": "not checked: needs missed_calls_per_week, close_rate (0–1) and avg_job_value",
        }
    return {
        "items": items,
        "gaps": sum(1 for i in items if i["status"] == "gap"),
        "leak": leak,
        "next": "Book a free call to get the full audit: " + SITE + "/book",
        "source": "sources/site_index.txt",
    }


def _call(fn, args):
    args = args if isinstance(args, dict) else {}
    return fn(
        **{
            k: v
            for k, v in args.items()
            if k in fn.__code__.co_varnames[: fn.__code__.co_argcount]
        }
    )


TOOLS = {
    "lookup_tier": lambda a: _call(lookup_tier, a),
    "find_booking_slot": lambda a: _call(find_booking_slot, a),
    "audit_checklist": lambda a: audit_checklist(a if isinstance(a, dict) else {}),
}
_bool_props = {k: {"type": "boolean", "description": l} for k, l in CHECKS}
TOOL_SCHEMAS = [
    {
        "name": "lookup_tier",
        "description": "Look up a Braylo SI tier: name, price, tagline and what it includes.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Tier name, number, price, or 'all'",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_booking_slot",
        "description": "Return the link to the site's booking page where the user picks a time. Never books.",
        "parameters": {
            "type": "object",
            "properties": {"lang": {"type": "string", "enum": ["en", "es"]}},
        },
    },
    {
        "name": "audit_checklist",
        "description": "Run the free Discoverability Audit checklist and leak math from facts the owner gives.",
        "parameters": {
            "type": "object",
            "properties": {
                **_bool_props,
                "missed_calls_per_week": {"type": "number"},
                "close_rate": {"type": "number", "description": "0-1"},
                "avg_job_value": {"type": "number", "description": "USD"},
            },
        },
    },
]
_BOOK = _r(
    r"\b(book|booking|schedule|appointment|call with|pick a time|reservar|cita|agendar)\b"
)
_ES = _r(r"reservar|cita|agendar")
_CHECK = _r(
    r"\b(checklist|audit me|run (the|an|my)? ?audit|check my|am i leaking|leak math)\b"
)
_PRICE = _r(r"\b(price|cost|how much|tier|plan|pricing|\$|precio|cuanto|cuesta)\b")


def route_intent(q):
    """Deterministic router used when the model is unavailable (and as an eval baseline for model routing)."""
    s = fold(q)
    if _BOOK.search(s):
        return {
            "tool": "find_booking_slot",
            "args": {"lang": "es" if _ES.search(s) else "en"},
        }
    if _CHECK.search(s):
        return {"tool": "audit_checklist", "args": {}}
    if _PRICE.search(s):
        return {"tool": "lookup_tier", "args": {"query": q}}
    return None
