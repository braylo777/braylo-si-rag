//! Agent tools (function calling). All deterministic: no model, no side effects, no bookings created.
use crate::kb::SITE;
use regex::Regex;
use serde_json::{json, Map, Value};
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

const SRC: &str = "sources/site_index.txt";

fn re(p: &str) -> Regex {
    Regex::new(&p.replace(r"\b", r"(?-u:\b)")).unwrap()
}

pub fn tiers() -> Vec<Value> {
    vec![
        json!({"step": 1, "name": "Discoverability Audit", "es": "Auditoría de Visibilidad", "price": "Free", "tagline": "See where you're leaking.", "includes": ["Your monthly revenue leak, in dollars, and where it's coming from.", "A step-by-step fix plan your AI agents can follow."]}),
        json!({"step": 2, "name": "Agent Infrastructure Install", "price": "$497", "billing": "one-time", "tagline": "Lay a digital foundation.", "includes": ["Your own AI agent team, set up and linked to your accounts.", "Your Google, Facebook, X and Grok accounts set up, with your SEO fixed.", "Phone, email and security cleaned up, and you own all of it."]}),
        json!({"step": 3, "name": "Autonomous Ads Campaign", "plan": "Basic", "price": "$500/wk", "tagline": "Stay in front of buyers.", "includes": ["3 ready-to-post ads or clips every week, delivered to your Drive.", "Copy written for you, posted from your own ad accounts."]}),
        json!({"step": 4, "name": "Inbound Closing Engine", "plan": "Pro", "price": "$1,500/wk", "tagline": "Turn leads into booked jobs.", "includes": ["A voice AI agent that answers, qualifies and books every lead.", "Automatic follow-up so no lead goes cold."]}),
        json!({"step": 5, "name": "Subcontractor Coordination", "plan": "VIP", "price": "$2,000/wk", "tagline": "Step into a broker role.", "includes": ["Subs recruited, onboarded and scheduled for you.", "Our agents run day-to-day communication with your crews."]}),
    ]
}

pub const CHECKS: [(&str, &str); 7] = [
    (
        "has_website",
        "Website that loads on mobile and states your service area",
    ),
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
];

pub fn fold(s: &str) -> String {
    s.to_lowercase()
        .nfd()
        .filter(|c| !('\u{300}'..='\u{36f}').contains(c))
        .collect()
}

pub fn lookup_tier(query: &str) -> Value {
    static R: OnceLock<[Regex; 6]> = OnceLock::new();
    let r = R.get_or_init(|| {
        [
            re(r"\b(all|every|ladder|tiers|plans|pricing|prices|precios|planes)\b"),
            re(r"audit|auditoria|free|gratis"),
            re(r"install|497|instala"),
            re(r"\bads?\b|anuncio|basic"),
            re(r"voice|closing|pro\b|leads"),
            re(r"vip|sub(contractor)?s?\b|crew"),
        ]
    });
    let q = fold(query);
    let all = tiers();
    if r[0].is_match(&q) {
        return json!({"tiers": all, "source": SRC});
    }
    let hit: Vec<Value> = all
        .iter()
        .filter(|t| {
            let keys: Vec<String> = ["name", "es", "plan", "price"]
                .iter()
                .filter_map(|k| t.get(*k).and_then(|v| v.as_str()).map(String::from))
                .chain(std::iter::once(t["step"].to_string()))
                .collect();
            let step = t["step"].as_i64().unwrap();
            keys.iter().any(|k| q.contains(&fold(k)))
                || (r[1].is_match(&q) && step == 1)
                || (r[2].is_match(&q) && step == 2)
                || (r[3].is_match(&q) && step == 3)
                || (r[4].is_match(&q) && step == 4)
                || (r[5].is_match(&q) && step == 5)
        })
        .cloned()
        .collect();
    if !hit.is_empty() {
        return json!({"tiers": hit, "source": SRC});
    }
    json!({"tiers": [], "note": format!("No matching tier. Ask about: {}", all.iter().map(|t| t["name"].as_str().unwrap()).collect::<Vec<_>>().join(", ")), "source": SRC})
}

pub fn find_booking_slot(lang: &str) -> Value {
    json!({"url": format!("{}{}", SITE, if lang == "es" { "/book.es" } else { "/book" }), "note": "Pick an open time on the booking page. This assistant does not create bookings.", "source": "sources/site_book.txt"})
}

fn num(a: &Value, k: &str) -> Option<f64> {
    a.get(k)
        .and_then(|v| v.as_f64())
        .filter(|x| x.is_finite() && *x >= 0.0)
}
/// JS String(number) for the values the leak math prints.
fn js_num(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e21 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

pub fn audit_checklist(a: &Value) -> Value {
    let items: Vec<Value> = CHECKS.iter().map(|(k, label)| json!({"check": k, "label": label, "status": match a.get(*k) { Some(Value::Bool(true)) => "pass", Some(Value::Bool(false)) => "gap", _ => "not checked" }})).collect();
    let (calls, job, rate) = (
        num(a, "missed_calls_per_week"),
        num(a, "avg_job_value"),
        num(a, "close_rate"),
    );
    let leak = match (calls, job, rate) {
        (Some(c), Some(j), Some(r)) if r <= 1.0 => {
            json!({"monthly_usd": (c * r * j * 52.0 / 12.0 + 0.5).floor() as i64, "math": format!("{} missed calls/wk × {} close rate × ${} avg job × 52/12", js_num(c), js_num(r), js_num(j))})
        }
        _ => {
            json!({"monthly_usd": null, "math": "not checked: needs missed_calls_per_week, close_rate (0–1) and avg_job_value"})
        }
    };
    let gaps = items.iter().filter(|i| i["status"] == "gap").count();
    json!({"items": items, "gaps": gaps, "leak": leak, "next": format!("Book a free call to get the full audit: {SITE}/book"), "source": "sources/site_index.txt"})
}

/// Run a tool by name with JSON args; None if the tool does not exist.
pub fn run_tool(name: &str, args: &Value) -> Option<Value> {
    match name {
        "lookup_tier" => Some(lookup_tier(
            args.get("query").and_then(|q| q.as_str()).unwrap_or(""),
        )),
        "find_booking_slot" => Some(find_booking_slot(
            args.get("lang").and_then(|l| l.as_str()).unwrap_or("en"),
        )),
        "audit_checklist" => Some(audit_checklist(args)),
        _ => None,
    }
}

pub fn tool_schemas() -> Value {
    let mut props = Map::new();
    for (k, l) in CHECKS {
        props.insert(k.into(), json!({"type": "boolean", "description": l}));
    }
    props.insert("missed_calls_per_week".into(), json!({"type": "number"}));
    props.insert(
        "close_rate".into(),
        json!({"type": "number", "description": "0-1"}),
    );
    props.insert(
        "avg_job_value".into(),
        json!({"type": "number", "description": "USD"}),
    );
    json!([
        {"name": "lookup_tier", "description": "Look up a Braylo SI tier: name, price, tagline and what it includes.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Tier name, number, price, or 'all'"}}, "required": ["query"]}},
        {"name": "find_booking_slot", "description": "Return the link to the site's booking page where the user picks a time. Never books.", "parameters": {"type": "object", "properties": {"lang": {"type": "string", "enum": ["en", "es"]}}}},
        {"name": "audit_checklist", "description": "Run the free Discoverability Audit checklist and leak math from facts the owner gives.", "parameters": {"type": "object", "properties": Value::Object(props)}}
    ])
}

/// Deterministic router used when the model is unavailable (and as an eval baseline for model routing).
pub fn route_intent(q: &str) -> Option<Value> {
    static R: OnceLock<[Regex; 4]> = OnceLock::new();
    let r = R.get_or_init(|| [re(r"\b(book|booking|schedule|appointment|call with|pick a time|reservar|cita|agendar)\b"), re(r"reservar|cita|agendar"), re(r"\b(checklist|audit me|run (the|an|my)? ?audit|check my|am i leaking|leak math)\b"), re(r"\b(price|cost|how much|tier|plan|pricing|\$|precio|cuanto|cuesta)\b")]);
    let s = fold(q);
    if r[0].is_match(&s) {
        return Some(
            json!({"tool": "find_booking_slot", "args": {"lang": if r[1].is_match(&s) { "es" } else { "en" }}}),
        );
    }
    if r[2].is_match(&s) {
        return Some(json!({"tool": "audit_checklist", "args": {}}));
    }
    if r[3].is_match(&s) {
        return Some(json!({"tool": "lookup_tier", "args": {"query": q}}));
    }
    None
}
