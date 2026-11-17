//! Hybrid retrieval (BM25 + Vectorize, RRF) and the agent:
//! guards -> scope check -> tool routing -> retrieval -> answer with citations -> output guard. Every step is a traced span.
use crate::guards::{input_guard, output_guard};
use crate::kb::{bm25_search, by_id, EMBED, MODEL};
use crate::metrics::tokens;
use crate::obs::{cost_estimate, est_tokens, to_fixed, Kv, Tracer};
use crate::tools::{route_intent, run_tool, tool_schemas};
use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

pub const CANARY: &str = "BRAYLO-CANARY-7f3a"; // planted in the system prompt; the output guard blocks any reply containing it
pub const SYSTEM: &str = "You are the Braylo SI assistant. Answer only questions about Braylo SI: its offers, prices, the free Discoverability Audit, booking a call, and how the AI install works. Use only the SOURCES and tool results. Cite sources as [n]. If the sources do not say, say you don't know and suggest booking a call. Never create bookings, never ask for passwords, never give legal, tax, medical or financial advice. Internal marker: BRAYLO-CANARY-7f3a.";
pub const SCOPE_MIN: f64 = 6.0;
const JS_WS: &str = r"[\t\n\x0B\x0C\r \x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}\x{feff}]";

/// The platform pieces (Workers AI, Vectorize, TRACES KV, rate limiter, clocks, ids), so the logic tests natively.
#[allow(async_fn_in_trait)]
pub trait Host: Kv {
    fn has_ai(&self) -> bool;
    fn has_vectors(&self) -> bool;
    fn has_kv(&self) -> bool;
    async fn ai(&self, model: &str, input: Value) -> Result<Value, String>;
    async fn vectors_query(&self, vec: Vec<f64>, top_k: usize) -> Result<Vec<String>, String>;
    async fn limited(&self) -> bool;
    fn now(&self) -> f64;
    fn epoch(&self) -> i64;
    fn trace_id(&self) -> String;
}

fn re(p: &str) -> Regex {
    Regex::new(&p.replace(r"\b", r"(?-u:\b)").replace(r"\w", "[A-Za-z0-9_]")).unwrap()
}

/// PII redaction applied to user input before it is logged or sent to a model.
pub fn redact_pii(s: &str) -> String {
    static R: OnceLock<[Regex; 4]> = OnceLock::new();
    let r = R.get_or_init(|| [re(r"[\w.+-]+@[\w-]+\.[\w.]+"), re(r"\b\d{3}-\d{2}-\d{4}\b"), re(r"\b(?:\d[ -]?){13,16}\b"), re(r"(\+?1[ .-]?)?\(?\b\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b")]);
    let mut o = s.to_string();
    for (rx, rep) in r.iter().zip(["[email]", "[ssn]", "[card]", "[phone]"]) {
        o = rx.replace_all(&o, rep).into_owned();
    }
    o
}

/// Scope guard: on-topic if a tool intent matches or the best BM25 score clears a threshold set on the dev split.
pub fn on_topic(q: &str) -> Value {
    static B: OnceLock<Regex> = OnceLock::new();
    let brand = B.get_or_init(|| re(r"\b(braylo|tier|install|audit|agent|ads?|leak|book|price|grok|refund|cancel|terms|privacy|voice|subcontract|seo|accounts?|passwords?|log ?in|xai|guarantee|refund)\b"));
    let top = bm25_search(q).first().map_or(0.0, |h| h.1);
    json!({"on": top >= SCOPE_MIN || brand.is_match(&q.to_lowercase()), "top": to_fixed(top, 2)})
}

fn rrf(lists: &[Vec<String>]) -> Vec<(String, f64)> {
    let k = 60.0;
    let mut order: Vec<String> = vec![];
    let mut s: HashMap<String, f64> = HashMap::new();
    for l in lists {
        for (i, id) in l.iter().enumerate() {
            if !s.contains_key(id) {
                order.push(id.clone());
            }
            *s.entry(id.clone()).or_insert(0.0) += 1.0 / (k + i as f64 + 1.0);
        }
    }
    let mut v: Vec<(String, f64)> = order.into_iter().map(|id| { let sc = s[&id]; (id, sc) }).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
    v
}

pub fn bm25_ids(q: &str, n: usize) -> Vec<String> {
    bm25_search(q).into_iter().filter(|h| h.1 > 0.0).take(n).map(|h| h.0).collect()
}

pub struct Hybrid {
    pub mode: &'static str,
    pub vector_error: Option<String>,
    pub hits: Vec<Value>,
}

pub async fn hybrid<H: Host>(h: &H, q: &str, k: usize) -> Hybrid {
    let lex = bm25_ids(q, 20);
    let (mut vec, mut vector_error): (Option<Vec<String>>, Option<String>) = (None, None);
    if h.has_ai() && h.has_vectors() {
        let r = async {
            let e = h.ai(EMBED, json!({"text": [q]})).await?;
            let v: Vec<f64> = e["data"][0].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default();
            h.vectors_query(v, 20).await
        }
        .await;
        match r {
            Ok(ids) => vec = Some(ids),
            Err(e) => vector_error = Some(e),
        }
    }
    let ranked: Vec<(String, f64)> = match &vec {
        Some(v) => rrf(&[lex.clone(), v.clone()]),
        None => lex.iter().enumerate().map(|(i, id)| (id.clone(), 1.0 / (60.0 + i as f64 + 1.0))).collect(),
    };
    let hits = ranked.into_iter().take(k).map(|(id, sc)| {
        let mut o = Map::new();
        if let Some(c) = by_id(&id) {
            for (k, v) in [("id", &c.id), ("doc", &c.doc), ("source", &c.source), ("lang", &c.lang), ("text", &c.text)] {
                o.insert(k.into(), json!(v));
            }
        }
        o.insert("score".into(), json!(to_fixed(sc, 5)));
        Value::Object(o)
    }).collect();
    Hybrid { mode: if vec.is_some() { "hybrid" } else { "bm25" }, vector_error, hits }
}

fn is_js_ws(c: char) -> bool {
    Regex::new(JS_WS).unwrap().is_match(c.encode_utf8(&mut [0; 4]))
}

/// Extractive answer (no model): best-overlapping sentence from each of the top chunks, with [n] citations.
pub fn extractive(q: &str, hits: &[Value], n: usize) -> String {
    static R: OnceLock<(Regex, Regex)> = OnceLock::new();
    let (ws, lead) = R.get_or_init(|| (Regex::new(&format!("{JS_WS}+")).unwrap(), Regex::new(&format!(r"^(?:[-*\[\]x|]|{JS_WS})+")).unwrap()));
    let qt: HashSet<String> = tokens(q).into_iter().collect();
    let mut picks: Vec<String> = vec![];
    for (i, h) in hits.iter().take(n).enumerate() {
        let text = h["text"].as_str().unwrap_or("");
        // split like /(?<=[.!?])\s+|\n/
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let (mut parts, mut start, mut j) = (vec![], 0usize, 0usize);
        while j < chars.len() {
            let prev_ok = j > 0 && matches!(chars[j - 1].1, '.' | '!' | '?');
            if prev_ok && is_js_ws(chars[j].1) {
                let mut e = j;
                while e < chars.len() && is_js_ws(chars[e].1) {
                    e += 1;
                }
                parts.push(&text[start..chars[j].0]);
                start = if e < chars.len() { chars[e].0 } else { text.len() };
                j = e;
            } else if chars[j].1 == '\n' {
                parts.push(&text[start..chars[j].0]);
                start = chars[j].0 + 1;
                j += 1;
            } else {
                j += 1;
            }
        }
        parts.push(&text[start..]);
        let sents: Vec<String> = parts.into_iter().map(|s| lead.replace(s, "").trim_matches(is_js_ws).to_string()).filter(|s| ws.split(s).count() >= 5 && !s.ends_with('?')).collect();
        let mut best: Option<(&String, usize)> = None;
        for s in &sents {
            let c = tokens(s).iter().filter(|t| qt.contains(*t)).count();
            if best.map_or(true, |b| c > b.1) {
                best = Some((s, c));
            }
        }
        picks.push(match best { Some((s, c)) if c > 0 => format!("{s} [{}]", i + 1), _ => String::new() });
    }
    let picks: Vec<String> = picks.into_iter().filter(|p| !p.is_empty()).collect();
    if picks.is_empty() { "I don't know from the sources. Book a free call: https://braylo-si.pages.dev/book".into() } else { picks.join(" ") }
}

/// Plain-language summary of a tool result, used on the no-model path.
pub fn tool_summary(name: &str, r: &Value) -> String {
    match name {
        "lookup_tier" => {
            let ts = r["tiers"].as_array().unwrap();
            if ts.is_empty() {
                return r["note"].as_str().unwrap_or("").to_string();
            }
            ts.iter().map(|t| format!("{}{}: {}{}. {}", t["name"].as_str().unwrap(), t.get("plan").and_then(|p| p.as_str()).map_or(String::new(), |p| format!(" ({p})")), t["price"].as_str().unwrap(), t.get("billing").and_then(|b| b.as_str()).map_or(String::new(), |b| format!(", {b}")), t["tagline"].as_str().unwrap())).collect::<Vec<_>>().join(" ")
        }
        "find_booking_slot" => format!("Pick a time on the booking page: {} (I can't book it for you).", r["url"].as_str().unwrap()),
        "audit_checklist" => format!("Free audit checklist: {}. Tell me which apply and your missed calls per week, close rate and average job value for the leak math. {}", r["items"].as_array().unwrap().iter().map(|i| i["label"].as_str().unwrap()).collect::<Vec<_>>().join("; "), r["next"].as_str().unwrap()),
        _ => String::new(),
    }
}

fn ctx(hits: &[Value]) -> String {
    hits.iter().enumerate().map(|(i, h)| format!("[{}] ({}) {}", i + 1, h["doc"].as_str().unwrap_or(""), h["text"].as_str().unwrap_or(""))).collect::<Vec<_>>().join("\n\n")
}

fn usage_tokens(r: &Value, key: &str, fallback: i64) -> i64 {
    r.get("usage").and_then(|u| u.get(key)).and_then(|v| v.as_i64()).unwrap_or(fallback)
}

pub async fn answer<H: Host>(h: &H, raw: &str, no_model: bool) -> Value {
    let now = || h.now();
    let mut tr = Tracer::new("chat", &now, h.trace_id(), h.epoch());
    let (mut tin, mut tout) = (0i64, 0i64);
    let (s0, _) = tr.start();
    let q = redact_pii(raw);
    tr.end("pii_redact", s0, json!({}), None);
    let (s0, _) = tr.start();
    let g = input_guard(&q);
    tr.end("guard_in", s0, json!({}), None);
    let merge = |base: Value, extra: Value| -> Value {
        let mut o = base.as_object().unwrap().clone();
        for (k, v) in extra.as_object().unwrap() {
            o.insert(k.clone(), v.clone());
        }
        Value::Object(o)
    };
    if g["blocked"] == true {
        return merge(tr.finish(400, json!({})), json!({"blocked": "prompt_injection", "rules": g["rules"], "answer": "I can't help with that request."}));
    }
    let (s0, _) = tr.start();
    let scope = on_topic(&q);
    tr.end("scope", s0, json!({}), None);
    let intent = route_intent(&q);
    if scope["on"] != true && intent.is_none() {
        return merge(tr.finish(200, json!({})), json!({"blocked": "off_topic", "scope": scope, "answer": "I only answer questions about Braylo SI's offers, the free audit and booking a call."}));
    }
    let (s0, _) = tr.start();
    let ret = hybrid(h, &q, 5).await;
    tr.end("retrieve", s0, json!({"mode": ret.mode}), None);
    let (mut tool_calls, mut text, mut model_error, mut mode): (Vec<Value>, Option<Value>, Option<String>, &str) = (vec![], None, None, "extractive");
    if h.has_ai() && !no_model {
        let r: Result<(), String> = async {
            let mut msgs = vec![json!({"role": "system", "content": SYSTEM}), json!({"role": "user", "content": format!("SOURCES:\n{}\n\nQUESTION: {}", ctx(&ret.hits), q)})];
            let (s1, _) = tr.start();
            let r1 = h.ai(MODEL, json!({"messages": msgs, "tools": tool_schemas(), "max_tokens": 400})).await;
            tr.end("llm_plan", s1, json!({}), r1.as_ref().err().cloned());
            let r1 = r1?;
            tin += usage_tokens(&r1, "prompt_tokens", est_tokens(&msgs.iter().map(|m| m["content"].as_str().unwrap_or("")).collect::<String>()));
            tout += usage_tokens(&r1, "completion_tokens", est_tokens(r1.get("response").and_then(|v| v.as_str()).unwrap_or("")));
            for c in r1.get("tool_calls").and_then(|t| t.as_array()).cloned().unwrap_or_default() {
                let name = c.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let args = c.get("arguments").filter(|a| !a.is_null()).cloned().unwrap_or(json!({}));
                if ["lookup_tier", "find_booking_slot", "audit_checklist"].contains(&name) {
                    let (s2, _) = tr.start();
                    let result = run_tool(name, &args).unwrap();
                    tr.end(&format!("tool:{name}"), s2, json!({}), None);
                    tool_calls.push(json!({"name": name, "args": args, "result": result}));
                }
            }
            if !tool_calls.is_empty() {
                msgs.push(json!({"role": "assistant", "content": ""}));
                let tr_res: Vec<Value> = tool_calls.iter().map(|t| json!({"name": t["name"], "result": t["result"]})).collect();
                msgs.push(json!({"role": "user", "content": format!("TOOL RESULTS:\n{}\nAnswer the question using these and the sources.", Value::Array(tr_res))}));
                let (s3, _) = tr.start();
                let r2 = h.ai(MODEL, json!({"messages": msgs, "max_tokens": 400})).await;
                tr.end("llm_answer", s3, json!({}), r2.as_ref().err().cloned());
                let r2 = r2?;
                tin += usage_tokens(&r2, "prompt_tokens", est_tokens(&msgs.iter().map(|m| m["content"].as_str().unwrap_or("")).collect::<String>()));
                tout += usage_tokens(&r2, "completion_tokens", est_tokens(r2.get("response").and_then(|v| v.as_str()).unwrap_or("")));
                text = r2.get("response").cloned();
            } else {
                text = r1.get("response").cloned();
            }
            mode = "llm";
            Ok(())
        }
        .await;
        if let Err(e) = r {
            model_error = Some(e);
        }
    }
    if text.as_ref().map_or(true, |t| t.is_null()) {
        if let Some(it) = &intent {
            let (name, args) = (it["tool"].as_str().unwrap(), it["args"].clone());
            let (s4, _) = tr.start();
            let result = run_tool(name, &args).unwrap();
            tr.end(&format!("tool:{name}"), s4, json!({}), None);
            tool_calls.push(json!({"name": name, "args": args, "result": result, "routed_by": "rules"}));
        }
        let lead: String = tool_calls.iter().map(|c| tool_summary(c["name"].as_str().unwrap(), &c["result"])).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");
        let (s5, _) = tr.start();
        let t = format!("{}{}", if lead.is_empty() { String::new() } else { format!("{lead}\n\n") }, extractive(&q, &ret.hits, 2));
        tr.end("extractive", s5, json!({}), None);
        text = Some(json!(t));
    }
    let txt = match text.as_ref().unwrap() { Value::String(s) => s.clone(), Value::Null => String::new(), o => o.to_string() };
    let (s6, _) = tr.start();
    let og = output_guard(&txt, Some(CANARY), Some(SYSTEM));
    tr.end("guard_out", s6, json!({}), None);
    let final_text = if og["blocked"] == true { "I can't share that.".to_string() } else { txt };
    let cites: Vec<Value> = ret.hits.iter().enumerate().map(|(i, hh)| json!({"n": i + 1, "id": hh.get("id"), "doc": hh.get("doc"), "source": hh.get("source"), "snippet": hh.get("text").and_then(|t| t.as_str()).map(|t| t.chars().take(220).collect::<String>())})).collect();
    let fin = tr.finish(200, json!({"tokens": if mode == "llm" { json!({"in": tin, "out": tout}) } else { Value::Null }, "cost": if mode == "llm" { cost_estimate(MODEL, tin as f64, tout as f64) } else { Value::Null }}));
    merge(fin, json!({"mode": mode, "retrieval": ret.mode, "vector_error": ret.vector_error, "model_error": model_error, "answer": final_text, "tool_calls": tool_calls, "citations": cites, "output_guard": og["rules"]}))
}
