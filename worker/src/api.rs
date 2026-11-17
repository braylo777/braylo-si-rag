//! HTTP routes. Pure router: returns (status, json, extra headers); status 0 = serve the page.
use crate::agent::{answer, hybrid, Host};
use crate::kb::{cap, chunks};
use crate::obs::{recent, store, summarize};
use crate::tools::{run_tool, tool_schemas};
use serde_json::{json, Value};
use std::cell::RefCell;

const RESULTS_JSON: &str = include_str!("../data/results.json");

thread_local! { pub static RING: RefCell<Vec<Value>> = RefCell::new(vec![]); } // per-isolate fallback when no KV is bound

fn query(q: &str, key: &str) -> Option<String> {
    regex::Regex::new(&format!(r"(?:^|&){}=([^&]*)", regex::escape(key))).unwrap().captures(q).map(|c| c[1].replace('+', " "))
}

pub async fn handle<H: Host>(h: &H, method: &str, path: &str, qs: &str, body: &Value) -> (u16, Value, Vec<(&'static str, String)>) {
    let none = vec![];
    if path == "/" {
        return (0, Value::Null, none);
    }
    if path == "/api/health" {
        return (200, json!({"ok": true, "chunks": chunks().len(), "ai": h.has_ai(), "vectors": h.has_vectors()}), none);
    }
    if path == "/api/tools" {
        return (200, tool_schemas(), none);
    }
    if path == "/api/eval" {
        return (200, serde_json::from_str(RESULTS_JSON).unwrap(), none);
    }
    if path == "/api/metrics" {
        let t = if h.has_kv() { recent(h).await } else { RING.with(|r| r.borrow().clone()) };
        return (200, json!({"n": t.len(), "by_route": summarize(&t), "store": if h.has_kv() { "kv" } else { "isolate memory" }}), none);
    }
    if path == "/api/search" {
        let q = query(qs, "q").unwrap_or_default();
        if !cap(Some(&q), 600) {
            return (400, json!({"error": "q required (<=600 chars)"}), none);
        }
        let r = hybrid(h, &q, 5).await;
        return (200, json!({"mode": r.mode, "vector_error": r.vector_error, "hits": r.hits}), none);
    }
    if method != "POST" {
        return (404, json!({"error": "not found"}), none);
    }
    if h.limited().await {
        return (429, json!({"error": "rate limited: 5 requests/min"}), none);
    }
    if path == "/api/tool" {
        let name = body.get("name").and_then(|n| n.as_str()).unwrap_or("");
        let args = body.get("args").filter(|a| !a.is_null() && **a != json!(false)).cloned().unwrap_or(json!({}));
        return match run_tool(name, &args) {
            Some(v) => (200, v, none),
            None => (400, json!({"error": "unknown tool"}), none),
        };
    }
    if path == "/api/chat" {
        let m = body.get("message").and_then(|m| m.as_str());
        if !cap(m, 600) {
            return (400, json!({"error": "message required (<=600 chars)"}), none);
        }
        let r = answer(h, m.unwrap(), false).await;
        let trace: Value = ["trace_id", "route", "status", "ts", "duration_ms", "spans", "tokens", "cost"].iter().map(|k| (k.to_string(), r.get(*k).cloned().unwrap_or(Value::Null))).collect::<serde_json::Map<_, _>>().into();
        if h.has_kv() {
            store(h, &trace).await;
        } else {
            RING.with(|ring| {
                let mut v = ring.borrow_mut();
                v.insert(0, trace);
                v.truncate(200);
            });
        }
        let status = r["status"].as_u64().unwrap_or(200) as u16;
        let tid = r["trace_id"].as_str().unwrap_or("").to_string();
        return (status, r, vec![("x-trace-id", tid)]);
    }
    (404, json!({"error": "not found"}), none)
}
