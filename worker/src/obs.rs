//! Observability: trace IDs, spans, token counting, published-price cost estimate, percentiles, KV trace store.
//! Pricing source: developers.cloudflare.com/workers-ai/platform/pricing (read Oct 27, 2026). ESTIMATE only.
use serde_json::{json, Map, Value};

pub const MODEL: &str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
pub const IDX: &str = "idx";
pub const MAX: usize = 200;

pub fn pricing() -> Value {
    json!({MODEL: {"in_neurons_per_m": 26668, "out_neurons_per_m": 204805}, "usd_per_1k_neurons": 0.011})
}

/// JS Number(x.toFixed(d)): exact decimal expansion, ties round up.
pub fn to_fixed(x: f64, d: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let neg = x < 0.0;
    let s = format!("{:.60}", x.abs());
    let (int, frac) = s.split_once('.').unwrap();
    let mut digits: Vec<u8> = int.bytes().chain(frac.bytes().take(d)).map(|b| b - b'0').collect();
    let next = frac.as_bytes().get(d).map_or(0, |b| b - b'0');
    if next >= 5 {
        let mut i = digits.len();
        loop {
            if i == 0 { digits.insert(0, 1); break; }
            i -= 1;
            if digits[i] == 9 { digits[i] = 0 } else { digits[i] += 1; break; }
        }
    }
    let n = digits.len();
    let (a, b) = digits.split_at(n - d);
    let txt = format!("{}{}.{}", if neg { "-" } else { "" }, a.iter().map(|x| x.to_string()).collect::<String>(), b.iter().map(|x| x.to_string()).collect::<String>());
    txt.trim_end_matches('.').parse().unwrap_or(x)
}
pub fn round2(x: f64) -> f64 {
    (x * 100.0 + 0.5).floor() / 100.0
}

pub struct Tracer<'a> {
    pub route: String,
    pub trace_id: String,
    now: &'a dyn Fn() -> f64,
    epoch: i64,
    t0: f64,
    pub spans: Vec<Value>,
}
impl<'a> Tracer<'a> {
    pub fn new(route: &str, now: &'a dyn Fn() -> f64, trace_id: String, epoch: i64) -> Self {
        Tracer { route: route.into(), trace_id, now, epoch, t0: now(), spans: vec![] }
    }
    pub fn start(&self) -> (f64, f64) {
        let start = round2((self.now)() - self.t0);
        (start, start)
    }
    /// Close a span begun with `start()`; pushes it like the JS `finally`.
    pub fn end(&mut self, name: &str, start: f64, attrs: Value, error: Option<String>) {
        let duration = round2((self.now)() - self.t0 - start);
        let mut s = json!({"name": name, "start_ms": start, "attrs": attrs});
        if let Some(e) = error {
            s["error"] = json!(e);
        }
        s["duration_ms"] = json!(duration);
        self.spans.push(s);
    }
    pub fn finish(&self, status: u16, extra: Value) -> Value {
        let mut o = Map::new();
        o.insert("trace_id".into(), json!(self.trace_id));
        o.insert("route".into(), json!(self.route));
        o.insert("status".into(), json!(status));
        o.insert("ts".into(), json!(self.epoch));
        o.insert("duration_ms".into(), json!(round2((self.now)() - self.t0)));
        o.insert("spans".into(), Value::Array(self.spans.clone()));
        if let Some(e) = extra.as_object() {
            for (k, v) in e {
                o.insert(k.clone(), v.clone());
            }
        }
        Value::Object(o)
    }
}

/// Rough token estimate (~4 chars/token) used only when the model returns no usage block.
pub fn est_tokens(s: &str) -> i64 {
    (s.encode_utf16().count() as f64 / 4.0).ceil() as i64
}

pub fn cost_estimate(model: &str, tin: f64, tout: f64) -> Value {
    let p = pricing();
    let Some(m) = p.get(model).filter(|v| v.is_object()) else { return Value::Null };
    let neurons = (tin * m["in_neurons_per_m"].as_f64().unwrap() + tout * m["out_neurons_per_m"].as_f64().unwrap()) / 1e6;
    json!({"neurons": to_fixed(neurons, 3), "usd": to_fixed(neurons / 1000.0 * 0.011, 8), "label": "estimate from published neuron pricing; free tier pays $0"})
}

pub fn pct(xs: &[f64], p: f64) -> Value {
    if xs.is_empty() {
        return Value::Null;
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    json!(s[(s.len() - 1).min(((p / 100.0 * s.len() as f64).ceil() as usize).saturating_sub(1))])
}

pub fn summarize(traces: &[Value]) -> Value {
    let mut order: Vec<String> = vec![];
    let mut by: std::collections::HashMap<String, Vec<&Value>> = Default::default();
    for t in traces {
        let r = t["route"].as_str().unwrap_or("").to_string();
        if !by.contains_key(&r) {
            order.push(r.clone());
        }
        by.entry(r).or_default().push(t);
    }
    let mut out = Map::new();
    for r in order {
        let ts = &by[&r];
        let d: Vec<f64> = ts.iter().map(|t| t["duration_ms"].as_f64().unwrap_or(0.0)).collect();
        let ok = ts.iter().filter(|t| t["status"].as_i64().unwrap_or(0) < 400).count();
        let cost: f64 = ts.iter().map(|t| t["cost"]["usd"].as_f64().unwrap_or(0.0)).sum();
        let tok: Vec<&&Value> = ts.iter().filter(|t| t.get("tokens").map_or(false, |x| !x.is_null())).collect();
        let avg = |k: &str| -> Value { if tok.is_empty() { Value::Null } else { json!(to_fixed(tok.iter().map(|t| t["tokens"][k].as_f64().unwrap_or(0.0)).sum::<f64>() / tok.len() as f64, 1)) } };
        out.insert(r, json!({"n": ts.len(), "errors": ts.len() - ok, "p50_ms": pct(&d, 50.0), "p95_ms": pct(&d, 95.0), "avg_tokens_in": avg("in"), "avg_tokens_out": avg("out"),
            "est_cost_usd_per_req": if tok.is_empty() { Value::Null } else { json!(to_fixed(cost / tok.len() as f64, 8)) }}));
    }
    Value::Object(out)
}

/// Key-value store for traces (the TRACES binding).
#[allow(async_fn_in_trait)]
pub trait Kv {
    async fn get(&self, key: &str) -> Option<String>;
    async fn put(&self, key: &str, val: &str);
    async fn delete(&self, key: &str);
}

pub async fn store<K: Kv>(kv: &K, trace: &Value) {
    let id = trace["trace_id"].as_str().unwrap_or("").to_string();
    kv.put(&format!("t:{id}"), &trace.to_string()).await;
    let mut cur: Vec<String> = kv.get(IDX).await.and_then(|r| serde_json::from_str(&r).ok()).unwrap_or_default();
    cur.insert(0, id);
    let mut seen = std::collections::HashSet::new();
    let next: Vec<String> = cur.iter().filter(|i| seen.insert((*i).clone())).take(MAX).cloned().collect();
    kv.put(IDX, &json!(next).to_string()).await;
    for drop in cur.iter().skip(MAX) {
        kv.delete(&format!("t:{drop}")).await;
    }
}

pub async fn recent<K: Kv>(kv: &K) -> Vec<Value> {
    let ids: Vec<String> = kv.get(IDX).await.and_then(|r| serde_json::from_str(&r).ok()).unwrap_or_default();
    let mut rows = vec![];
    for i in ids {
        if let Some(r) = kv.get(&format!("t:{i}")).await.and_then(|r| serde_json::from_str::<Value>(&r).ok()) {
            rows.push(r);
        }
    }
    rows
}
